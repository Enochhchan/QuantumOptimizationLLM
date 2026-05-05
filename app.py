from __future__ import annotations

from flask import Flask, send_file, request, jsonify
import ast
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import sys
import threading
import time
import webbrowser
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional
from uuid import uuid4

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# ---- Prometheus Metrics ----
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests received",
    ["method", "endpoint"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
)

BASE_DIR = Path(__file__).resolve().parent
LEGACY_VALIDATOR_PATH = BASE_DIR / "legacy" / "py" / "qubo_validator.py"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv, find_dotenv

        # Local dev expects values in `.env` to be authoritative.
        load_dotenv(find_dotenv(), override=True)
    except Exception:
        # Optional dependency. If unavailable, rely on process env vars.
        pass


_load_dotenv_if_available()

def _default_demo_mode() -> bool:
    # EXE users expect the real backend by default; local dev keeps demo fallback on.
    return not getattr(sys, "frozen", False)


DEMO_MODE = _env_flag("DEMO_MODE", _default_demo_mode())
USE_LEGACY_BACKEND = _env_flag("USE_LEGACY_BACKEND", True)
OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or os.getenv("LEGACY_OPENAI_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini"
MAX_SOLUTION_PREVIEW = max(5, int(os.getenv("MAX_SOLUTION_PREVIEW", "24")))
USE_REFERENCE_HINTS = _env_flag("USE_REFERENCE_HINTS", True)
ALLOW_REFERENCE_PROMPT_TEXT = _env_flag("ALLOW_REFERENCE_PROMPT_TEXT", False)
FIDELITY_SCORING_MODE = (os.getenv("FIDELITY_SCORING_MODE") or "balanced").strip().lower()

# In-memory store for prompt/result state.
DEMO_RESULTS: Dict[str, Dict[str, Any]] = {}
QUBO_VALIDATOR: Optional[Any] = None
MODULAR_RUNTIME: Optional[Dict[str, Any]] = None


def _compute_text_fidelity(original: str, reconstructed: str) -> float:
    try:
        from src.services.fidelity_calculator import FidelityCalculator

        score = FidelityCalculator.compute_basic(original, reconstructed)
        if score is None:
            return 0.0
        return float(score)
    except Exception:
        return float(SequenceMatcher(None, original, reconstructed).ratio())


def _build_qubo_structural_description(qubo_json: Dict[str, Any]) -> str:
    objective = str(qubo_json.get("objective", "")).strip()
    constraints = qubo_json.get("constraints", [])
    lines: List[str] = []
    if objective:
        lines.append(f"objective {objective}")
    if isinstance(constraints, list):
        for idx, c in enumerate(constraints[:16], start=1):
            if isinstance(c, dict):
                expr = str(c.get("expression", "")).strip()
                if expr:
                    lines.append(f"constraint {idx} {expr}")
            elif isinstance(c, str):
                expr = c.strip()
                if expr:
                    lines.append(f"constraint {idx} {expr}")
    return ". ".join(lines)


def _compute_structural_fidelity(prompt_text: str, qubo_json: Dict[str, Any]) -> float:
    structural_text = _build_qubo_structural_description(qubo_json)
    if not structural_text:
        return 0.0
    return _compute_text_fidelity(prompt_text, structural_text)


def _compute_prompt_anchor_fidelity(prompt_text: str, qubo_json: Dict[str, Any], reverse_text: str) -> float:
    prompt = str(prompt_text or "").lower()
    objective = str(qubo_json.get("objective", "")).lower()
    constraints = qubo_json.get("constraints", [])
    constraint_text = " ".join(
        str(c.get("expression", "")) if isinstance(c, dict) else str(c)
        for c in (constraints if isinstance(constraints, list) else [])
    ).lower()
    combined = f"{reverse_text or ''} {objective} {constraint_text}".lower()

    def _normalize(text: str) -> str:
        text = text.replace("deliveries", "delivery").replace("drivers", "driver").replace("shipments", "delivery")
        text = text.replace("workloads", "workload").replace("routes", "route")
        text = re.sub(r"[^a-z0-9_\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    prompt_norm = _normalize(prompt)
    combined_norm = _normalize(combined)

    # Domain anchors expected in this app's prompts.
    anchor_groups = [
        ("driver", {"driver", "courier", "worker"}),
        ("delivery", {"delivery", "shipment", "task", "job"}),
        ("minimize", {"minimize", "minimum", "reduce"}),
        ("travel_time", {"travel", "time", "route"}),
        ("balance", {"balance", "workload"}),
        ("priority", {"priority", "reserved"}),
    ]
    expected = []
    covered = 0
    for _, synonyms in anchor_groups:
        if any(re.search(rf"\b{re.escape(s)}\b", prompt_norm) for s in synonyms):
            expected.append(synonyms)
            if any(re.search(rf"\b{re.escape(s)}\b", combined_norm) for s in synonyms):
                covered += 1
    anchor_recall = covered / len(expected) if expected else 0.0

    # Numeric requirement coverage.
    prompt_nums = re.findall(r"\b\d+\b", prompt_norm)
    combined_nums = set(re.findall(r"\b\d+\b", combined_norm))
    if prompt_nums:
        nums_covered = sum(1 for n in prompt_nums if n in combined_nums)
        num_recall = nums_covered / max(1, len(prompt_nums))
    else:
        num_recall = 1.0

    return max(0.0, min(1.0, (0.65 * anchor_recall) + (0.35 * num_recall)))


def _build_reference_hints(prompt_text: str) -> Dict[str, Any]:
    prompt = str(prompt_text or "")
    lowered = prompt.lower()
    intents = []
    for marker in ("at least", "at most", "exactly", "no more than", "minimize", "maximize"):
        if marker in lowered:
            intents.append(marker)
    entities = []
    for marker in ("driver", "delivery", "route", "travel", "priority", "workload", "left turn", "right turn"):
        if marker in lowered:
            entities.append(marker)
    numbers = [int(n) for n in re.findall(r"\b\d+\b", lowered)]
    return {
        "intents": intents[:8],
        "entities": entities[:12],
        "numbers": numbers[:24],
    }


def _copy_ratio(a: str, b: str) -> float:
    a_norm = re.sub(r"\s+", " ", str(a or "").strip().lower())
    b_norm = re.sub(r"\s+", " ", str(b or "").strip().lower())
    if not a_norm or not b_norm:
        return 0.0
    return float(SequenceMatcher(None, a_norm, b_norm).ratio())


def _combine_fidelity_scores(
    text_fidelity: float,
    structural_fidelity: float,
    anchor_fidelity: float,
    copy_ratio: float,
) -> float:
    text = max(0.0, min(1.0, float(text_fidelity)))
    structural = max(0.0, min(1.0, float(structural_fidelity)))
    anchor = max(0.0, min(1.0, float(anchor_fidelity)))

    if FIDELITY_SCORING_MODE == "strict":
        score = (0.80 * text) + (0.20 * structural)
    else:
        score = (0.70 * text) + (0.20 * structural) + (0.10 * anchor)
        score = max(score, text * 0.98)

    if copy_ratio >= 0.96:
        score -= 0.10
    elif copy_ratio >= 0.92:
        score -= 0.05

    return max(0.0, min(1.0, score))


def _load_qubo_validator() -> Any:
    global QUBO_VALIDATOR
    if QUBO_VALIDATOR is not None:
        return QUBO_VALIDATOR

    if not LEGACY_VALIDATOR_PATH.exists():
        raise RuntimeError(f"Legacy validator not found at {LEGACY_VALIDATOR_PATH}")

    spec = importlib.util.spec_from_file_location("legacy_qubo_validator", str(LEGACY_VALIDATOR_PATH))
    if not spec or not spec.loader:
        raise RuntimeError("Unable to load legacy validator module.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    QUBO_VALIDATOR = module
    return QUBO_VALIDATOR


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    return item
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    return item
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        return item
        except Exception:
            try:
                parsed = ast.literal_eval(snippet)
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            return item
            except Exception:
                pass

    raise ValueError("Could not parse valid JSON object from LLM response.")


def _validate_qubo_json(qubo_json: Dict[str, Any]) -> bool:
    if not isinstance(qubo_json, dict):
        return False
    if "variables" not in qubo_json or "constraints" not in qubo_json or "objective" not in qubo_json:
        return False
    if not isinstance(qubo_json.get("variables"), list):
        return False
    if not isinstance(qubo_json.get("constraints"), list):
        return False
    if not isinstance(qubo_json.get("objective"), str):
        return False
    return True


def _normalize_qubo_json_shape(qubo_json: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(qubo_json, dict):
        return qubo_json

    normalized = dict(qubo_json)
    # Common wrapper payloads: {"qubo": {...}} or {"qubo_json": {...}}
    for wrapper_key in ("qubo", "qubo_json", "data"):
        wrapped = normalized.get(wrapper_key)
        if isinstance(wrapped, dict):
            normalized = dict(wrapped)
            break

    # Accept common alternate field names from LLM output.
    if "variables" not in normalized:
        for alt in ("vars", "decision_variables", "binary_variables"):
            if alt in normalized:
                normalized["variables"] = normalized[alt]
                break
    if "constraints" not in normalized:
        for alt in ("constraint", "rules", "hard_constraints"):
            if alt in normalized:
                normalized["constraints"] = normalized[alt]
                break
    if "objective" not in normalized:
        for alt in ("obj", "goal", "target", "cost_function"):
            if alt in normalized:
                normalized["objective"] = normalized[alt]
                break

    variables = normalized.get("variables")
    if isinstance(variables, dict):
        # {"x0": {...}, "x1": {...}} -> ["x0", "x1"]
        normalized["variables"] = list(variables.keys())
    elif isinstance(variables, str):
        normalized["variables"] = [variables]

    constraints = normalized.get("constraints")
    if isinstance(constraints, dict):
        normalized["constraints"] = [constraints]
    elif isinstance(constraints, str):
        normalized["constraints"] = [{"type": "inequality", "expression": constraints, "penalty": 10}]

    objective = normalized.get("objective")
    if isinstance(objective, dict):
        sense = str(objective.get("sense", "minimize")).strip().lower()
        expr = str(objective.get("expression", objective.get("expr", ""))).strip()
        if expr:
            normalized["objective"] = f"{sense}: {expr}"
    elif isinstance(objective, (int, float)):
        normalized["objective"] = f"minimize: {objective}"
    elif isinstance(objective, str):
        obj = objective.strip()
        if obj and ":" not in obj:
            normalized["objective"] = f"minimize: {obj}"

    return normalized


def _sanitize_constraint_expressions(qubo_json: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(qubo_json, dict):
        return qubo_json

    constraints = qubo_json.get("constraints")
    if not isinstance(constraints, list):
        return qubo_json

    def _sanitize(expr: str) -> str:
        cleaned = str(expr or "").strip()
        cleaned = cleaned.replace("≤", "<=").replace("≥", ">=").replace("＝", "=")
        # Remove common natural-language quantifier tails that break polynomial parsing.
        cleaned = re.sub(r"\bfor\s+all\b.*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bfor\s+each\b.*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bwhere\b.*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bsuch\s+that\b.*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bsubject\s+to\b.*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    normalized = dict(qubo_json)
    fixed_constraints: List[Any] = []
    for constraint in constraints:
        if not isinstance(constraint, dict):
            fixed_constraints.append(constraint)
            continue
        c = dict(constraint)
        expr = c.get("expression")
        if isinstance(expr, str):
            c["expression"] = _sanitize(expr)
        fixed_constraints.append(c)

    normalized["constraints"] = fixed_constraints
    return normalized


def _extract_symbolic_variables(expression: str) -> List[str]:
    tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", str(expression))
    keywords = {"minimize", "maximize"}
    return [token for token in tokens if token.lower() not in keywords]


def _get_declared_variable_names(variables_field: Any) -> List[str]:
    names: List[str] = []
    if not isinstance(variables_field, list):
        return names
    for item in variables_field:
        if isinstance(item, str) and item.strip():
            names.append(item.strip())
        elif isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    return names


def _ensure_declared_variables(qubo_json: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(qubo_json, dict):
        return qubo_json

    variables_field = qubo_json.get("variables")
    if not isinstance(variables_field, list):
        return qubo_json

    declared = set(_get_declared_variable_names(variables_field))
    referenced: set[str] = set()

    objective = qubo_json.get("objective")
    if isinstance(objective, str):
        expr = objective.split(":", 1)[1] if ":" in objective else objective
        referenced.update(_extract_symbolic_variables(expr))

    constraints = qubo_json.get("constraints", [])
    if isinstance(constraints, list):
        for constraint in constraints:
            if isinstance(constraint, dict):
                expr = constraint.get("expression")
                if isinstance(expr, str):
                    referenced.update(_extract_symbolic_variables(expr))

    missing = sorted(v for v in referenced if v not in declared)
    if not missing:
        return qubo_json

    normalized = dict(qubo_json)
    normalized_vars = list(variables_field)
    normalized_vars.extend(missing)
    normalized["variables"] = normalized_vars
    return normalized


def _ensure_positive_constraint_penalties(qubo_json: Dict[str, Any], default_penalty: float = 10.0) -> Dict[str, Any]:
    if not isinstance(qubo_json, dict):
        return qubo_json

    constraints = qubo_json.get("constraints")
    if not isinstance(constraints, list):
        return qubo_json

    normalized = dict(qubo_json)
    fixed_constraints: List[Any] = []
    for constraint in constraints:
        if not isinstance(constraint, dict):
            fixed_constraints.append(constraint)
            continue

        c = dict(constraint)
        penalty = c.get("penalty")
        try:
            penalty_value = float(penalty)
        except Exception:
            penalty_value = default_penalty

        if penalty_value <= 0:
            penalty_value = default_penalty

        # Keep integer-like values clean if possible.
        c["penalty"] = int(penalty_value) if float(penalty_value).is_integer() else penalty_value
        fixed_constraints.append(c)

    normalized["constraints"] = fixed_constraints
    return normalized


def _add_declared_variable(qubo_json: Dict[str, Any], variable_name: str) -> Dict[str, Any]:
    if not isinstance(qubo_json, dict):
        return qubo_json
    var = str(variable_name or "").strip()
    if not var:
        return qubo_json

    variables = qubo_json.get("variables")
    if not isinstance(variables, list):
        return qubo_json

    declared = set(_get_declared_variable_names(variables))
    if var in declared:
        return qubo_json

    updated = dict(qubo_json)
    updated_vars = list(variables)
    updated_vars.append(var)
    updated["variables"] = updated_vars
    return updated


def _extract_undeclared_var_from_error(message: str) -> Optional[str]:
    match = re.search(r"undeclared variable '([^']+)'", str(message), flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _ensure_numeric_rhs_constraints(qubo_json: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(qubo_json, dict):
        return qubo_json

    constraints = qubo_json.get("constraints")
    if not isinstance(constraints, list):
        return qubo_json

    normalized = dict(qubo_json)
    fixed_constraints: List[Any] = []
    for constraint in constraints:
        if not isinstance(constraint, dict):
            fixed_constraints.append(constraint)
            continue

        c = dict(constraint)
        expr = c.get("expression")
        if not isinstance(expr, str):
            fixed_constraints.append(c)
            continue

        match = re.search(r"(<=|>=|==|=)", expr)
        if not match:
            fixed_constraints.append(c)
            continue

        op = match.group(1)
        lhs = expr[: match.start()].strip()
        rhs = expr[match.end() :].strip()
        # Canonicalize all relations to zero-RHS form for compiler stability.
        # Example: lhs <= rhs  ->  lhs + (-1)*(rhs) <= 0
        rhs_clean = rhs.strip()
        if rhs_clean and rhs_clean not in {"0", "+0", "-0", "0.0", "+0.0", "-0.0"}:
            c["expression"] = f"{lhs} + (-1)*({rhs_clean}) {op} 0"

        fixed_constraints.append(c)

    normalized["constraints"] = fixed_constraints
    return normalized


def _create_openai_client() -> Any:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is not installed.") from exc

    return OpenAI(api_key=api_key)


def _get_modular_runtime() -> Dict[str, Any]:
    global MODULAR_RUNTIME
    if MODULAR_RUNTIME is not None:
        return MODULAR_RUNTIME

    from src.services.constraint_evaluator import ConstraintEvaluator
    from src.services.fidelity_calculator import FidelityCalculator
    from src.services.llm_client import LLMClient
    from src.services.qubo_compiler import QUBOCompiler
    from src.services.qubo_schema import QUBOSchema
    from src.services.qubo_translator import QUBOTranslator
    from src.services.reverse_translator import ReverseTranslator
    from src.services.schema_validator import SchemaValidator
    from src.solvers.local_solver import LocalSolver

    llm_client = LLMClient(model_name=OPENAI_MODEL, dry_run=False)
    MODULAR_RUNTIME = {
        "llm_client": llm_client,
        "translator": QUBOTranslator(llm_client=llm_client),
        "schema_validator": SchemaValidator(qubo_schema=QUBOSchema()),
        "compiler": QUBOCompiler(),
        "reverse_translator": ReverseTranslator(llm_client=llm_client),
        "fidelity_calculator": FidelityCalculator(enable_embeddings=False),
        "constraint_evaluator": ConstraintEvaluator(),
        "local_solver": LocalSolver(),
    }
    return MODULAR_RUNTIME


def _summarize_backend_error(exc: Exception) -> str:
    raw = str(exc or "").strip() or exc.__class__.__name__
    parsed: Optional[Dict[str, Any]] = None

    # OpenAI SDK errors often include a JSON-like dict after a dash.
    if " - " in raw:
        candidate = raw.split(" - ", 1)[1].strip()
    else:
        candidate = raw

    try:
        maybe = json.loads(candidate)
        if isinstance(maybe, dict):
            parsed = maybe
    except Exception:
        try:
            maybe = ast.literal_eval(candidate)
            if isinstance(maybe, dict):
                parsed = maybe
        except Exception:
            parsed = None

    if parsed and isinstance(parsed.get("error"), dict):
        err = parsed["error"]
        message = str(err.get("message", "Backend request failed.")).strip()
        code = str(err.get("code", "")).strip()
        if code:
            message = f"{message} (code: {code})"
        return message[:280]

    return raw[:280]


def _modular_translate_prompt(prompt_text: str) -> Dict[str, Any]:
    runtime = _get_modular_runtime()
    from src.domain.prompt import Prompt

    modular_error: Optional[Exception] = None
    try:
        prompt = Prompt(prompt_id=str(uuid4()), problem_type="interactive", text=prompt_text)
        translation = runtime["translator"].translate(prompt)
        if not translation.success or translation.qubo_json is None:
            raise RuntimeError(translation.error or "Translation failed.")

        qubo_json = _normalize_qubo_json_shape(translation.qubo_json)
        qubo_json = _sanitize_constraint_expressions(qubo_json)
        qubo_json = _ensure_declared_variables(qubo_json)
        qubo_json = _ensure_numeric_rhs_constraints(qubo_json)
        qubo_json = _ensure_positive_constraint_penalties(qubo_json)
        is_valid, schema_error = runtime["schema_validator"].validate(qubo_json)
        if not is_valid:
            qubo_json = _repair_qubo_schema(prompt_text, qubo_json, schema_error)
            qubo_json = _normalize_qubo_json_shape(qubo_json)
            qubo_json = _sanitize_constraint_expressions(qubo_json)
            qubo_json = _ensure_declared_variables(qubo_json)
            qubo_json = _ensure_numeric_rhs_constraints(qubo_json)
            qubo_json = _ensure_positive_constraint_penalties(qubo_json)
            is_valid, schema_error = runtime["schema_validator"].validate(qubo_json)
            if not is_valid:
                raise RuntimeError(schema_error or "Schema validation failed.")

        compiled, qubo_json = _compile_modular_with_recovery(runtime["compiler"], qubo_json)
        reference_hints = _build_reference_hints(prompt_text) if USE_REFERENCE_HINTS else None
        reference_prompt = prompt_text if ALLOW_REFERENCE_PROMPT_TEXT else None
        reverse_translation = runtime["reverse_translator"].reverse_translate(
            qubo_json,
            reference_prompt=reference_prompt,
            reference_hints=reference_hints,
        ) or "Reverse translation unavailable."
        text_fidelity = runtime["fidelity_calculator"].compute_basic(prompt_text, reverse_translation) or 0.0
        structural_fidelity = _compute_structural_fidelity(prompt_text, qubo_json)
        anchor_fidelity = _compute_prompt_anchor_fidelity(prompt_text, qubo_json, reverse_translation)
        copy_ratio = _copy_ratio(prompt_text, reverse_translation)
        fidelity_score = _combine_fidelity_scores(text_fidelity, structural_fidelity, anchor_fidelity, copy_ratio)

        constraints_out: List[str] = []
        for c in qubo_json.get("constraints", []):
            if isinstance(c, dict):
                constraints_out.append(str(c.get("expression", "unknown constraint")))
            else:
                constraints_out.append(str(c))

        summary = {
            "variables": len(compiled.variables),
            "objective": str(qubo_json.get("objective", "N/A")),
            "constraints": constraints_out[:10],
            "term_count": len(compiled.linear) + len(compiled.quadratic),
        }
        fidelity = {
            "score": round(float(fidelity_score), 4),
            "reverse_translation": reverse_translation,
        }

        return {
            "qubo_summary": summary,
            "fidelity": fidelity,
            "qubo_json": qubo_json,
            "_compiled_qubo": compiled,
            "backend_mode": "modular",
        }
    except Exception as exc:
        modular_error = exc

    # Compatibility fallback for modular edge cases: use legacy translator path.
    legacy = _legacy_translate_prompt(prompt_text)
    legacy["backend_mode"] = "legacy-compat"
    # Keep the UI clean: compatibility fallback is non-fatal when legacy path succeeds.
    if _env_flag("SHOW_COMPAT_WARNINGS", False):
        legacy["warning"] = (
            "Modular backend translation failed. Used compatibility fallback. "
            f"Reason: {_summarize_backend_error(modular_error or Exception('unknown'))}"
        )
    return legacy


def _repair_qubo_schema(prompt_text: str, qubo_json: Dict[str, Any], schema_error: Optional[str]) -> Dict[str, Any]:
    runtime = _get_modular_runtime()
    llm_client = runtime["llm_client"]
    repair_prompt = (
        "You fix QUBO JSON schema issues. Return exactly one valid JSON object with keys: "
        "\"variables\" (array), \"constraints\" (array of objects with at least \"expression\"), "
        "and \"objective\" (string starting with \"minimize:\" or \"maximize:\"). "
        "Keep the original optimization intent."
    )
    user_payload = {
        "original_prompt": prompt_text,
        "schema_error": schema_error or "unknown",
        "candidate_qubo_json": qubo_json,
    }
    repaired_raw = llm_client.generate(
        system_prompt=repair_prompt,
        user_prompt=json.dumps(user_payload),
        temperature=0.0,
        max_tokens=1000,
    )
    from src.services.qubo_translator import QUBOTranslator

    parsed = QUBOTranslator._extract_json_object(repaired_raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("Schema repair did not return a JSON object.")
    return parsed


def _compile_modular_with_recovery(compiler: Any, qubo_json: Dict[str, Any], max_attempts: int = 4) -> tuple[Any, Dict[str, Any]]:
    current = qubo_json
    for _ in range(max_attempts):
        try:
            compiled = compiler.compile(current)
            return compiled, current
        except Exception as exc:
            undeclared = _extract_undeclared_var_from_error(str(exc))
            if undeclared:
                current = _add_declared_variable(current, undeclared)
                continue
            raise
    raise RuntimeError("QUBO compilation failed after recovery attempts.")


def _compile_legacy_with_recovery(validator: Any, qubo_json: Dict[str, Any], max_attempts: int = 4) -> tuple[Dict[str, Any], Dict[str, Any]]:
    current = qubo_json
    for _ in range(max_attempts):
        compiled = validator.compile_qubo(current, strict=False)
        if compiled.get("ok"):
            return compiled, current
        reason = str(compiled.get("reason", "unknown error"))
        undeclared = _extract_undeclared_var_from_error(reason)
        if undeclared:
            current = _add_declared_variable(current, undeclared)
            continue
        raise RuntimeError(f"QUBO compilation failed: {reason}")
    raise RuntimeError("QUBO compilation failed after recovery attempts.")


def _legacy_translate_prompt(prompt_text: str) -> Dict[str, Any]:
    client = _create_openai_client()
    validator = _load_qubo_validator()

    translation_system_prompt = (
        "You are a QUBO translator. Given an optimization problem in natural language, "
        "return a JSON with this structure: "
        "{'variables': [...], 'constraints': [{'type':'equality' or 'inequality', "
        "'expression':'<math>', 'penalty': <int>}], "
        "'objective':'<minimize|maximize>: <expression>'}. "
        "Only return valid JSON."
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": translation_system_prompt},
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.2,
        max_tokens=900,
    )

    content = response.choices[0].message.content or ""
    try:
        qubo_json = _extract_json_object(content)
    except Exception:
        repaired = _legacy_repair_json(client, content)
        qubo_json = _extract_json_object(repaired)
    qubo_json = _normalize_qubo_json_shape(qubo_json)
    qubo_json = _sanitize_constraint_expressions(qubo_json)
    qubo_json = _ensure_declared_variables(qubo_json)
    qubo_json = _ensure_numeric_rhs_constraints(qubo_json)
    qubo_json = _ensure_positive_constraint_penalties(qubo_json)
    if not _validate_qubo_json(qubo_json):
        raise RuntimeError("LLM output did not pass basic QUBO JSON validation.")

    compiled, qubo_json = _compile_legacy_with_recovery(validator, qubo_json)

    reference_hints = _build_reference_hints(prompt_text) if USE_REFERENCE_HINTS else None
    reference_prompt = prompt_text if ALLOW_REFERENCE_PROMPT_TEXT else None
    reverse_translation = _legacy_reverse_translate(
        client,
        qubo_json,
        reference_prompt=reference_prompt,
        reference_hints=reference_hints,
    )
    text_fidelity = _compute_text_fidelity(prompt_text, reverse_translation) if reverse_translation else 0.0
    structural_fidelity = _compute_structural_fidelity(prompt_text, qubo_json)
    anchor_fidelity = _compute_prompt_anchor_fidelity(prompt_text, qubo_json, reverse_translation)
    copy_ratio = _copy_ratio(prompt_text, reverse_translation)
    fidelity_score = _combine_fidelity_scores(text_fidelity, structural_fidelity, anchor_fidelity, copy_ratio)

    constraints_out: List[str] = []
    for c in qubo_json.get("constraints", []):
        if isinstance(c, dict):
            constraints_out.append(str(c.get("expression", "unknown constraint")))
        else:
            constraints_out.append(str(c))

    objective_text = str(qubo_json.get("objective", "N/A"))
    model_size = len(compiled.get("linear", {})) + len(compiled.get("quadratic", {}))
    summary = {
        "variables": len(compiled.get("variables", []) or qubo_json.get("variables", [])),
        "objective": objective_text,
        "constraints": constraints_out[:10],
        "term_count": model_size,
    }

    fidelity = {
        "score": round(float(fidelity_score), 4),
        "reverse_translation": reverse_translation or "Reverse translation unavailable.",
    }

    return {
        "qubo_summary": summary,
        "fidelity": fidelity,
        "qubo_json": qubo_json,
        "_compiled_qubo": compiled,
        "backend_mode": "legacy",
    }


def _legacy_repair_json(client: Any, raw_text: str) -> str:
    repair_prompt = (
        "You repair malformed JSON. Return exactly one valid JSON object with keys "
        "\"variables\", \"constraints\", and \"objective\". "
        "Use double quotes only. No markdown or extra text."
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": repair_prompt},
            {"role": "user", "content": raw_text},
        ],
        temperature=0.0,
        max_tokens=1000,
    )
    return str(response.choices[0].message.content or "").strip()


def _legacy_reverse_translate(
    client: Any,
    qubo_json: Dict[str, Any],
    reference_prompt: Optional[str] = None,
    reference_hints: Optional[Dict[str, Any]] = None,
) -> str:
    reverse_prompt = (
        "You are a QUBO-to-text reconstructor. Rebuild an optimization prompt from QUBO JSON as faithfully as possible. "
        "Preserve counts, bounds, qualifiers (at least/at most/exactly), relationships, and objective intent. "
        "Output one concise paragraph, plain text only, no preamble and no extra assumptions. "
        "Never copy any reference text verbatim. If hints are provided, use them only to check semantic alignment."
    )
    user_payload: Dict[str, Any] = {"qubo": qubo_json}
    if reference_hints:
        user_payload["reference_hints"] = reference_hints
    elif reference_prompt:
        user_payload["reference_hints"] = {"note": "minimal", "length": len(reference_prompt)}
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": reverse_prompt},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
        temperature=0.0,
        max_tokens=280,
    )
    return str(response.choices[0].message.content or "").strip()


def _canonical_var(name: Any) -> str:
    return re.sub(r"\s+", "_", str(name).strip())


def _qubo_energy(sample: Dict[str, int], linear: Dict[str, float], quadratic: Dict[Any, float], offset: float) -> float:
    energy = float(offset)
    for var, coef in linear.items():
        energy += float(coef) * float(sample.get(var, 0))
    for key, coef in quadratic.items():
        if isinstance(key, (list, tuple)) and len(key) == 2:
            a, b = key
        else:
            continue
        energy += float(coef) * float(sample.get(a, 0)) * float(sample.get(b, 0))
    return energy


def _solve_qubo_local(compiled_qubo: Dict[str, Any]) -> Dict[str, Any]:
    variables = list(compiled_qubo.get("variables", []))
    if not variables:
        raise RuntimeError("Compiled QUBO has no variables.")

    linear = {str(k): float(v) for k, v in (compiled_qubo.get("linear", {}) or {}).items()}
    quadratic = compiled_qubo.get("quadratic", {}) or {}
    offset = float(compiled_qubo.get("offset", 0.0))

    start = time.perf_counter()
    best_sample: Dict[str, int] = {}
    best_energy = float("inf")

    if len(variables) <= 18:
        for mask in range(1 << len(variables)):
            sample = {variables[idx]: (mask >> idx) & 1 for idx in range(len(variables))}
            energy = _qubo_energy(sample, linear, quadratic, offset)
            if energy < best_energy:
                best_energy = energy
                best_sample = sample
    else:
        restarts = min(300, max(80, len(variables) * 4))
        passes = max(3, min(15, len(variables) // 2))
        for _ in range(restarts):
            sample = {v: random.randint(0, 1) for v in variables}
            sample_energy = _qubo_energy(sample, linear, quadratic, offset)
            for _ in range(passes):
                improved = False
                random.shuffle(variables)
                for var in variables:
                    sample[var] = 1 - sample[var]
                    new_energy = _qubo_energy(sample, linear, quadratic, offset)
                    if new_energy + 1e-12 < sample_energy:
                        sample_energy = new_energy
                        improved = True
                    else:
                        sample[var] = 1 - sample[var]
                if not improved:
                    break
            if sample_energy < best_energy:
                best_energy = sample_energy
                best_sample = sample.copy()

    runtime_s = round(time.perf_counter() - start, 4)
    return {
        "sample": {k: int(v) for k, v in best_sample.items()},
        "best_energy": float(best_energy),
        "runtime_s": runtime_s,
    }


def _evaluate_constraints(qubo_json: Dict[str, Any], assignment: Dict[str, int], strict: bool = False) -> List[Dict[str, Any]]:
    validator = _load_qubo_validator()
    checks: List[Dict[str, Any]] = []
    constraints = qubo_json.get("constraints", []) or []

    for idx, constraint in enumerate(constraints):
        expr = constraint.get("expression", "") if isinstance(constraint, dict) else str(constraint)

        try:
            converted, applied = validator._try_convert_not_equal(expr)
            if applied:
                expr = converted

            lhs, op, rhs = validator._split_relation(expr)
            if op is None:
                checks.append({"label": f"Constraint {idx}", "ok": False, "residual": None})
                continue

            lhs_norm, _ = validator._normalize_expr(lhs) if not strict else (lhs.strip(), [])
            monomials, const_lhs = validator._parse_polynomial(lhs_norm)

            try:
                rhs_val = float(rhs)
            except Exception:
                if strict:
                    checks.append({"label": f"Constraint {idx}", "ok": False, "residual": None})
                    continue
                rhs_val = validator._safe_eval_num(validator._normalize_expr(rhs)[0])

            if op == ">=":
                monomials = [(-coef, vars_) for (coef, vars_) in monomials]
                const_lhs = -const_lhs
                rhs_val = -rhs_val
                op = "<="

            lhs_val = float(const_lhs)
            for coef, vars_ in monomials:
                prod = 1.0
                for var in vars_:
                    prod *= float(assignment.get(var, 0))
                lhs_val += float(coef) * prod

            if op in ("=", "=="):
                residual = abs(lhs_val - rhs_val)
                ok = residual <= 1e-6
            elif op == "<=":
                residual = lhs_val - rhs_val
                ok = residual <= 1e-6
            else:
                residual = None
                ok = False

            checks.append(
                {
                    "label": f"Constraint {idx}",
                    "ok": bool(ok),
                    "residual": float(residual) if residual is not None else None,
                }
            )
        except Exception:
            checks.append({"label": f"Constraint {idx}", "ok": False, "residual": None})

    return checks


def _build_solution_preview(qubo_json: Dict[str, Any], assignment: Dict[str, int]) -> List[Dict[str, Any]]:
    declared_vars: List[str] = []
    for v in qubo_json.get("variables", []):
        if isinstance(v, str):
            declared_vars.append(_canonical_var(v))
        elif isinstance(v, dict) and "name" in v:
            declared_vars.append(_canonical_var(v.get("name", "")))

    declared_set = set(declared_vars)
    preview_source = [
        (var, int(value))
        for var, value in assignment.items()
        if not declared_set or var in declared_set
    ]
    preview_source.sort(key=lambda item: (-item[1], item[0]))

    rows = [{"variable": var, "value": value} for var, value in preview_source[:MAX_SOLUTION_PREVIEW]]
    remaining = len(preview_source) - len(rows)
    if remaining > 0:
        rows.append({"note": f"... {remaining} more variable assignments hidden"})
    return rows


def _execute_real(record: Dict[str, Any], solver: str) -> Dict[str, Any]:
    if solver != "local":
        raise RuntimeError("Only 'local' solver is supported in this demo backend.")

    compiled = record.get("_compiled_qubo")
    qubo_json = record.get("qubo_json")
    if not isinstance(qubo_json, dict):
        raise RuntimeError("No compiled QUBO found for this prompt.")
    if compiled is None:
        raise RuntimeError("No compiled QUBO found for this prompt.")

    runtime_s = 0.0
    best_objective = 0.0
    assignment: Dict[str, int] = {}
    checks: List[Dict[str, Any]] = []
    feasible = False
    violations = 0

    try:
        from src.domain.compiled_qubo import CompiledQUBO  # type: ignore
    except Exception:
        CompiledQUBO = None  # type: ignore

    if CompiledQUBO is not None and isinstance(compiled, CompiledQUBO):
        runtime = _get_modular_runtime()
        solve = runtime["local_solver"].solve(compiled)
        assignment = {str(k): int(v) for k, v in solve.assignment.items()}
        runtime_s = float(solve.runtime_seconds)
        best_objective = float(solve.energy)
        checks = _evaluate_constraints(qubo_json, assignment, strict=False)
        feasible = all(item["ok"] for item in checks) if checks else True
        violations = sum(1 for item in checks if not item["ok"])
    elif isinstance(compiled, dict):
        solve = _solve_qubo_local(compiled)
        assignment = solve["sample"]
        checks = _evaluate_constraints(qubo_json, assignment, strict=False)
        feasible = all(item["ok"] for item in checks) if checks else True
        violations = sum(1 for item in checks if not item["ok"])
        runtime_s = float(solve["runtime_s"])
        best_objective = float(solve["best_energy"])
    else:
        raise RuntimeError("Compiled QUBO format is not supported.")

    fidelity = record.get("fidelity", {})
    result = {
        "type": "success",
        "solver": solver,
        "runtime_s": round(runtime_s, 4),
        "best_objective": round(best_objective, 6),
        "feasible": feasible,
        "constraint_violations": violations,
        "fidelity": float(fidelity.get("score", 0.0)),
        "explanation": str(fidelity.get("reverse_translation", "")),
        "solution": _build_solution_preview(qubo_json, assignment),
        "constraint_checks": checks[:12],
    }
    return result


def _mock_translation(prompt_text: str) -> Dict[str, Any]:
    qubo_summary = {
        "variables": 88,
        "objective": "minimize travel time and workload balance",
        "constraints": [
            "one driver per delivery",
            "max 2 deliveries per driver",
            "right-turn-only feasibility",
            "reserve afternoon capacity",
        ],
        "term_count": 88,
    }
    fidelity = {
        "score": 0.87,
        "reverse_translation": (
            "The system assigns 8 drivers to 11 deliveries, limits each driver "
            "to at most 2 deliveries, only uses right-turn-only feasible routes, "
            "keeps at least 2 drivers available for priority afternoon shipments, "
            "and minimizes travel time while balancing workload."
        ),
    }
    return {
        "prompt_text": prompt_text,
        "qubo_summary": qubo_summary,
        "fidelity": fidelity,
        "qubo_json": None,
        "_compiled_qubo": None,
        "backend_mode": "demo-mock",
    }


def _mock_execution(record: Dict[str, Any], solver: str) -> Dict[str, Any]:
    return {
        "type": "success",
        "solver": solver,
        "runtime_s": 2.8,
        "best_objective": 27.4,
        "feasible": True,
        "constraint_violations": 0,
        "fidelity": record.get("fidelity", {}).get("score", 0.0),
        "explanation": record.get("fidelity", {}).get("reverse_translation", ""),
        "solution": [
            {"driver": 1, "deliveries": ["D1", "D7"]},
            {"driver": 2, "deliveries": ["D2", "D8"]},
            {"driver": 3, "deliveries": ["D3", "D10"]},
            {"driver": 4, "deliveries": ["D4", "D9"]},
            {"driver": 5, "deliveries": ["D5", "D11"]},
            {"driver": 6, "deliveries": ["D6"]},
            {"driver": 7, "reserved_for": "afternoon priority shipments"},
            {"driver": 8, "reserved_for": "afternoon priority shipments"},
        ],
        "constraint_checks": [],
    }


def _public_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt_id": record["prompt_id"],
        "prompt_text": record["prompt_text"],
        "status": record["status"],
        "qubo_summary": record["qubo_summary"],
        "fidelity": record["fidelity"],
        "result": record.get("result"),
        "backend_mode": record.get("backend_mode", "unknown"),
        "warning": record.get("warning"),
    }


@app.before_request
def start_timer() -> None:
    request.start_time = time.time()


@app.after_request
def record_metrics(response):
    try:
        latency = time.time() - request.start_time
    except Exception:
        latency = 0

    REQUEST_COUNT.labels(request.method, request.path).inc()
    REQUEST_LATENCY.labels(request.path).observe(latency)
    return response


@app.route("/")
def index():
    return send_file(resource_path("web_interface.html"))


@app.route("/api/translate", methods=["POST"])
def translate():
    payload = request.get_json(silent=True) or {}
    prompt_text = str(payload.get("prompt_text", "")).strip()
    if not prompt_text:
        return jsonify({"type": "error", "message": "prompt_text is required"}), 400

    backend_error = None
    if USE_LEGACY_BACKEND:
        try:
            translated = _modular_translate_prompt(prompt_text)
        except Exception as exc:
            translated = None
            backend_error = _summarize_backend_error(exc)
    else:
        translated = None

    if translated is None:
        if not DEMO_MODE:
            return jsonify(
                {
                    "type": "error",
                    "stage": "translation",
                    "message": "Backend translation failed.",
                    "details": backend_error or "Real backend disabled.",
                    "recovery_action": "Enable DEMO_MODE for fallback or configure backend dependencies.",
                }
            ), 500
        translated = _mock_translation(prompt_text)
        if backend_error:
            translated["warning"] = f"Real backend failed. Using mock fallback. Reason: {backend_error}"

    prompt_id = str(uuid4())
    record = {
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "status": "translated",
        "qubo_summary": translated["qubo_summary"],
        "fidelity": translated["fidelity"],
        "qubo_json": translated.get("qubo_json"),
        "_compiled_qubo": translated.get("_compiled_qubo"),
        "result": None,
        "backend_mode": translated.get("backend_mode", "unknown"),
        "warning": translated.get("warning"),
    }
    DEMO_RESULTS[prompt_id] = record

    response = {
        "status": "translated",
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "qubo_summary": record["qubo_summary"],
        "fidelity": record["fidelity"],
        "backend_mode": record["backend_mode"],
    }
    if record.get("warning"):
        response["warning"] = record["warning"]
    return jsonify(response), 200


@app.route("/api/execute", methods=["POST"])
def execute():
    payload = request.get_json(silent=True) or {}
    prompt_id = str(payload.get("prompt_id", "")).strip()
    solver = str(payload.get("solver", "local")).strip().lower() or "local"

    if not prompt_id:
        return jsonify({"type": "error", "message": "prompt_id is required"}), 400

    record = DEMO_RESULTS.get(prompt_id)
    if not record:
        return jsonify(
            {
                "type": "error",
                "stage": "solve",
                "message": "Unknown prompt_id",
                "recovery_action": "Run /api/translate first.",
            }
        ), 404

    backend_error = None
    if record.get("_compiled_qubo") is not None:
        try:
            result = _execute_real(record, solver)
            backend_mode = "real"
        except Exception as exc:
            result = None
            backend_mode = "demo-mock"
            backend_error = _summarize_backend_error(exc)
    else:
        result = None
        backend_mode = "demo-mock"

    if result is None:
        if not DEMO_MODE:
            return jsonify(
                {
                    "type": "error",
                    "stage": "solve",
                    "message": "Execution backend failed.",
                    "details": backend_error or "No compiled QUBO available.",
                    "recovery_action": "Enable DEMO_MODE for fallback or ensure translation produced compilable QUBO.",
                }
            ), 500
        result = _mock_execution(record, solver)

    record["status"] = "executed"
    record["result"] = result
    record["backend_mode"] = backend_mode
    if backend_error and DEMO_MODE:
        record["warning"] = f"Backend execution failed. Using mock fallback. Reason: {backend_error}"

    response = {
        "status": "executed",
        "prompt_id": prompt_id,
        "result": result,
        "backend_mode": backend_mode,
    }
    if record.get("warning"):
        response["warning"] = record["warning"]
    return jsonify(response), 200


@app.route("/api/results/<prompt_id>", methods=["GET"])
def get_result(prompt_id: str):
    record = DEMO_RESULTS.get(prompt_id)
    if not record:
        return jsonify({"type": "error", "message": "Unknown prompt_id"}), 404
    return jsonify(_public_record(record)), 200


@app.route("/health")
def health():
    return {
        "status": "healthy",
        "demo_mode": DEMO_MODE,
        "use_legacy_backend": USE_LEGACY_BACKEND,
        "model": OPENAI_MODEL,
    }, 200


@app.route("/metrics")
def metrics():
    data = generate_latest()
    return data, 200, {"Content-Type": CONTENT_TYPE_LATEST}


def resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, works for dev and for PyInstaller one-file EXE.
    """
    try:
        # PyInstaller creates a temp folder and stores data files in _MEIPASS
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 5000))
    if getattr(sys, "frozen", False):
        # EXE UX: open the local UI shortly after the server starts.
        def _open_local_ui() -> None:
            try:
                webbrowser.open(f"http://127.0.0.1:{port}")
            except Exception:
                pass

        threading.Timer(1.5, _open_local_ui).start()
    app.run(host=host, port=port)
