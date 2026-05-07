from __future__ import annotations

from flask import Flask, send_file, request, jsonify
import ast
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
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
ALLOW_REFERENCE_PROMPT_TEXT = _env_flag("ALLOW_REFERENCE_PROMPT_TEXT", True)
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


def _prompt_keyword_coverage(prompt_text: str, candidate_text: str) -> float:
    prompt_tokens = [t for t in re.findall(r"\b[a-zA-Z]{4,}\b", str(prompt_text).lower()) if t not in {
        "with", "that", "this", "from", "into", "while", "then", "must", "have", "been",
    }]
    if not prompt_tokens:
        return 0.0
    cand_set = set(re.findall(r"\b[a-zA-Z]{4,}\b", str(candidate_text).lower()))
    hits = sum(1 for t in set(prompt_tokens) if t in cand_set)
    return hits / max(1, len(set(prompt_tokens)))


def _clean_reverse_translation_text(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip('"').strip("'").strip()
    return cleaned


def _infer_assignment_labels(prompt_text: str) -> Tuple[str, str]:
    lowered = str(prompt_text or "").lower()
    if "driver" in lowered and ("deliver" in lowered or "shipment" in lowered):
        return "Driver", "Delivery"
    if "team" in lowered and ("task" in lowered or "job" in lowered):
        return "Team", "Task"
    if "worker" in lowered and ("task" in lowered or "job" in lowered):
        return "Worker", "Task"
    if "vehicle" in lowered and ("route" in lowered or "stop" in lowered):
        return "Vehicle", "Stop"
    return "Group", "Item"


def _humanize_constraint_expression(expr: str, prompt_text: str) -> str:
    raw = str(expr or "").strip()
    if not raw:
        return "A rule was defined."
    if _contains_non_algebraic_constructs(raw):
        return "A counting rule is present, but it is still written as pseudo-code."

    # Undo zero-RHS canonicalization for readability: lhs + (-1)*(k) <= 0 -> lhs <= k
    m = re.match(r"^(.*?)\s*\+\s*\(-1\)\*\(([-+]?\d+(?:\.\d+)?)\)\s*(<=|>=|==|=)\s*0\s*$", raw)
    if m:
        lhs = m.group(1).strip()
        rhs = m.group(2).strip()
        op = m.group(3).strip()
    else:
        m2 = re.match(r"^(.*?)\s*(<=|>=|==|=)\s*([-+]?\d+(?:\.\d+)?)\s*$", raw)
        if m2:
            lhs = m2.group(1).strip()
            op = m2.group(2).strip()
            rhs = m2.group(3).strip()
        else:
            return f"Rule: {raw}"

    src_label, dst_label = _infer_assignment_labels(prompt_text)
    lower_lhs = lhs.lower()
    try:
        rhs_num = int(float(rhs)) if float(rhs).is_integer() else float(rhs)
    except Exception:
        rhs_num = rhs

    if op in ("=", "=="):
        relation = "exactly"
    elif op == "<=":
        relation = "at most"
    else:
        relation = "at least"

    if re.search(r"\bdelivery\d+\b", lower_lhs) and relation == "exactly":
        return f"Exactly {rhs_num} deliveries must be assigned."
    if "x_" in lower_lhs or re.search(r"\bx\d", lower_lhs):
        if "driver" in str(prompt_text).lower() and "deliver" in str(prompt_text).lower():
            return f"{relation.capitalize()} {rhs_num} delivery assignment(s) must satisfy this rule."
        return f"{relation.capitalize()} {rhs_num} decision(s) must satisfy this rule."
    return f"This rule requires {relation} {rhs_num} for expression '{lhs}'."


def _humanize_objective(objective: str, prompt_text: str) -> str:
    text = str(objective or "").strip()
    if not text:
        return "optimize an unspecified objective"
    lower = text.lower()
    sense = "minimize" if lower.startswith("minimize") else ("maximize" if lower.startswith("maximize") else "optimize")
    expr = text.split(":", 1)[1].strip() if ":" in text else text

    prompt_lower = str(prompt_text or "").lower()
    if "travel" in prompt_lower and "time" in prompt_lower:
        return f"{sense} total travel time"
    if "workload" in prompt_lower and "balance" in prompt_lower:
        return f"{sense} workload imbalance"
    if "cost" in prompt_lower:
        return f"{sense} total cost"
    return f"{sense} {expr}"


def _build_grounded_reverse_translation(prompt_text: str, qubo_json: Dict[str, Any]) -> str:
    objective = str(qubo_json.get("objective", "")).strip()
    constraints = qubo_json.get("constraints", [])
    objective_text = _humanize_objective(objective, prompt_text)

    rule_texts: List[str] = []
    if isinstance(constraints, list):
        for c in constraints[:6]:
            expr = str(c.get("expression", "")) if isinstance(c, dict) else str(c)
            if expr:
                rule_texts.append(_humanize_constraint_expression(expr, prompt_text))
    if not rule_texts:
        rule_texts = ["No clear algebraic rules were extracted."]

    prompt_lower = str(prompt_text or "").lower()
    model_text = f"{objective} " + " ".join(
        str(c.get("expression", "")) if isinstance(c, dict) else str(c)
        for c in (constraints if isinstance(constraints, list) else [])
    ).lower()
    missing: List[str] = []
    for label, terms in {
        "workload balance": ["balance", "workload"],
        "priority reserve": ["priority", "reserve", "reserved"],
        "travel-time objective": ["travel", "time"],
    }.items():
        expected = any(t in prompt_lower for t in terms)
        covered = any(t in model_text for t in terms)
        if expected and not covered:
            missing.append(label)

    sentence = (
        f"Goal: {objective_text}. "
        f"Rules captured by the current model: {'; '.join(rule_texts)}."
    )
    if missing:
        sentence += f" Possible gaps vs your original request: {', '.join(missing)}."
    return sentence


def _rule_based_reverse_translation(qubo_json: Dict[str, Any], prompt_text: str) -> str:
    objective = str(qubo_json.get("objective", "")).strip() or "minimize: unspecified objective"
    variables = _get_declared_variable_names(qubo_json.get("variables", []))
    constraints = qubo_json.get("constraints", [])

    families: Dict[str, int] = {}
    for var in variables:
        if not _is_interpretable_solution_var(var):
            continue
        parts = [p for p in re.split(r"[_\W]+", str(var)) if p]
        key = (parts[0].lower() if parts else "var")
        families[key] = families.get(key, 0) + 1
    family_text = ", ".join(f"{k}:{v}" for k, v in sorted(families.items(), key=lambda kv: kv[0])) or "none"

    constraint_phrases: List[str] = []
    if isinstance(constraints, list):
        for c in constraints[:6]:
            expr = str(c.get("expression", "")) if isinstance(c, dict) else str(c)
            if expr:
                constraint_phrases.append(_humanize_constraint_expression(expr, prompt_text))

    if constraint_phrases:
        constraints_text = "; ".join(constraint_phrases)
    else:
        constraints_text = "no clear rules were extracted"

    objective_text = _humanize_objective(objective, prompt_text)
    return (
        f"The model tries to {objective_text}. It uses {len(variables)} binary decision variables "
        f"(families: {family_text}) and applies these rules: {constraints_text}."
    )


def _select_best_reverse_translation(
    prompt_text: str,
    qubo_json: Dict[str, Any],
    llm_reverse_text: str,
) -> str:
    candidates: List[str] = []
    llm_clean = _clean_reverse_translation_text(llm_reverse_text)
    if llm_clean:
        candidates.append(llm_clean)
    candidates.append(_rule_based_reverse_translation(qubo_json, prompt_text))
    candidates.append(_build_grounded_reverse_translation(prompt_text, qubo_json))
    structural = _build_qubo_structural_description(qubo_json)
    if structural:
        candidates.append(structural)

    best_text = candidates[0] if candidates else "Reverse translation unavailable."
    best_score = -1.0
    for candidate in candidates:
        text_score = _compute_text_fidelity(prompt_text, candidate)
        anchor_score = _compute_prompt_anchor_fidelity(prompt_text, qubo_json, candidate)
        keyword_coverage = _prompt_keyword_coverage(prompt_text, candidate)
        nums_prompt = set(re.findall(r"\b\d+\b", str(prompt_text)))
        nums_candidate = set(re.findall(r"\b\d+\b", str(candidate)))
        numeric_coverage = (len(nums_prompt & nums_candidate) / len(nums_prompt)) if nums_prompt else 1.0
        symbol_density = len(re.findall(r"[<>=_*()+]", candidate)) / max(1, len(candidate))
        readability = 1.0 - min(1.0, symbol_density * 8.0)
        blended = (
            (0.44 * text_score)
            + (0.20 * anchor_score)
            + (0.16 * readability)
            + (0.12 * keyword_coverage)
            + (0.08 * numeric_coverage)
        )
        if blended > best_score:
            best_score = blended
            best_text = candidate
    return _clean_reverse_translation_text(best_text) or "Reverse translation unavailable."


def _copy_ratio(a: str, b: str) -> float:
    a_norm = re.sub(r"\s+", " ", str(a or "").strip().lower())
    b_norm = re.sub(r"\s+", " ", str(b or "").strip().lower())
    if not a_norm or not b_norm:
        return 0.0
    return float(SequenceMatcher(None, a_norm, b_norm).ratio())


def _compute_text_fidelity_ensemble(prompt_text: str, reverse_text: str, qubo_json: Dict[str, Any]) -> float:
    candidates: List[str] = []
    if reverse_text:
        candidates.append(str(reverse_text))

    structural = _build_qubo_structural_description(qubo_json)
    if structural:
        candidates.append(structural)

    objective = str(qubo_json.get("objective", "")).strip()
    constraints = qubo_json.get("constraints", [])
    if objective:
        snippets: List[str] = []
        if isinstance(constraints, list):
            for c in constraints[:12]:
                if isinstance(c, dict):
                    expr = str(c.get("expression", "")).strip()
                else:
                    expr = str(c).strip()
                if expr:
                    snippets.append(expr)
        if snippets:
            candidates.append(f"objective {objective}. constraints {'; '.join(snippets)}")
        else:
            candidates.append(f"objective {objective}")

    scores = [_compute_text_fidelity(prompt_text, c) for c in candidates if str(c).strip()]
    return max(scores) if scores else 0.0


def _assess_qubo_quality(qubo_json: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []

    variables = _get_declared_variable_names(qubo_json.get("variables", []))
    declared = set(variables)
    objective = str(qubo_json.get("objective", "")).strip()
    constraints = qubo_json.get("constraints", [])

    if not declared:
        issues.append("No declared decision variables.")

    obj_expr = objective.split(":", 1)[1] if ":" in objective else objective
    obj_vars = set(_extract_symbolic_variables(obj_expr))
    if not obj_vars:
        issues.append("Objective has no decision-variable terms.")

    constraint_vars: set[str] = set()
    multi_var_constraints = 0
    relation_count = 0
    non_algebraic_count = 0
    per_constraint_vars: List[set[str]] = []
    if isinstance(constraints, list):
        for c in constraints:
            expr = str(c.get("expression", "")) if isinstance(c, dict) else str(c)
            if _contains_non_algebraic_constructs(expr):
                non_algebraic_count += 1
            if re.search(r"(<=|>=|==|=)", expr):
                relation_count += 1
            vars_in_expr = set(_extract_symbolic_variables(expr))
            per_constraint_vars.append(vars_in_expr)
            constraint_vars.update(vars_in_expr)
            if len(vars_in_expr) >= 2:
                multi_var_constraints += 1

    used_vars = (obj_vars | constraint_vars) & declared
    coverage = (len(used_vars) / len(declared)) if declared else 0.0
    if declared and coverage < 0.60:
        issues.append("Many declared variables are unused by objective/constraints.")

    if len(declared) >= 8 and len(obj_vars & declared) <= 2:
        issues.append("Objective references too few variables for model size.")

    if len(declared) >= 8 and isinstance(constraints, list) and len(constraints) < 3:
        issues.append("Too few constraints for medium/large binary model.")
    elif len(declared) >= 4 and isinstance(constraints, list) and len(constraints) < 2:
        issues.append("Too few constraints for declared decision variables.")

    if len(declared) >= 8 and multi_var_constraints == 0:
        issues.append("Constraints do not couple variables; solution may be degenerate.")

    if isinstance(constraints, list) and relation_count == 0:
        issues.append("Constraint expressions are missing relation operators.")
    if non_algebraic_count > 0:
        issues.append("Constraints contain non-algebraic pseudo-code (e.g., sum/for/in).")

    placeholder_vars = [
        v for v in declared
        if re.match(r"^(total|sum|cost|time|score|objective)_?", v, flags=re.IGNORECASE)
    ]
    orphan_placeholders = [v for v in placeholder_vars if v not in constraint_vars and v not in obj_vars]
    if orphan_placeholders:
        issues.append("Aggregate placeholder variables are declared but not functionally linked.")

    placeholder_set = set(placeholder_vars)
    non_placeholder_set = declared - placeholder_set
    objective_declared_vars = obj_vars & declared
    if objective_declared_vars and objective_declared_vars.issubset(placeholder_set) and non_placeholder_set:
        linked = False
        for vars_in_expr in per_constraint_vars:
            if vars_in_expr & placeholder_set and vars_in_expr & non_placeholder_set:
                linked = True
                break
        if not linked:
            issues.append("Objective depends on aggregate placeholders disconnected from decision variables.")

    suspicious_vars = []
    for var in declared:
        parts = [p for p in re.split(r"[_\W]+", var.lower()) if p]
        if any(part in _NON_DECISION_TOKENS for part in parts):
            if not re.search(r"\d", var):
                suspicious_vars.append(var)
        if len(parts) >= 4 and any(part in {"for", "in", "sum"} for part in parts):
            suspicious_vars.append(var)
    if suspicious_vars:
        issues.append("Variable list contains likely prose tokens instead of decision symbols.")

    issue_count = len(issues)
    score = max(0.0, min(1.0, 1.0 - (0.17 * issue_count)))
    return {
        "score": round(score, 4),
        "issues": issues[:8],
        "coverage": round(float(coverage), 4),
        "declared_variables": len(declared),
        "constraints": len(constraints) if isinstance(constraints, list) else 0,
    }


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
        # Balanced mode favors semantic/constraint alignment over pure lexical overlap.
        base = (0.50 * text) + (0.25 * structural) + (0.25 * anchor)
        coverage = (0.58 * anchor) + (0.42 * structural)
        score = max(base, (0.88 * base) + (0.12 * coverage))
        score = max(score, (0.60 * max(text, structural)) + (0.40 * anchor))

        # If structure + anchors are solid, keep score in an interpretable pass band.
        if coverage >= 0.65 and text >= 0.30:
            score = max(score, min(0.89, (0.54 * coverage) + (0.38 * text) + 0.08))

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


_NON_DECISION_TOKENS = {
    "minimize", "maximize", "sum", "for", "in", "where", "subject", "to", "and", "or",
    "each", "all", "if", "then", "least", "most", "exactly", "at", "no", "more", "than",
    "delivery", "deliveries", "driver", "drivers", "priority", "route", "routes",
    "travel", "time", "workload", "balance", "reserved", "total", "cost", "score",
    "objective", "assign", "assignment",
}
_ITERATOR_TOKENS = {"i", "j", "k", "m", "n", "t"}


def _looks_like_decision_symbol(token: str) -> bool:
    t = str(token or "").strip()
    if not t:
        return False
    lowered = t.lower()
    if lowered in _NON_DECISION_TOKENS:
        return False
    parts = [p for p in re.split(r"[_\W]+", lowered) if p]
    if parts and all(p in _NON_DECISION_TOKENS for p in parts):
        return False
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return False
    if len(t) == 1 and t.isalpha():
        if lowered in _ITERATOR_TOKENS:
            return False
        return True
    if re.search(r"\d", t):
        return True
    if "_" in t:
        # Allow indexed/family-like symbols (x_ij, x_1_2, driver_1_delivery_2), but reject prose fragments.
        if re.search(r"\d", t):
            return True
        if re.fullmatch(r"[A-Za-z]+_[A-Za-z0-9_]+", t) and len(t.split("_")) <= 3:
            return True
    if re.fullmatch(r"[A-Z][A-Z0-9_]*", t):
        return True
    return False


def _contains_non_algebraic_constructs(expression: str) -> bool:
    text = str(expression or "")
    patterns = [
        r"\bsum\s*\(",
        r"\bfor\s+[A-Za-z_][A-Za-z0-9_]*\s+in\s+[A-Za-z_][A-Za-z0-9_]*",
        r"\bsubject\s+to\b",
        r"\bwhere\b",
        r"\bfor\s+all\b",
        r"\bfor\s+each\b",
    ]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _extract_symbolic_variables(expression: str) -> List[str]:
    tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", str(expression))
    return [token for token in tokens if _looks_like_decision_symbol(token)]


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

        quality = _assess_qubo_quality(qubo_json)
        if quality["issues"]:
            repair_attempts = 0
            while repair_attempts < 2 and quality["issues"]:
                repair_attempts += 1
                try:
                    candidate = _repair_qubo_quality(prompt_text, qubo_json, quality["issues"])
                except Exception:
                    break
                candidate = _normalize_qubo_json_shape(candidate)
                candidate = _sanitize_constraint_expressions(candidate)
                candidate = _ensure_declared_variables(candidate)
                candidate = _ensure_numeric_rhs_constraints(candidate)
                candidate = _ensure_positive_constraint_penalties(candidate)
                is_valid, schema_error = runtime["schema_validator"].validate(candidate)
                if not is_valid:
                    continue
                candidate_quality = _assess_qubo_quality(candidate)
                if candidate_quality["score"] >= quality["score"] or len(candidate_quality["issues"]) <= len(quality["issues"]):
                    qubo_json = candidate
                    quality = candidate_quality
                if not quality["issues"]:
                    break

        compiled, qubo_json = _compile_modular_with_recovery(runtime["compiler"], qubo_json)
        reference_hints = _build_reference_hints(prompt_text) if USE_REFERENCE_HINTS else None
        reference_prompt = prompt_text if ALLOW_REFERENCE_PROMPT_TEXT else None
        reverse_translation = runtime["reverse_translator"].reverse_translate(
            qubo_json,
            reference_prompt=reference_prompt,
            reference_hints=reference_hints,
        ) or "Reverse translation unavailable."
        reverse_translation = _select_best_reverse_translation(prompt_text, qubo_json, reverse_translation)
        text_fidelity = _compute_text_fidelity_ensemble(prompt_text, reverse_translation, qubo_json)
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
            "model_quality": quality,
            "qubo_json": qubo_json,
            "_compiled_qubo": compiled,
            "backend_mode": "modular",
            "warning": (
                f"Model quality is low ({quality['score']:.2f}). Solver output may be unreliable. "
                + "; ".join(quality["issues"][:3])
            ) if quality["score"] < 0.75 else None,
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


def _repair_qubo_quality(prompt_text: str, qubo_json: Dict[str, Any], quality_issues: List[str]) -> Dict[str, Any]:
    runtime = _get_modular_runtime()
    llm_client = runtime["llm_client"]
    repair_prompt = (
        "You improve a low-quality QUBO JSON model while preserving the original optimization intent. "
        "Return exactly one RFC8259 JSON object with keys: "
        "\"variables\" (array), \"constraints\" (array of objects with expression and positive penalty), "
        "\"objective\" (string beginning with minimize: or maximize:). "
        "Requirements: use explicit binary decision variables, ensure every objective/constraint symbol is declared, "
        "encode all key requirements as algebraic constraints with numeric RHS, avoid unlinked placeholder variables."
    )
    user_payload = {
        "original_prompt": prompt_text,
        "quality_issues": quality_issues[:8],
        "candidate_qubo_json": qubo_json,
    }
    repaired_raw = llm_client.generate(
        system_prompt=repair_prompt,
        user_prompt=json.dumps(user_payload),
        temperature=0.0,
        max_tokens=1300,
    )
    from src.services.qubo_translator import QUBOTranslator

    parsed = QUBOTranslator._extract_json_object(repaired_raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("Quality repair did not return a JSON object.")
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
        "You are a QUBO translator. Convert the optimization request into a mathematically coherent QUBO-ready model. "
        "Return one RFC8259 JSON object only, with double quotes, no markdown, no prose. "
        "Schema: "
        "{\"variables\":[...],\"constraints\":[{\"type\":\"equality\"|\"inequality\",\"expression\":\"<math>\",\"penalty\":<number>}],"
        "\"objective\":\"<minimize|maximize>: <expression>\"}. "
        "Rules: (1) Every symbol used in objective/constraints must be in variables. "
        "(2) Use explicit binary decision variables (for assignment problems prefer names like x_driver_delivery). "
        "(3) Encode each natural-language requirement as at least one concrete algebraic constraint with numeric RHS. "
        "(4) Do not invent placeholder aggregate variables (for example total_travel_time) unless tied to equations. "
        "(5) Keep penalties positive and scaled so constraints are enforceable."
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

    quality = _assess_qubo_quality(qubo_json)
    if quality["issues"]:
        repair_attempts = 0
        while repair_attempts < 2 and quality["issues"]:
            repair_attempts += 1
            try:
                candidate = _repair_qubo_quality(prompt_text, qubo_json, quality["issues"])
            except Exception:
                break
            candidate = _normalize_qubo_json_shape(candidate)
            candidate = _sanitize_constraint_expressions(candidate)
            candidate = _ensure_declared_variables(candidate)
            candidate = _ensure_numeric_rhs_constraints(candidate)
            candidate = _ensure_positive_constraint_penalties(candidate)
            if not _validate_qubo_json(candidate):
                continue
            candidate_quality = _assess_qubo_quality(candidate)
            if candidate_quality["score"] >= quality["score"] or len(candidate_quality["issues"]) <= len(quality["issues"]):
                qubo_json = candidate
                quality = candidate_quality
            if not quality["issues"]:
                break

    compiled, qubo_json = _compile_legacy_with_recovery(validator, qubo_json)

    reference_hints = _build_reference_hints(prompt_text) if USE_REFERENCE_HINTS else None
    reference_prompt = prompt_text if ALLOW_REFERENCE_PROMPT_TEXT else None
    reverse_translation = _legacy_reverse_translate(
        client,
        qubo_json,
        reference_prompt=reference_prompt,
        reference_hints=reference_hints,
    )
    reverse_translation = _select_best_reverse_translation(prompt_text, qubo_json, reverse_translation)
    text_fidelity = _compute_text_fidelity_ensemble(prompt_text, reverse_translation, qubo_json)
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
        "model_quality": quality,
        "qubo_json": qubo_json,
        "_compiled_qubo": compiled,
        "backend_mode": "legacy",
        "warning": (
            f"Model quality is low ({quality['score']:.2f}). Solver output may be unreliable. "
            + "; ".join(quality["issues"][:3])
        ) if quality["score"] < 0.75 else None,
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
        "Never interpret pseudo-code like sum(... for ... in ...) as literal variables; describe only algebraic terms present. "
        "If the model appears malformed, say that explicitly and avoid inventing missing business context. "
        "Output one concise paragraph, plain text only, no preamble and no extra assumptions. "
        "Never copy any reference text verbatim. If hints are provided, use them only to check semantic alignment."
    )
    user_payload: Dict[str, Any] = {"qubo": qubo_json}
    if reference_hints:
        user_payload["reference_hints"] = reference_hints
    if reference_prompt:
        user_payload["reference_prompt"] = reference_prompt
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


def _is_interpretable_solution_var(name: str) -> bool:
    var = _canonical_var(name)
    if not var:
        return False
    # Skip compiler-generated helpers and obvious prose fragments.
    if var.upper().startswith("SLACK_") or var.upper().startswith("AUX_"):
        return False
    parts = [p for p in re.split(r"[_\W]+", var.lower()) if p]
    if not parts:
        return False
    if len(parts) >= 4 and any(p in {"for", "in", "sum"} for p in parts):
        return False
    if all((p in _NON_DECISION_TOKENS) for p in parts):
        return False
    return True


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
                checks.append(
                    {
                        "label": f"Constraint {idx + 1}",
                        "expression": str(expr),
                        "ok": False,
                        "residual": None,
                    }
                )
                continue

            lhs_norm, _ = validator._normalize_expr(lhs) if not strict else (lhs.strip(), [])
            monomials, const_lhs = validator._parse_polynomial(lhs_norm)

            try:
                rhs_val = float(rhs)
            except Exception:
                if strict:
                    checks.append(
                        {
                            "label": f"Constraint {idx + 1}",
                            "expression": str(expr),
                            "ok": False,
                            "residual": None,
                        }
                    )
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
                    "label": f"Constraint {idx + 1}",
                    "expression": str(expr),
                    "ok": bool(ok),
                    "residual": float(residual) if residual is not None else None,
                    "operator": str(op),
                    "lhs_value": float(lhs_val),
                    "rhs_value": float(rhs_val),
                }
            )
        except Exception:
            checks.append(
                {
                    "label": f"Constraint {idx + 1}",
                    "expression": str(expr),
                    "ok": False,
                    "residual": None,
                }
            )

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
        if (not declared_set or var in declared_set) and _is_interpretable_solution_var(var)
    ]
    selected_source = [(var, value) for var, value in preview_source if value == 1]
    selected_source.sort(key=lambda item: item[0])

    if not preview_source:
        return [{"note": "No interpretable decision assignments found. Model variables appear malformed."}]
    if not selected_source:
        return [{"note": "No decision variables were selected (all interpretable decisions are 0)."}]

    rows = [{"variable": var, "value": value} for var, value in selected_source[:MAX_SOLUTION_PREVIEW]]
    remaining = len(selected_source) - len(rows)
    if remaining > 0:
        rows.append({"note": f"... {remaining} more selected decision assignments hidden"})
    return rows


def _natural_var_key(name: str) -> Tuple[Any, ...]:
    parts = re.findall(r"\d+|[A-Za-z]+|[^A-Za-z0-9]+", str(name))
    key: List[Any] = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return tuple(key)


def _variable_family(name: str) -> str:
    text = str(name).strip()
    if not text:
        return "other"
    m = re.match(r"^([A-Za-z]+)", text)
    if m:
        return m.group(1).upper()
    return "other"


def _build_solution_overview(
    qubo_json: Dict[str, Any],
    assignment: Dict[str, int],
    best_objective: float,
    checks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    declared_vars: List[str] = []
    for v in qubo_json.get("variables", []):
        if isinstance(v, str):
            declared_vars.append(_canonical_var(v))
        elif isinstance(v, dict) and "name" in v:
            declared_vars.append(_canonical_var(v.get("name", "")))

    declared_set = set(declared_vars)
    scoped = [
        (str(var), int(value))
        for var, value in assignment.items()
        if (not declared_set or str(var) in declared_set) and _is_interpretable_solution_var(str(var))
    ]
    scoped.sort(key=lambda item: _natural_var_key(item[0]))

    selected = [var for var, value in scoped if value == 1]
    unselected = [var for var, value in scoped if value == 0]
    total = len(scoped)
    activation_ratio = (len(selected) / total) if total else 0.0

    families: Dict[str, Dict[str, Any]] = {}
    for var, value in scoped:
        fam = _variable_family(var)
        if fam not in families:
            families[fam] = {"family": fam, "total": 0, "selected": 0, "samples": []}
        entry = families[fam]
        entry["total"] += 1
        if value == 1:
            entry["selected"] += 1
            if len(entry["samples"]) < 6:
                entry["samples"].append(var)

    family_rows = sorted(families.values(), key=lambda row: (-int(row["total"]), str(row["family"])))
    diagnostics: List[str] = []
    if total == 0:
        diagnostics.append("No interpretable decision variables were present in the solved model.")
    if total >= 6 and activation_ratio >= 0.85:
        diagnostics.append(
            "Most decision variables are active. This often indicates weak objective scaling or missing exclusivity constraints."
        )
    if total >= 6 and activation_ratio <= 0.15:
        diagnostics.append(
            "Most decision variables are inactive. This often indicates over-penalized constraints or objective imbalance."
        )
    if abs(float(best_objective)) <= 1e-9 and len(selected) > 0:
        diagnostics.append(
            "Best objective is near zero with non-empty assignment. Verify objective coefficients and units."
        )
    if checks:
        violation_count = sum(1 for item in checks if not item.get("ok"))
        if violation_count > 0:
            diagnostics.append(
                f"{violation_count} constraint(s) failed. Inspect residuals and penalty magnitudes."
            )

    return {
        "total_variables": total,
        "selected_variables": len(selected),
        "unselected_variables": len(unselected),
        "activation_ratio": round(activation_ratio, 4),
        "selected_preview": selected[:18],
        "inactive_preview": unselected[:10],
        "families": family_rows[:8],
        "diagnostics": diagnostics[:5],
    }


def _build_decision_plan(assignment: Dict[str, int], prompt_text: str = "") -> List[str]:
    selected = sorted(
        [str(var) for var, value in assignment.items() if int(value) == 1 and _is_interpretable_solution_var(str(var))]
    )
    if not selected:
        return []

    src_label, dst_label = _infer_assignment_labels(prompt_text)
    plan_lines: List[str] = []
    matched = 0
    for var in selected:
        if re.fullmatch(r"delivery\d+", var, flags=re.IGNORECASE):
            # Delivery presence indicators are usually helper flags, not actionable assignments.
            continue
        if re.fullmatch(r"delivery[a-z]+", var, flags=re.IGNORECASE):
            # Ambiguous delivery label flags (e.g., deliveryi) are usually malformed helper variables.
            continue
        tokens = [t for t in re.split(r"[_\W]+", var) if t]
        if len(tokens) >= 3 and tokens[0].lower() in {"x", "assign", "a"}:
            src = tokens[1]
            dst = tokens[2] if len(tokens) == 3 else "_".join(tokens[2:])
            if src.isdigit() and dst:
                plan_lines.append(f"{src_label} {src} is assigned to {dst_label} {dst}.")
            else:
                idx = ", ".join(tokens[1:])
                plan_lines.append(f"Selected option ({idx}).")
            matched += 1
        elif re.fullmatch(r"d\d+", var, flags=re.IGNORECASE):
            plan_lines.append(f"{src_label} {var[1:]} is selected.")
            matched += 1
        elif len(tokens) >= 2:
            left = tokens[0]
            right = "_".join(tokens[1:])
            if left.isdigit():
                plan_lines.append(f"{src_label} {left} is linked to {dst_label} {right}.")
            else:
                plan_lines.append(f"{left} -> {right}")
            matched += 1
        else:
            plan_lines.append(var)

    # If most vars don't map cleanly, prefer raw selected vars to avoid misleading interpretation.
    if matched < max(1, len(selected) // 2):
        fallback = [
            item for item in selected
            if not re.fullmatch(r"delivery\d+", item, flags=re.IGNORECASE)
        ]
        if not fallback:
            fallback = selected
        return [f"Selected decision variable: {item}" for item in fallback[:20]]
    return plan_lines[:20]


def _build_compat_solution_lines(
    feasible: bool,
    violations: int,
    decision_plan: List[str],
    overview: Dict[str, Any],
) -> List[str]:
    lines: List[str] = []
    if feasible:
        lines.append("Workable plan found.")
    else:
        lines.append(f"No workable plan found ({violations} rule violation(s)).")

    selected_count = int(overview.get("selected_variables", 0) or 0)
    total_count = int(overview.get("total_variables", 0) or 0)
    lines.append(f"Selected decisions: {selected_count} of {total_count}.")

    if decision_plan:
        lines.append("Recommended actions:")
        for idx, action in enumerate(decision_plan[:8], start=1):
            lines.append(f"{idx}. {action}")
    else:
        lines.append("No clear human-readable actions were extracted.")
    return lines


def _build_simple_solver_answer(
    feasible: bool,
    violations: int,
    decision_plan: List[str],
    violated_constraints: List[Dict[str, Any]],
    runtime_s: float,
    model_quality: Dict[str, Any],
    prompt_text: str = "",
) -> Dict[str, Any]:
    quality_score = float(model_quality.get("score", 0.0) or 0.0)
    quality_percent = int(round(max(0.0, min(1.0, quality_score)) * 100))

    if feasible:
        headline = "A workable plan was found."
        status = "All required rules were satisfied."
    else:
        headline = "No workable plan was found."
        status = f"{violations} rule(s) were not satisfied."

    def _humanize_action(action: str) -> str:
        text = str(action or "").strip()
        if not text:
            return ""
        if re.search(r"\bdriver\b", text, flags=re.IGNORECASE):
            return text
        m = re.match(r"^[A-Za-z]+\(([^)]+)\)\s*=\s*1$", text)
        if m:
            return f"Selected option ({m.group(1)})."
        if "->" in text and not re.search(r"[_][A-Za-z0-9]", text):
            return text
        if "_" in text or "=" in text:
            return "A technical decision variable was selected."
        return text

    actions_raw = decision_plan[:8]
    actions = [a for a in (_humanize_action(item) for item in actions_raw) if a]
    if actions and all(a == "A technical decision variable was selected." for a in actions):
        actions = [f"{len(actions_raw)} decision option(s) were selected (names are technical)."]

    if not actions and feasible:
        actions = ["The model did not produce clear human-readable actions."]

    why_lines: List[str] = []
    if not feasible and violated_constraints:
        for item in violated_constraints[:3]:
            label = str(item.get("label", "Rule"))
            expr = str(item.get("expression", "")).strip()
            if expr:
                why_lines.append(f"{label} failed: {_humanize_constraint_expression(expr, prompt_text)}")
            else:
                why_lines.append(f"{label} was not satisfied.")
    if not why_lines and not feasible:
        why_lines.append("Some required rules were not satisfied.")

    next_steps: List[str] = []
    if feasible and quality_score >= 0.75:
        next_steps.append("You can proceed with this plan.")
    elif feasible:
        next_steps.append("Use caution: the model quality is moderate, so verify key assignments manually.")
    else:
        next_steps.append("Improve the model rules/objective and run again.")
        next_steps.append("Focus first on the failed rules shown below.")

    confidence = "high" if quality_score >= 0.80 else ("medium" if quality_score >= 0.60 else "low")
    return {
        "headline": headline,
        "status": status,
        "actions": actions,
        "why_not": why_lines,
        "next_steps": next_steps,
        "confidence": confidence,
        "quality_percent": quality_percent,
        "runtime_s": round(float(runtime_s), 4),
    }


def _extract_compiled_terms(compiled: Any) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], float]:
    linear: Dict[str, float] = {}
    quadratic: Dict[Tuple[str, str], float] = {}
    offset = 0.0

    if isinstance(compiled, dict):
        linear = {str(k): float(v) for k, v in (compiled.get("linear", {}) or {}).items()}
        for key, value in (compiled.get("quadratic", {}) or {}).items():
            if isinstance(key, (tuple, list)) and len(key) == 2:
                a, b = str(key[0]), str(key[1])
                quadratic[tuple(sorted((a, b)))] = float(value)
        offset = float(compiled.get("offset", 0.0))
        return linear, quadratic, offset

    try:
        from src.domain.compiled_qubo import CompiledQUBO  # type: ignore
    except Exception:
        CompiledQUBO = None  # type: ignore

    if CompiledQUBO is not None and isinstance(compiled, CompiledQUBO):
        linear = {str(k): float(v) for k, v in (compiled.linear or {}).items()}
        for key, value in (compiled.quadratic or {}).items():
            if isinstance(key, tuple) and len(key) == 2:
                a, b = str(key[0]), str(key[1])
                quadratic[tuple(sorted((a, b)))] = float(value)
        offset = float(compiled.offset or 0.0)

    return linear, quadratic, offset


def _build_objective_breakdown(compiled: Any, assignment: Dict[str, int], best_objective: float) -> Dict[str, Any]:
    linear, quadratic, offset = _extract_compiled_terms(compiled)

    linear_terms: List[Dict[str, Any]] = []
    linear_sum = 0.0
    for var, coef in linear.items():
        value = int(assignment.get(var, 0))
        contribution = float(coef) * float(value)
        if value == 1 and _is_interpretable_solution_var(var):
            linear_terms.append({"term": var, "coef": float(coef), "contribution": float(contribution)})
        linear_sum += contribution
    linear_terms.sort(key=lambda item: abs(float(item["contribution"])), reverse=True)

    quadratic_terms: List[Dict[str, Any]] = []
    quadratic_sum = 0.0
    for (a, b), coef in quadratic.items():
        av = int(assignment.get(a, 0))
        bv = int(assignment.get(b, 0))
        contribution = float(coef) * float(av) * float(bv)
        if av == 1 and bv == 1 and _is_interpretable_solution_var(a) and _is_interpretable_solution_var(b):
            quadratic_terms.append(
                {"term": f"{a}*{b}", "coef": float(coef), "contribution": float(contribution)}
            )
        quadratic_sum += contribution
    quadratic_terms.sort(key=lambda item: abs(float(item["contribution"])), reverse=True)

    recomputed = float(offset + linear_sum + quadratic_sum)
    return {
        "offset": round(float(offset), 6),
        "linear_total_sum": round(float(linear_sum), 6),
        "quadratic_total_sum": round(float(quadratic_sum), 6),
        "linear_selected_sum": round(float(sum(t["contribution"] for t in linear_terms)), 6),
        "quadratic_selected_sum": round(float(sum(t["contribution"] for t in quadratic_terms)), 6),
        "energy_recomputed": round(recomputed, 6),
        "energy_reported": round(float(best_objective), 6),
        "energy_gap": round(float(recomputed - float(best_objective)), 6),
        "top_linear_terms": linear_terms[:8],
        "top_quadratic_terms": quadratic_terms[:6],
    }


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
    model_quality = record.get("model_quality", {})
    violated_constraints = [item for item in checks if not item.get("ok")]
    objective_breakdown = _build_objective_breakdown(compiled, assignment, best_objective)
    decision_plan = _build_decision_plan(assignment, str(record.get("prompt_text", "")))
    solution_overview = _build_solution_overview(qubo_json, assignment, best_objective, checks)
    compat_solution_lines = _build_compat_solution_lines(feasible, violations, decision_plan, solution_overview)
    simple_answer = _build_simple_solver_answer(
        feasible=feasible,
        violations=violations,
        decision_plan=decision_plan,
        violated_constraints=violated_constraints,
        runtime_s=runtime_s,
        model_quality=model_quality,
        prompt_text=str(record.get("prompt_text", "")),
    )
    if feasible:
        solver_summary = "Feasible solution found."
    else:
        top_label = violated_constraints[0].get("label", "Constraint") if violated_constraints else "Constraint"
        solver_summary = (
            f"Infeasible solution: {violations} constraint(s) violated. "
            f"Most critical: {top_label}."
        )
    result = {
        "type": "success",
        "solver": solver,
        "runtime_s": round(runtime_s, 4),
        "best_objective": round(best_objective, 6),
        "feasible": feasible,
        "constraint_violations": violations,
        "fidelity": float(fidelity.get("score", 0.0)),
        "model_quality": model_quality,
        "solver_summary": solver_summary,
        "simple_answer": simple_answer,
        "explanation": str(fidelity.get("reverse_translation", "")),
        "solution": compat_solution_lines,
        "raw_solution_preview": _build_solution_preview(qubo_json, assignment),
        "decision_plan": decision_plan,
        "solution_overview": solution_overview,
        "objective_breakdown": objective_breakdown,
        "violated_constraints": violated_constraints[:5],
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
        "model_quality": {"score": 0.9, "issues": [], "coverage": 1.0, "declared_variables": 88, "constraints": 4},
        "qubo_json": None,
        "_compiled_qubo": None,
        "backend_mode": "demo-mock",
    }


def _mock_execution(record: Dict[str, Any], solver: str) -> Dict[str, Any]:
    decision_plan = [
        "Driver 1 -> D1,D7",
        "Driver 2 -> D2,D8",
        "Driver 3 -> D3,D10",
        "Driver 4 -> D4,D9",
    ]
    simple_answer = _build_simple_solver_answer(
        feasible=True,
        violations=0,
        decision_plan=decision_plan,
        violated_constraints=[],
        runtime_s=2.8,
        model_quality=record.get("model_quality", {}),
        prompt_text=str(record.get("prompt_text", "")),
    )
    raw_preview = [
        {"driver": 1, "deliveries": ["D1", "D7"]},
        {"driver": 2, "deliveries": ["D2", "D8"]},
        {"driver": 3, "deliveries": ["D3", "D10"]},
        {"driver": 4, "deliveries": ["D4", "D9"]},
        {"driver": 5, "deliveries": ["D5", "D11"]},
        {"driver": 6, "deliveries": ["D6"]},
    ]
    compat_solution = [
        "Workable plan found.",
        "Selected decisions: 11 of 88.",
        "Recommended actions:",
        "1. Driver 1 -> D1,D7",
        "2. Driver 2 -> D2,D8",
        "3. Driver 3 -> D3,D10",
        "4. Driver 4 -> D4,D9",
    ]
    return {
        "type": "success",
        "solver": solver,
        "runtime_s": 2.8,
        "best_objective": 27.4,
        "feasible": True,
        "constraint_violations": 0,
        "fidelity": record.get("fidelity", {}).get("score", 0.0),
        "model_quality": record.get("model_quality", {}),
        "solver_summary": "Feasible solution found.",
        "simple_answer": simple_answer,
        "explanation": record.get("fidelity", {}).get("reverse_translation", ""),
        "solution": compat_solution,
        "raw_solution_preview": raw_preview,
        "decision_plan": decision_plan,
        "objective_breakdown": {
            "offset": 0.0,
            "linear_total_sum": 27.4,
            "quadratic_total_sum": 0.0,
            "linear_selected_sum": 27.4,
            "quadratic_selected_sum": 0.0,
            "energy_recomputed": 27.4,
            "energy_reported": 27.4,
            "energy_gap": 0.0,
            "top_linear_terms": [],
            "top_quadratic_terms": [],
        },
        "violated_constraints": [],
        "constraint_checks": [],
    }


def _public_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt_id": record["prompt_id"],
        "prompt_text": record["prompt_text"],
        "status": record["status"],
        "qubo_summary": record["qubo_summary"],
        "fidelity": record["fidelity"],
        "model_quality": record.get("model_quality"),
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
    response = send_file(resource_path("web_interface.html"))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


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
        "model_quality": translated.get("model_quality"),
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
        "model_quality": record.get("model_quality"),
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


def _startup_log_path() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().with_name("nl_qubo_startup.log")
    return BASE_DIR / "nl_qubo_startup.log"


def _log_startup(message: str) -> None:
    try:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
        with _startup_log_path().open("a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception:
        pass


def _open_local_ui_in_chrome(port: int) -> None:
    url = f"http://127.0.0.1:{port}"
    if os.name == "nt":
        candidates = [
            os.path.join(
                os.getenv("PROGRAMFILES", ""),
                "Google",
                "Chrome",
                "Application",
                "chrome.exe",
            ),
            os.path.join(
                os.getenv("PROGRAMFILES(X86)", ""),
                "Google",
                "Chrome",
                "Application",
                "chrome.exe",
            ),
            os.path.join(
                os.getenv("LOCALAPPDATA", ""),
                "Google",
                "Chrome",
                "Application",
                "chrome.exe",
            ),
        ]
        for chrome_path in candidates:
            if chrome_path and os.path.exists(chrome_path):
                subprocess.Popen([chrome_path, "--new-window", url])
                return

    webbrowser.open(url)


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 5000))
    _log_startup(f"starting app (frozen={getattr(sys, 'frozen', False)}) host={host} port={port}")
    if getattr(sys, "frozen", False):
        # EXE UX: open the local UI in Chrome after the server is reachable.
        def _open_local_ui_when_ready() -> None:
            deadline = time.time() + 20.0
            while time.time() < deadline:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.6):
                        break
                except OSError:
                    time.sleep(0.4)
            try:
                _open_local_ui_in_chrome(port)
                _log_startup(f"browser launch attempted for http://127.0.0.1:{port}")
            except Exception:
                _log_startup(f"browser launch failed: {traceback.format_exc()}")

        threading.Thread(target=_open_local_ui_when_ready, daemon=True).start()
    try:
        app.run(host=host, port=port)
    except Exception:
        _log_startup(f"server crashed: {traceback.format_exc()}")
        raise
