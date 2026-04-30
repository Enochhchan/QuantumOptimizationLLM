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
import time
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

        load_dotenv(find_dotenv())
    except Exception:
        # Optional dependency. If unavailable, rely on process env vars.
        pass


_load_dotenv_if_available()

DEMO_MODE = _env_flag("DEMO_MODE", True)
USE_LEGACY_BACKEND = _env_flag("USE_LEGACY_BACKEND", True)
OPENAI_MODEL = os.getenv("LEGACY_OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
MAX_SOLUTION_PREVIEW = max(5, int(os.getenv("MAX_SOLUTION_PREVIEW", "24")))

# In-memory store for prompt/result state.
DEMO_RESULTS: Dict[str, Dict[str, Any]] = {}
QUBO_VALIDATOR: Optional[Any] = None


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
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return parsed
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
        except Exception:
            parsed = ast.literal_eval(snippet)
            if isinstance(parsed, dict):
                return parsed

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


def _create_openai_client() -> Any:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is not installed.") from exc

    return OpenAI(api_key=api_key)


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
    qubo_json = _extract_json_object(content)
    if not _validate_qubo_json(qubo_json):
        raise RuntimeError("LLM output did not pass basic QUBO JSON validation.")

    compiled = validator.compile_qubo(qubo_json, strict=False)
    if not compiled.get("ok"):
        raise RuntimeError(f"QUBO compilation failed: {compiled.get('reason', 'unknown error')}")

    reverse_translation = _legacy_reverse_translate(client, qubo_json)
    fidelity_score = SequenceMatcher(None, prompt_text, reverse_translation).ratio() if reverse_translation else 0.0

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


def _legacy_reverse_translate(client: Any, qubo_json: Dict[str, Any]) -> str:
    reverse_prompt = (
        "You are a QUBO-to-text explainer. Translate this QUBO JSON into concise natural language."
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": reverse_prompt},
            {"role": "user", "content": json.dumps(qubo_json)},
        ],
        temperature=0.2,
        max_tokens=500,
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
    if not isinstance(compiled, dict) or not isinstance(qubo_json, dict):
        raise RuntimeError("No compiled QUBO found for this prompt.")

    solve = _solve_qubo_local(compiled)
    assignment = solve["sample"]
    checks = _evaluate_constraints(qubo_json, assignment, strict=False)
    feasible = all(item["ok"] for item in checks) if checks else True
    violations = sum(1 for item in checks if not item["ok"])

    fidelity = record.get("fidelity", {})
    result = {
        "type": "success",
        "solver": solver,
        "runtime_s": solve["runtime_s"],
        "best_objective": round(float(solve["best_energy"]), 6),
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
            translated = _legacy_translate_prompt(prompt_text)
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
                    "details": backend_error or "Legacy backend disabled.",
                    "recovery_action": "Enable DEMO_MODE for fallback or configure legacy dependencies.",
                }
            ), 500
        translated = _mock_translation(prompt_text)
        if backend_error:
            translated["warning"] = f"Legacy backend failed. Using mock fallback. Reason: {backend_error}"

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
            backend_mode = "legacy"
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
        record["warning"] = f"Real execution failed. Using mock fallback. Reason: {backend_error}"

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
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
