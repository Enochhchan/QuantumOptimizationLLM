from flask import Flask, send_file, request, jsonify
import os
import time
import sys
from uuid import uuid4

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# ---- Prometheus Metrics ----
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests received",
    ["method", "endpoint"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"]
)


# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_MODE = os.getenv("DEMO_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}

# In-memory store for demo-only prompt/result state.
DEMO_RESULTS = {}


@app.before_request
def start_timer():
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

    if not DEMO_MODE:
        return jsonify({
            "type": "error",
            "stage": "translation",
            "message": "Backend integration is not enabled yet.",
            "recovery_action": "Enable DEMO_MODE or connect backend services."
        }), 501

    prompt_id = str(uuid4())
    qubo_summary = {
        "variables": 20,
        "objective": "minimize overtime hours",
        "constraints": [
            "shift coverage",
            "max technicians per shift"
        ],
        "term_count": 420
    }
    fidelity = {
        "score": 0.87,
        "reverse_translation": (
            "The system schedules 20 technicians across 4 shifts "
            "with constraints to reduce overtime."
        )
    }

    DEMO_RESULTS[prompt_id] = {
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "status": "translated",
        "qubo_summary": qubo_summary,
        "fidelity": fidelity,
        "result": None
    }

    return jsonify({
        "status": "translated",
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "qubo_summary": qubo_summary,
        "fidelity": fidelity
    }), 200


@app.route("/api/execute", methods=["POST"])
def execute():
    payload = request.get_json(silent=True) or {}
    prompt_id = str(payload.get("prompt_id", "")).strip()
    solver = str(payload.get("solver", "local")).strip().lower() or "local"

    if not prompt_id:
        return jsonify({"type": "error", "message": "prompt_id is required"}), 400

    if not DEMO_MODE:
        return jsonify({
            "type": "error",
            "stage": "solve",
            "message": "Backend integration is not enabled yet.",
            "recovery_action": "Enable DEMO_MODE or connect backend services."
        }), 501

    record = DEMO_RESULTS.get(prompt_id)
    if not record:
        return jsonify({
            "type": "error",
            "stage": "solve",
            "message": "Unknown prompt_id",
            "recovery_action": "Run /api/translate first."
        }), 404

    result = {
        "type": "success",
        "solver": solver,
        "runtime_s": 0.38,
        "best_objective": 14.0,
        "feasible": True,
        "fidelity": record["fidelity"]["score"],
        "explanation": record["fidelity"]["reverse_translation"],
        "solution": [
            {"shift": 1, "technicians": 5},
            {"shift": 2, "technicians": 5},
            {"shift": 3, "technicians": 5},
            {"shift": 4, "technicians": 5}
        ]
    }

    record["status"] = "executed"
    record["result"] = result

    return jsonify({
        "status": "executed",
        "prompt_id": prompt_id,
        "result": result
    }), 200


@app.route("/api/results/<prompt_id>", methods=["GET"])
def get_result(prompt_id):
    if not DEMO_MODE:
        return jsonify({
            "type": "error",
            "message": "Backend integration is not enabled yet."
        }), 501

    record = DEMO_RESULTS.get(prompt_id)
    if not record:
        return jsonify({"type": "error", "message": "Unknown prompt_id"}), 404

    return jsonify(record), 200



@app.route('/health')
def health():
    return {'status': 'healthy', 'demo_mode': DEMO_MODE}, 200


@app.route('/metrics')
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


# Added a load route that simulates work

# @app.route("/load")
# def load():
#     # Simulate work for demo (200ms)
#     import time
#     time.sleep(0.2)
#     return "Simulated load!"




if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
