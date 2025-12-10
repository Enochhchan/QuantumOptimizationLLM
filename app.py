from flask import Flask, send_file, request
import os
import time

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


@app.route('/')
def index():
    html_path = os.path.join(BASE_DIR, 'web_interface.html')
    return send_file(html_path)


@app.route('/health')
def health():
    return {'status': 'healthy'}, 200


@app.route('/metrics')
def metrics():
    data = generate_latest()
    return data, 200, {"Content-Type": CONTENT_TYPE_LATEST}

# @app.route("/load")
# def load():
#     # Simulate work for demo (200ms)
#     import time
#     time.sleep(0.2)
#     return "Simulated load!"

    


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
