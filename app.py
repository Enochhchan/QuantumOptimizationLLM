from flask import Flask, send_file
import os

app = Flask(__name__)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def index():
    """Serve the main web interface."""
    html_path = os.path.join(BASE_DIR, 'web_interface.html')
    return send_file(html_path)


@app.route('/health')
def health():
    """Health check endpoint."""
    return {'status': 'healthy'}, 200


if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 5000))
    # host MUST be 0.0.0.0 so Docker can expose it
    app.run(host="0.0.0.0", port=port)
