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


if __name__ == '__main__':
    # Run on all interfaces (0.0.0.0) so it works in Docker
    # Use port 5000 by default
    app.run(host='0.0.0.0', port=5000, debug=True)