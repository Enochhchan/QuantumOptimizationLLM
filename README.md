# NL-QUBO Translation & Execution System

## Local Development Setup

### Prerequisites
- Python 3.9 or higher
- pip

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Web Server Locally

#### Option 1: Using Flask CLI
```bash
flask run --host=0.0.0.0 --port=5000
```

#### Option 2: Using Python directly
```bash
python app.py
```

The web interface will be available at:
- **Local:** http://localhost:5000
- **Network:** http://0.0.0.0:5000

### Health Check

Test that the server is running:
```bash
curl http://localhost:5000/health
```

Or visit http://localhost:5000/health in your browser.

### Testing the Interface

1. Start the server using one of the methods above
2. Open your browser and navigate to http://localhost:5000
3. You should see the "Team Extreme's NL-QUBO Translation & Execution System" interface

## Docker Deployment

See Docker setup in Phase 2 (coming soon).

## Metrics & Monitoring

Prometheus metrics endpoint will be available at `/metrics` (Phase 3).

