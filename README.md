# NL-QUBO Translation & Execution System

## Local Development Setup

## Windows One-File App (No Install)

1. Open the repository's **Releases** page on GitHub.
2. Download the latest `NL-QUBO-App-*.exe` asset.
3. Double-click the `.exe` file.
4. The app starts a local server and opens your browser automatically at `http://127.0.0.1:5000`.

Notes:
- No Python or pip install is required for this mode.
- If Windows SmartScreen appears, choose **More info** -> **Run anyway**.
- Extract the download first; do not run the `.exe` directly from inside a `.zip` preview window.
- If startup fails, check `nl_qubo_startup.log` in the same folder as the `.exe`.

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

## Running the Experiment Script

The experiment runner lives at `src/llm_to_qubo_gpt4.py`.

Prereqs:
- Set `OPENAI_API_KEY` (via `.env` or environment variables)

Examples:
```bash
# Use the repo's current prompts file location
python src/llm_to_qubo_gpt4.py --prompts legacy/generated_prompts.csv --model gpt-4.1 --limit 5

# Or configure via env vars
set OPENAI_MODEL=gpt-4.1
set QUBO_PROMPTS_PATH=legacy/generated_prompts.csv
python src/llm_to_qubo_gpt4.py --limit 5
```

## Docker Deployment

See Docker setup in Phase 2 (coming soon).

## Metrics & Monitoring

Prometheus metrics endpoint will be available at `/metrics` (Phase 3).

