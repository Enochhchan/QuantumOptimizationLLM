# DevOps Pipeline Presentation Outline & Script

## Presentation Structure (8 Required + 1 Bonus)

### Slide 1: Title & Overview

**Script:** "Today I'll demonstrate a complete DevOps pipeline for a LaTeX document compilation and containerized web application. The pipeline includes issue tracking, automated testing, CI/CD, versioning, and monitoring."

---

### Slide 2: Issue Tracking Integration

**What to show:** GitHub Issues page

**Script:**

- "I use GitHub Issues for issue tracking. Here's an issue I created: [Show issue]"
- "I'll add a comment describing the fix I'm about to implement: 'Adding automatic version incrementing to the CI/CD pipeline using a Python script that reads version.txt and increments the build number on each commit.'"
- "This links the issue to the code change, providing traceability."

**Key terms:** Issue tracking, traceability, change management

---

### Slide 3: Source Code Update & Commit

**What to show:** GitHub commit diff

**Script:**

- "I made a meaningful change to the codebase - added automatic version incrementing"
- "Here's the commit with a clear message: 'Add automatic build number incrementing with version.py'"
- "The commit message follows best practices: it's descriptive and explains what was changed and why."

**Key terms:** Version control, commit message, semantic versioning

---

### Slide 4: Pipeline Triggering (Live Demo)

**What to show:** GitHub Actions workflow run

**Script:**

- "The pipeline triggers automatically on every push to the main branch"
- "I can also trigger it manually using workflow_dispatch"
- "Let me show you the pipeline running in real-time" [Show Actions tab]
- "The workflow is event-driven - it responds to git push events"

**Key terms:** CI/CD pipeline, event-driven, workflow trigger, continuous integration

---

### Slide 5: Build Process Visibility

**What to show:** GitHub Actions logs showing each step

**Script:**

- "The pipeline provides full visibility into each build step:"
- "Step 1: Checkout repository - pulls the latest code"
- "Step 2: Set up Python environment - installs Python 3.11"
- "Step 3: Install dependencies - installs project requirements"
- "Step 4: Run automated tests - executes pytest"
- "Step 5: Auto-increment version - runs version.py script"
- "Step 6: Compile LaTeX - builds the PDF document using latexmk"
- "Step 7: Package artifact - creates versioned ZIP file"
- "Each step shows real-time output, making debugging easy"

**Key terms:** Build visibility, pipeline stages, build logs, compilation

---

### Slide 6: Automated Testing

**What to show:** Test output from pytest

**Script:**

- "The pipeline runs automated tests using pytest"
- "Here you can see the test results: [Show test output]"
- "Tests validate version format, file existence, and import functionality"
- "If tests fail, the pipeline stops and reports the failure"
- "This ensures code quality before deployment"

**Key terms:** Automated testing, test automation, continuous testing, test-driven development

---

### Slide 7: Deployment Artifact with Versioning

**What to show:** Artifact download and version.txt

**Script:**

- "The pipeline generates a deployment-ready artifact with semantic versioning"
- "Version format: major.minor.buildNumber - currently 1.0.214"
- "The build number auto-increments on every commit"
- "Artifact naming: dsnManual_v1.0.214_c75d79c.zip"
- "The ZIP contains the compiled PDF with version embedded in the title"
- "This ensures every deployment is traceable to a specific version"

**Key terms:** Deployment artifact, semantic versioning, build number, artifact versioning

---

### Slide 8: SWOT Analysis - GitHub Actions vs Jenkins

**What to show:** SWOT chart slide

**GitHub Actions:**

- **Strengths:** Native GitHub integration, no infrastructure setup, extensive marketplace, YAML-based configuration
- **Weaknesses:** Limited to GitHub ecosystem, runner minutes can be expensive at scale, less flexible than self-hosted
- **Opportunities:** Growing ecosystem, easy integration with GitHub features, cloud-native approach
- **Threats:** Vendor lock-in, potential cost scaling, dependency on GitHub availability

**Jenkins:**

- **Strengths:** Highly customizable, self-hosted control, extensive plugin ecosystem, works with any SCM
- **Weaknesses:** Requires infrastructure management, more complex setup, steeper learning curve
- **Opportunities:** Enterprise adoption, on-premise deployments, fine-grained control
- **Threats:** Maintenance overhead, plugin compatibility issues, resource-intensive

**Key terms:** CI/CD tools comparison, infrastructure as code, self-hosted vs cloud

---

### Slide 9: Configuration Files Presentation

**What to show:** 4 key configuration files with code snippets

#### Code Snippet 1: CI/CD Pipeline (.github/workflows/devops-pipeline.yml)

```yaml
name: Build LaTeX and Package PDF

on:
  push:
    branches:
      - main
  workflow_dispatch:

jobs:
  build-latex:
    runs-on: ubuntu-latest
    permissions:
      contents: write
```

**Script:** "This is the GitHub Actions workflow file. It defines the CI/CD pipeline with triggers, permissions, and build steps. The `on: push` section enables automatic triggering on commits."

#### Code Snippet 2: Version Management (version.py)

```python
def increment_build_number():
    current = VERSION_FILE.read_text().strip()
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", current)
    major, minor, build = map(int, match.groups())
    new_build = build + 1
    new_version = f"{major}.{minor}.{new_build}"
    VERSION_FILE.write_text(f"{new_version}\n")
    return new_version
```

**Script:** "This Python script implements automated version incrementing. It reads version.txt, parses the semantic version, increments the build number, and writes it back. This ensures every build gets a unique version."

#### Code Snippet 3: Containerization (Dockerfile)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt
COPY app.py web_interface.html ./
EXPOSE 5000
CMD ["python", "app.py"]
```

**Script:** "The Dockerfile defines the container image for the web application. It uses a multi-stage approach: base image, dependency installation, code copying, and service exposure. This enables consistent deployments across environments."

#### Code Snippet 4: Orchestration (docker-compose.yml)

```yaml
services:
  web:
    build: .
    ports:
      - "5000:5000"
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
```

**Script:** "Docker Compose orchestrates multiple services: the web application, Prometheus for metrics collection, and Grafana for visualization. It defines service dependencies, port mappings, and volume mounts for configuration."

**Key terms:** Infrastructure as Code, containerization, orchestration, declarative configuration

---

### Slide 10: Bonus - Cloud Execution Monitoring

**What to show:** Grafana dashboard with metrics

**Script:**

- "For the bonus component, I've implemented monitoring using Prometheus and Grafana"
- "Prometheus scrapes metrics from the Flask application every 5 seconds"
- "Grafana visualizes system metrics: CPU usage, memory consumption, request rates, and latency"
- "During the build process, you can see spikes in resource usage"
- "This provides observability into pipeline performance and system health"

**Key terms:** Observability, metrics collection, monitoring, system health, performance metrics

---

### Slide 11: Live Pipeline Demonstration

**What to do:** Trigger workflow and walk through execution

**Script:**

- "Let me trigger the pipeline now and walk you through the execution"
- [Trigger workflow manually or show recent run]
- "Watch as each step executes: checkout, setup, test, build, package"
- "Notice the version incrementing automatically"
- "The artifact is generated and uploaded"
- "The entire process is automated and repeatable"

---

### Slide 12: Summary & Key Takeaways

**Script:**

- "I've demonstrated a complete DevOps pipeline with:"
- "Issue tracking integration for change management"
- "Automated CI/CD with GitHub Actions"
- "Automated testing with pytest"
- "Semantic versioning with auto-incrementing build numbers"
- "Containerized deployment with Docker"
- "Infrastructure as Code with YAML configurations"
- "Monitoring and observability with Prometheus/Grafana"
- "This pipeline ensures consistent, traceable, and automated software delivery"

**Key terms:** DevOps, continuous integration, continuous deployment, infrastructure as code, observability

---

## Files to Demonstrate (in order):

1. **GitHub Issues** - Show issue tracking
2. **GitHub Commits** - Show source code changes
3. **.github/workflows/devops-pipeline.yml** - Main CI/CD configuration
4. **version.py** - Automated versioning script
5. **Dockerfile** - Container definition
6. **docker-compose.yml** - Service orchestration
7. **Jenkinsfile** - Alternative CI/CD (for comparison)
8. **prometheus.yml** - Monitoring configuration
9. **GitHub Actions Run** - Live pipeline execution
10. **Grafana Dashboard** - Monitoring visualization (bonus)

## Key DevOps Terminology to Use:

- **CI/CD**: Continuous Integration/Continuous Deployment
- **Pipeline**: Automated workflow for building, testing, and deploying
- **Artifact**: Deployable output (ZIP, container image)
- **Semantic Versioning**: Version format (major.minor.patch)
- **Infrastructure as Code**: Defining infrastructure in configuration files
- **Containerization**: Packaging applications in containers
- **Orchestration**: Managing multiple containers/services
- **Observability**: Monitoring, logging, and tracing
- **Event-driven**: Triggered by events (git push)
- **Declarative**: Describing desired state vs. imperative commands
- **Traceability**: Linking changes to issues and commits
- **Automated Testing**: Tests run automatically in pipeline
- **Build Visibility**: Transparent build process with logs
- **Deployment Artifact**: Versioned, deployable package

