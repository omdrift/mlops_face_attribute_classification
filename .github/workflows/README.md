# GitHub Actions Workflows

This directory contains CI/CD workflows for the MLOps Face Attribute Classification project.

## Workflows

### 1. CI Pipeline (`ci.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

**Jobs:**

#### Lint Job
- **Purpose:** Check code quality and style
- **Tools:** Flake8
- **Actions:**
  - Checks for Python syntax errors (E9, F63, F7, F82)
  - Reports code complexity and style issues
  - Uses max line length of 120 characters
  - Allows warnings but blocks on critical errors

#### Test Job
- **Purpose:** Run unit tests and measure code coverage
- **Tools:** pytest, pytest-cov
- **Actions:**
  - Runs all tests in `tests/` directory
  - Generates coverage report for `src/` directory
  - Uploads coverage to Codecov (optional)
  - Requires lint job to pass first

**Dependencies:**
- Python 3.10
- Core dependencies: pytest, pytest-cov, torch, torchvision, numpy, scikit-learn

---

### 2. Docker Build and Push (`docker.yml`)

**Triggers:**
- Git tags matching `v*.*.*` pattern (e.g., v1.0.0)
- GitHub releases
- Manual workflow dispatch

**Jobs:**

#### Build and Push Job
- **Purpose:** Build and publish Docker images
- **Registry:** GitHub Container Registry (ghcr.io)
- **Actions:**
  - Builds Docker image from `deployment/Dockerfile`
  - Tags images with version numbers and commit SHA
  - Pushes to GitHub Container Registry
  - Uses Docker layer caching for faster builds

**Permissions Required:**
- `contents: read` - Read repository contents
- `packages: write` - Push to GitHub Container Registry

**Image Tags Generated:**
- Branch name (e.g., `main`)
- Semantic version (e.g., `1.0.0`, `1.0`, `1`)
- Git commit SHA

---

### 3. DVC Pipeline Validation (`dvc.yml`)

**Triggers:**
- Push/PR to `main` or `develop` branches
- Changes to: `dvc.yaml`, `dvc.lock`, `params.yaml`

**Jobs:**

#### Validate DVC Job
- **Purpose:** Validate DVC pipeline configuration
- **Tools:** DVC, PyYAML
- **Actions:**
  - Validates YAML syntax of `dvc.yaml` and `params.yaml`
  - Checks pipeline structure and stages
  - Displays pipeline DAG (if possible)
  - Verifies all required fields are present

**Validations Performed:**
1. File existence checks
2. YAML syntax validation
3. Pipeline structure validation
4. Stage configuration checks

---

## Usage

### Running Workflows Locally

#### Test the CI Pipeline
```bash
# Install dependencies
pip install pytest pytest-cov torch torchvision numpy scikit-learn flake8

# Run linting
flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=120 --statistics

# Run tests
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

#### Test Docker Build
```bash
cd deployment
docker build -t mlops-face-classification:test .
```

#### Validate DVC Pipeline
```bash
pip install dvc pyyaml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('dvc.yaml'))"
python -c "import yaml; yaml.safe_load(open('params.yaml'))"

# Show pipeline
dvc dag
```

### Creating a Release

To trigger the Docker workflow and create a new release:

```bash
# Tag the release
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Or create a release on GitHub UI
# The Docker image will be automatically built and pushed
```

### Manual Workflow Dispatch

The Docker workflow can be triggered manually:
1. Go to Actions tab in GitHub
2. Select "Docker Build and Push"
3. Click "Run workflow"
4. Select branch and run

---

## Configuration

### Flake8 Configuration
Settings are defined in `pyproject.toml`:
- Max line length: 120
- Excluded directories: `.git`, `__pycache__`, `venv`, `.dvc`, `mlruns`
- Ignored rules: E203, W503

### Pytest Configuration
Settings are defined in `pyproject.toml`:
- Test path: `tests/`
- Coverage source: `src/`
- Coverage format: Terminal + HTML

---

## Troubleshooting

### Common Issues

#### Test Failures
- Ensure all dependencies are installed
- Check Python version (3.8+)
- Verify test fixtures in `tests/conftest.py`

#### Docker Build Failures
- Check if `deployment/Dockerfile` exists
- Verify all COPY paths are correct
- Ensure required files are available

#### DVC Validation Failures
- Validate YAML syntax manually
- Check if all referenced files exist
- Ensure stage commands are correct

---

## Status Badges

Add these badges to your main README.md:

```markdown
![CI Pipeline](https://github.com/omdrift/mlops_face_attribute_classification/workflows/CI%20Pipeline/badge.svg)
![Docker Build](https://github.com/omdrift/mlops_face_attribute_classification/workflows/Docker%20Build%20and%20Push/badge.svg)
![DVC Validation](https://github.com/omdrift/mlops_face_attribute_classification/workflows/DVC%20Pipeline%20Validation/badge.svg)
```
