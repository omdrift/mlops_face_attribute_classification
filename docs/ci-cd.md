# CI/CD Guide

This guide explains the CI/CD pipelines implemented using GitHub Actions for the MLOps face attribute classification project.

## Overview

The CI/CD setup includes:

- **CI (Continuous Integration)**: Code quality checks, testing, type checking
- **CD (Continuous Deployment)**: Docker builds, registry pushes, deployments
- **DVC Pipeline**: Data versioning and pipeline execution
- **Model Training**: Manual training triggers
- **Monitoring**: Automated drift detection

## Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)

**Triggers**: Push or PR to `master`, `main`, `devlopment`

**Jobs**:

#### Lint
- Runs flake8 for syntax errors
- Checks code formatting with black
- Caches pip dependencies

#### Test
- Runs pytest with coverage
- Uploads coverage reports to Codecov
- Requires lint to pass first

#### Type Check
- Runs mypy for type checking
- Ignores missing imports
- Requires lint to pass first

**Usage**:
```bash
# Automatically runs on push/PR
git push origin devlopment

# View workflow runs
gh run list --workflow=ci.yml
```

### 2. CD Workflow (`.github/workflows/cd.yml`)

**Triggers**: 
- Push to `master` or `main`
- Tags matching `v*`

**Jobs**:

#### Build and Push
- Builds Docker image
- Pushes to GitHub Container Registry (ghcr.io)
- Uses BuildKit caching
- Exports image as artifact for tags

#### Deploy
- Runs only for version tags (`v*`)
- Placeholder for deployment logic
- Can be extended for K8s/ECS deployment

**Usage**:
```bash
# Tag a release
git tag v1.0.0
git push origin v1.0.0

# View the deployed image
docker pull ghcr.io/omdrift/mlops_face_attribute_classification:v1.0.0
```

### 3. DVC Pipeline Workflow (`.github/workflows/dvc-pipeline.yml`)

**Triggers**: 
- Push to `master` or `main`
- Manual trigger (`workflow_dispatch`)

**Jobs**:

#### DVC Pipeline
- Configures DVC remote (AWS S3 or GCS)
- Pulls data from remote storage
- Runs `dvc repro` to execute pipeline
- Pushes results back to remote
- Commits updated `dvc.lock`

**Setup**:

1. Add secrets in GitHub repository settings:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `GCS_PROJECT_ID`

2. Configure DVC remote:
```bash
# S3
dvc remote add -d myremote s3://my-bucket/path
dvc remote modify myremote region us-west-2

# GCS
dvc remote add -d myremote gs://my-bucket/path
```

**Usage**:
```bash
# Automatically runs on push
git push origin master

# Trigger manually
gh workflow run dvc-pipeline.yml
```

### 4. Model Training Workflow (`.github/workflows/model-training.yml`)

**Triggers**: Manual (`workflow_dispatch`)

**Inputs**:
- `epochs`: Number of training epochs (default: 10)
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Learning rate (default: 0.001)

**Jobs**:

#### Train Model
- Updates `params.yaml` with inputs
- Pulls data with DVC
- Runs training script
- Uploads model and metrics as artifacts
- Posts results as workflow comment

**Usage**:

Via GitHub UI:
1. Go to Actions → Model Training
2. Click "Run workflow"
3. Enter parameters
4. Click "Run workflow"

Via CLI:
```bash
gh workflow run model-training.yml \
  -f epochs=20 \
  -f batch_size=64 \
  -f learning_rate=0.0005
```

### 5. Monitoring Workflow (`.github/workflows/monitoring.yml`)

**Triggers**: 
- Schedule: Daily at 2 AM UTC
- Manual trigger (`workflow_dispatch`)

**Jobs**:

#### Drift Check
- Pulls data with DVC
- Runs drift detection script
- Uploads drift report as artifact
- Creates GitHub issue if drift detected
- Sends notification

**Usage**:
```bash
# Runs automatically daily

# Trigger manually
gh workflow run monitoring.yml

# View drift reports
gh run download <run-id> -n drift-report
```

## GitHub Secrets

### Required Secrets

Set these in Settings → Secrets and variables → Actions:

#### For Docker Registry
- Automatically uses `GITHUB_TOKEN` (no setup needed)

#### For DVC Remote (optional)
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `GCS_PROJECT_ID`: Google Cloud project ID

#### For MLflow (optional)
- `MLFLOW_TRACKING_URI`: MLflow tracking server URL

### Setting Secrets

Via GitHub UI:
1. Go to repository Settings
2. Secrets and variables → Actions
3. Click "New repository secret"
4. Add name and value

Via CLI:
```bash
gh secret set AWS_ACCESS_KEY_ID -b"AKIAIOSFODNN7EXAMPLE"
gh secret set AWS_SECRET_ACCESS_KEY -b"wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
```

## Dependabot

Dependabot automatically creates PRs for dependency updates.

**Configuration** (`.github/dependabot.yml`):
- Python dependencies: Weekly on Monday
- GitHub Actions: Weekly on Monday
- Docker images: Weekly on Monday

**Usage**:
- Review and merge Dependabot PRs
- CI runs automatically on PRs
- Configure auto-merge if desired

## Best Practices

### Workflow Organization

1. **Separate workflows**: One workflow per responsibility
2. **Reusable workflows**: Extract common logic
3. **Conditional execution**: Use `if` conditions to skip unnecessary jobs
4. **Caching**: Cache dependencies to speed up workflows

### Security

1. **Secrets**: Never commit secrets to code
2. **Permissions**: Use least-privilege permissions
3. **GITHUB_TOKEN**: Prefer over personal access tokens
4. **Environment protection**: Use environments for production

### Performance

1. **Concurrency**: Limit concurrent workflow runs
```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

2. **Caching**: Cache pip, Docker layers
```yaml
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
```

3. **Artifacts**: Clean up old artifacts regularly

### Testing

1. **Branch protection**: Require status checks before merge
2. **Review process**: Require approvals
3. **Testing in PRs**: Run full CI on all PRs
4. **Manual approval**: For production deployments

## Extending Workflows

### Adding a New Workflow

1. Create file in `.github/workflows/`
2. Define triggers and jobs
3. Add any required secrets
4. Test with manual trigger first

Example:
```yaml
name: My Custom Workflow

on:
  workflow_dispatch:

jobs:
  my-job:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run custom script
        run: python my_script.py
```

### Reusable Workflows

Create reusable workflow:
```yaml
# .github/workflows/reusable-test.yml
name: Reusable Test Workflow

on:
  workflow_call:
    inputs:
      python-version:
        required: true
        type: string

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ inputs.python-version }}
      - run: pytest
```

Use reusable workflow:
```yaml
# .github/workflows/main.yml
jobs:
  test:
    uses: ./.github/workflows/reusable-test.yml
    with:
      python-version: '3.10'
```

### Matrix Builds

Test multiple versions:
```yaml
jobs:
  test:
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']
        os: [ubuntu-latest, macos-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
```

## Deployment Strategies

### Blue-Green Deployment

```yaml
deploy:
  steps:
    - name: Deploy to green environment
      run: |
        kubectl apply -f k8s/green-deployment.yml
    
    - name: Run smoke tests
      run: |
        ./smoke-tests.sh green
    
    - name: Switch traffic to green
      run: |
        kubectl patch service my-service -p '{"spec":{"selector":{"version":"green"}}}'
    
    - name: Cleanup blue environment
      run: |
        kubectl delete deployment blue
```

### Canary Deployment

```yaml
deploy:
  steps:
    - name: Deploy canary (10%)
      run: |
        kubectl apply -f k8s/canary-10.yml
    
    - name: Monitor metrics
      run: |
        ./monitor-canary.sh
    
    - name: Increase to 50%
      run: |
        kubectl apply -f k8s/canary-50.yml
    
    - name: Full rollout
      run: |
        kubectl apply -f k8s/production.yml
```

## Monitoring Workflows

### Workflow Status

```bash
# List recent runs
gh run list --limit 10

# View specific run
gh run view <run-id>

# Watch a running workflow
gh run watch <run-id>
```

### Notifications

Configure notifications in GitHub:
- Settings → Notifications
- Choose email/web notifications for workflow failures

### Badges

Add status badges to README:
```markdown
![CI](https://github.com/omdrift/mlops_face_attribute_classification/workflows/CI/badge.svg)
![CD](https://github.com/omdrift/mlops_face_attribute_classification/workflows/CD/badge.svg)
```

## Troubleshooting

### Workflow not triggering

1. Check trigger conditions match
2. Verify branch name is correct
3. Check if workflow is disabled
4. Review workflow syntax

### Secrets not available

1. Check secret name matches exactly
2. Verify secret is set at repo/org level
3. Check workflow permissions
4. Secrets are not available in PRs from forks

### Docker build failing

1. Check Dockerfile syntax
2. Verify build context
3. Review build logs
4. Test build locally first

### DVC pull failing

1. Verify remote is configured
2. Check credentials are valid
3. Test DVC commands locally
4. Review DVC remote permissions

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [GitHub CLI](https://cli.github.com/)
- [DVC Documentation](https://dvc.org/doc)
- [Dependabot](https://docs.github.com/en/code-security/dependabot)
