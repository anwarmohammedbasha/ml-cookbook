# 14 — CI/CD with GitHub Actions

**Source file:** [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)

## What Is CI/CD?

**CI** (Continuous Integration) is the practice of automatically testing every code change before it is merged. This catches bugs early, when they are cheapest to fix.

**CD** (Continuous Delivery or Continuous Deployment) is the practice of automatically packaging and delivering the application after tests pass. This ensures that the latest working version is always available.

## The Workflow File

```yaml
name: Customer Churn CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: customer-churn
    steps:
      - name: Check out repository
        uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          python -m pip install -r requirements.txt
      - name: Run tests
        run: python -m pytest -q

  build-and-push:
    needs: [test]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - name: Check out repository
        uses: actions/checkout@v4
      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build and push Docker image
        uses: docker/build-push-action@v6
        with:
          context: ./customer-churn
          push: true
          tags: ghcr.io/${{ github.repository_owner }}/customer-churn-api:latest
```

## Trigger Conditions

```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
```

The workflow runs on two events:
- **`push` to `main`** — when a commit is pushed directly to the main branch (or a PR is merged).
- **`pull_request` targeting `main`** — when a PR is opened, updated, or synchronised against `main`.

This means every proposed change is tested before merging, and every merge triggers a new Docker image build.

## Job 1: `test`

### `defaults.run.working-directory: customer-churn`

All `run` steps in this job execute from the `customer-churn/` directory, so `requirements.txt` and `pytest` are found without needing to `cd` in every step.

### Steps

| Step | Action | Purpose |
|------|--------|---------|
| Check out repository | `actions/checkout@v4` | Downloads the repository code onto the runner |
| Set up Python | `actions/setup-python@v5` | Installs Python 3.11 on the runner |
| Install dependencies | `pip install -r requirements.txt` | Installs all project libraries |
| Run tests | `python -m pytest -q` | Runs the test suite; fails the job if any test fails |

If the `Run tests` step exits with a non-zero code (i.e., any test fails), the job fails and the `build-and-push` job is blocked.

## Job 2: `build-and-push`

### Conditions

```yaml
needs: [test]
if: github.event_name == 'push' && github.ref == 'refs/heads/main'
```

- **`needs: [test]`** — This job only starts after the `test` job completes successfully.
- **`if: ...`** — This job only runs on direct pushes to `main`, not on pull requests. This prevents publishing an image for every PR.

### Permissions

```yaml
permissions:
  contents: read
  packages: write
```

`packages: write` grants the workflow permission to push to GitHub Container Registry (GHCR). The `GITHUB_TOKEN` secret is automatically provided by GitHub Actions and scoped to these permissions.

### Steps

**Log in to GHCR:**
```yaml
uses: docker/login-action@v3
with:
  registry: ghcr.io
  username: ${{ github.actor }}
  password: ${{ secrets.GITHUB_TOKEN }}
```

`github.actor` is the GitHub username that triggered the workflow. `secrets.GITHUB_TOKEN` is an automatically generated token with the permissions defined above.

**Build and push:**
```yaml
uses: docker/build-push-action@v6
with:
  context: ./customer-churn
  push: true
  tags: ghcr.io/${{ github.repository_owner }}/customer-churn-api:latest
```

- `context: ./customer-churn` — The Docker build context is the `customer-churn/` directory (where the `Dockerfile` lives).
- `push: true` — Pushes the built image to GHCR.
- `tags` — The image is tagged as `latest`. `github.repository_owner` is the GitHub user or organisation that owns the repository.

## Workflow Diagram

```mermaid
flowchart TD
    A["Git push or PR to main"] --> B{Event type?}
    B -->|push or PR| C["Job: test\nubuntu-latest"]
    C --> D["checkout@v4"]
    D --> E["setup-python@v5\nPython 3.11"]
    E --> F["pip install -r requirements.txt"]
    F --> G["pytest -q"]
    G --> H{Tests pass?}
    H -->|No| I["❌ Workflow fails\nPR cannot be merged"]
    H -->|Yes| J{Is this a push to main?}
    J -->|No (PR)| K["✅ Tests passed\nPR can be merged"]
    J -->|Yes| L["Job: build-and-push\nubuntu-latest"]
    L --> M["checkout@v4"]
    M --> N["docker/login-action\nghcr.io"]
    N --> O["docker/build-push-action\ncontext: ./customer-churn"]
    O --> P["✅ Image pushed to\nghcr.io/<owner>/customer-churn-api:latest"]
```

## Pulling the Published Image

Once the workflow has run successfully on `main`, the image can be pulled from GHCR:

```bash
docker pull ghcr.io/<owner>/customer-churn-api:latest
docker run --rm -p 8000:8000 ghcr.io/<owner>/customer-churn-api:latest
```

Replace `<owner>` with the GitHub username or organisation that owns the repository.

---

Previous: [Docker and Containerisation ←](13-docker.md) | Back to [Index →](index.md)
