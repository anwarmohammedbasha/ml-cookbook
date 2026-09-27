#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -d data/processed ]]; then
    printf 'Processed data directory not found. Run src/data_processor.py first.\n' >&2
    exit 1
fi

if [[ ! -d .dvc ]]; then
    dvc init --subdir
fi

dvc add data/processed

cat <<'INSTRUCTIONS'
DVC metadata is ready. Review it, then commit these tracking files to Git:

  git add .dvc .dvcignore data/.gitignore data/processed.dvc
  git commit -m "chore: track processed data with DVC"

The dataset contents remain in DVC's cache; push them to your configured remote with:

  dvc push
INSTRUCTIONS