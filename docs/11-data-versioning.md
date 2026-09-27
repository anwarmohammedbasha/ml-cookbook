# 11 — Data Versioning with DVC

**Source file:** [`customer-churn/setup_data_tracking.sh`](../customer-churn/setup_data_tracking.sh)

## What Is Data Versioning?

In software engineering, Git tracks changes to source code. But Git is not designed for large binary files or datasets — committing a 50 MB CSV to Git bloats the repository permanently and makes cloning slow.

**Data versioning** solves this by separating the *reference* to a dataset from the *contents* of that dataset:

- The reference (a small pointer file) is committed to Git.
- The contents (the actual data) are stored elsewhere — in a local cache or a remote storage bucket.

When a teammate clones the repository, they get the pointer file. They then run `dvc pull` to download the actual data from the remote.

## What Is DVC?

**DVC** (Data Version Control) is an open-source tool that implements data versioning alongside Git. It works by:

1. Computing a hash (MD5 checksum) of the data file or directory.
2. Writing a `.dvc` pointer file that records the hash and the original path.
3. Moving the actual data into a local cache (`.dvc/cache/`).
4. Creating a `.gitignore` entry so Git ignores the original data path.

The `.dvc` pointer file is small (a few lines of YAML) and is committed to Git. The data itself is never committed.

## How DVC Is Used in This Project

The project tracks the `data/processed/` directory — the train and test CSVs produced by `data_processor.py`.

### `setup_data_tracking.sh`

```bash
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
```

**`set -euo pipefail`** — Strict mode: exit on any error (`-e`), treat unset variables as errors (`-u`), and propagate pipe failures (`-o pipefail`).

**`dvc init --subdir`** — Initialises DVC inside the `customer-churn/` subdirectory rather than the repository root. The `--subdir` flag is needed because the Git repository root is one level up.

**`dvc add data/processed`** — Hashes the directory, moves its contents to the DVC cache, and creates `data/processed.dvc`.

### What `dvc add` Creates

After running the script, three new files appear:

| File | Committed to Git? | Purpose |
|------|-------------------|---------|
| `data/processed.dvc` | Yes | Pointer file with the directory hash |
| `data/.gitignore` | Yes | Tells Git to ignore `data/processed/` |
| `.dvc/` directory | Yes (metadata only) | DVC configuration and cache index |

### The Pointer File (`data/processed.dvc`)

```yaml
outs:
- md5: abc123...
  size: 12345
  nfiles: 2
  hash: md5
  path: data/processed
```

This file records the MD5 hash of the entire `data/processed/` directory. If the data changes (e.g., you regenerate it with a different seed), the hash changes, and the new pointer file can be committed to Git — creating a new version of the data.

## Committing DVC Metadata

After running the script, commit the generated files:

```bash
git add .dvc .dvcignore data/.gitignore data/processed.dvc
git commit -m "chore: track processed data with DVC"
```

## Configuring a Remote and Pushing Data

A DVC remote is where the actual data is stored for sharing. Common options include S3, GCS, Azure Blob Storage, or an SSH server.

```bash
# Add a remote (example using S3)
dvc remote add -d storage s3://my-bucket/dvc-store

# Push data to the remote
dvc push
```

Commit the remote configuration (but only if it contains no credentials):

```bash
git add .dvc/config
git commit -m "chore: configure DVC remote"
```

## Pulling Data on a New Machine

```bash
git clone <repo-url>
cd customer-churn
dvc pull  # downloads data/processed/ from the configured remote
```

## `.gitignore` Integration

The root `.gitignore` uses a careful pattern to allow the DVC pointer file while blocking the actual data:

```gitignore
customer-churn/data/*          # ignore everything in data/
!customer-churn/data/processed.dvc   # except the DVC pointer file
!customer-churn/data/.gitignore      # and the DVC-generated gitignore
```

---

Previous: [Remote Prediction ←](10-remote-prediction.md) | Next: [Testing →](12-testing.md)
