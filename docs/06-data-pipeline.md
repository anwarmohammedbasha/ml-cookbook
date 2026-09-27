# 06 — Data Pipeline

**Source file:** [`customer-churn/src/data_processor.py`](../customer-churn/src/data_processor.py)

## Responsibilities

`data_processor.py` is the first script in the pipeline. It has two jobs:

1. **Generate** a synthetic dataset of 1,000 SaaS customer records.
2. **Split** the dataset into training and test sets and write all three CSVs to disk.

## Module-Level Constants

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
FEATURE_COLUMNS = ["account_age", "monthly_spend", "support_tickets"]
TARGET_COLUMN = "churn"
```

`FEATURE_COLUMNS` and `TARGET_COLUMN` are defined here and imported by `train.py`, `evaluate.py`, and the tests, making them the single source of truth for column names.

## `generate_data(n_rows=1000, random_state=42)`

This function creates a realistic synthetic dataset using a logistic model. Here is a step-by-step breakdown.

### Step 1 — Create a Reproducible Random Number Generator

```python
rng = np.random.default_rng(random_state)
```

`np.random.default_rng` creates a new random number generator seeded with `random_state=42`. Using a fixed seed means the same dataset is produced every time the function is called with the same arguments.

### Step 2 — Generate Feature Values

```python
account_age = rng.integers(1, 73, size=n_rows)
monthly_spend = np.round(np.clip(rng.normal(100, 30, size=n_rows), 20, 250), 2)
support_tickets = rng.poisson(2.5, size=n_rows)
```

| Feature | Distribution | Range | Reasoning |
|---------|-------------|-------|-----------|
| `account_age` | Uniform integers | 1–72 months | Customers have been subscribed between 1 month and 6 years |
| `monthly_spend` | Normal (mean=100, std=30), clipped | $20–$250 | Most customers pay around $100/month; outliers are clipped |
| `support_tickets` | Poisson (λ=2.5) | 0+ | Support tickets are count data; Poisson is the natural distribution for counts |

### Step 3 — Compute Churn Probability Using a Logistic Model

```python
churn_logit = (
    -1.3
    - 0.025 * (account_age - 36)
    + 0.012 * (monthly_spend - 100)
    + 0.45 * (support_tickets - 2.5)
)
churn_probability = 1 / (1 + np.exp(-churn_logit))
```

This is a **logistic regression model** used to generate realistic labels. The logit (log-odds) is a linear combination of the features, centred around their mean values:

- The intercept `-1.3` sets the baseline churn rate to roughly 21% (when all features are at their mean).
- `account_age` has a **negative** coefficient (`-0.025`): older customers are less likely to churn.
- `monthly_spend` has a small **positive** coefficient (`+0.012`): higher-spending customers churn slightly more.
- `support_tickets` has a large **positive** coefficient (`+0.45`): more support tickets strongly predict churn.

The logistic function `1 / (1 + exp(-x))` converts the logit to a probability between 0 and 1.

For the full mathematical derivation — including the sigmoid function properties, a worked numerical example for the sample customer, and why each distribution was chosen — see [Synthetic Data Generation: Mathematics and Design →](C-synthetic-data-math.md).

### Step 4 — Sample Binary Churn Labels

```python
churn = rng.binomial(1, churn_probability)
```

For each customer, a Bernoulli trial is drawn: the customer churns (`1`) with probability `churn_probability` and stays (`0`) with probability `1 - churn_probability`. This introduces realistic noise — not every high-risk customer churns, and not every low-risk customer stays.

### Step 5 — Return a DataFrame

```python
return pd.DataFrame({
    "account_age": account_age,
    "monthly_spend": monthly_spend,
    "support_tickets": support_tickets,
    TARGET_COLUMN: churn,
})
```

## `process_data()`

This function orchestrates the full data preparation pipeline.

```python
def process_data():
    config = load_config()
    # ... resolve paths from config ...

    dataset = generate_data()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(raw_path, index=False)

    train_data, test_data = train_test_split(
        dataset,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=dataset[TARGET_COLUMN],
    )
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
```

### Key Design Decisions

**`mkdir(parents=True, exist_ok=True)`** — Creates the `data/raw/` and `data/processed/` directories if they do not exist, without raising an error if they already do. This makes the script idempotent (safe to run multiple times).

**`stratify=dataset[TARGET_COLUMN]`** — Ensures the proportion of churned customers is the same in both the training and test sets. Without stratification, a random split might put most churned customers in one set, making evaluation unreliable.

**`index=False`** — Prevents pandas from writing the DataFrame's row index as an extra column in the CSV.

## Output Files

After running `python src/data_processor.py`:

| File | Rows | Description |
|------|------|-------------|
| `data/raw/customers.csv` | 1,000 | Full dataset before splitting |
| `data/processed/train.csv` | 800 | 80% split, used for model training |
| `data/processed/test.csv` | 200 | 20% split, used for model evaluation |

## Execution

```bash
cd customer-churn
python src/data_processor.py
# Output: Saved 800 training and 200 test rows.
```

---

Previous: [Configuration ←](05-configuration.md) | Next: [Model Training →](07-model-training.md)
