# 08 — Model Evaluation

**Source file:** [`customer-churn/src/evaluate.py`](../customer-churn/src/evaluate.py)

## Responsibilities

`evaluate.py` is a standalone script that loads a model from a local `.joblib` file and prints a full classification report against the test set. It is independent of MLflow.

## When to Use This Script

This script is useful when you have saved a model locally (e.g., with `joblib.dump(model, "models/random_forest.joblib")`) and want a quick evaluation without connecting to MLflow. It is a simpler alternative to the MLflow-based evaluation that `train.py` performs automatically.

> **Note:** The current pipeline in `train.py` does not save a `.joblib` file — it logs the model to MLflow. To use `evaluate.py`, you would need to add a `joblib.dump()` call to `train.py` or save the model separately.

## The `evaluate()` Function

```python
def evaluate():
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    test_path = PROJECT_ROOT / config["data"]["test_path"]
    model_path = PROJECT_ROOT / "models" / "random_forest.joblib"
    test_data = pd.read_csv(test_path)
    model = joblib.load(model_path)
    predictions = model.predict(test_data[FEATURE_COLUMNS])

    print(classification_report(test_data[TARGET_COLUMN], predictions))
```

The function:
1. Loads the config to find the test CSV path.
2. Loads the model from `models/random_forest.joblib`.
3. Reads the test CSV.
4. Generates predictions.
5. Prints a classification report.

## Understanding the Classification Report

`sklearn.metrics.classification_report` prints a table with per-class and overall metrics:

```
              precision    recall  f1-score   support

           0       0.85      0.92      0.88       160
           1       0.72      0.58      0.64        40

    accuracy                           0.84       200
   macro avg       0.79      0.75      0.76       200
weighted avg       0.82      0.84      0.83       200
```

| Metric | Definition |
|--------|-----------|
| **Precision** | True positives / (True positives + False positives). "Of all predicted churners, how many actually churned?" |
| **Recall** | True positives / (True positives + False negatives). "Of all actual churners, how many did we catch?" |
| **F1-score** | Harmonic mean of precision and recall. A single balanced metric |
| **Support** | Number of actual instances of each class in the test set |
| **Macro avg** | Unweighted average across classes |
| **Weighted avg** | Average weighted by class support |

## Precision vs. Recall Trade-off

For churn prediction, **recall** is often more important than precision. Missing a churner (false negative) means losing a customer. Incorrectly flagging a loyal customer (false positive) means wasting a retention effort. The right balance depends on the cost of each type of error in the business context.

For a complete mathematical treatment of all metrics — including the confusion matrix, F1-score formula, class imbalance effects, and how to interpret macro vs. weighted averages — see [Classification Metrics Deep Dive →](B-classification-metrics.md).

## Execution

```bash
cd customer-churn
python src/evaluate.py
```

This will fail with a `FileNotFoundError` unless `models/random_forest.joblib` exists. The current pipeline does not create this file automatically.

---

Previous: [Model Training ←](07-model-training.md) | Next: [Prediction API →](09-prediction-api.md)
