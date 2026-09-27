# B — Classification Metrics: Theory and Implementation

This document explains every evaluation metric used in the project — accuracy, precision, recall, F1-score — from first principles, including the confusion matrix, the mathematical formulas, and how to interpret them in the context of churn prediction.

Related documents: [Model Training](07-model-training.md) | [Model Evaluation](08-model-evaluation.md) | [Random Forest Deep Dive](A-random-forest-deep-dive.md)

---

## 1. The Confusion Matrix

Every metric in binary classification is derived from four counts. Given a set of predictions and true labels, every sample falls into one of four categories:

|  | Predicted: STAY (0) | Predicted: CHURN (1) |
|--|---------------------|----------------------|
| **Actual: STAY (0)** | True Negative (TN) | False Positive (FP) |
| **Actual: CHURN (1)** | False Negative (FN) | True Positive (TP) |

**Definitions:**

- **True Positive (TP):** The model predicted churn, and the customer actually churned. Correct alarm.
- **True Negative (TN):** The model predicted stay, and the customer actually stayed. Correct silence.
- **False Positive (FP):** The model predicted churn, but the customer actually stayed. False alarm. Also called a **Type I error**.
- **False Negative (FN):** The model predicted stay, but the customer actually churned. Missed alarm. Also called a **Type II error**.

### Example

Suppose the model evaluates 200 test customers (160 non-churners, 40 churners):

|  | Predicted: STAY | Predicted: CHURN |
|--|-----------------|------------------|
| **Actual: STAY** | TN = 147 | FP = 13 |
| **Actual: CHURN** | FN = 17 | TP = 23 |

Total correct: 147 + 23 = 170 out of 200.

---

## 2. Accuracy

**Formula:**

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Using the example above:

```
Accuracy = (23 + 147) / 200 = 170 / 200 = 0.85
```

**What it means:** 85% of all predictions were correct.

**Why it can be misleading:** If only 20% of customers churn, a model that always predicts "stay" achieves 80% accuracy without learning anything useful. This is the **accuracy paradox** — accuracy is a poor metric for imbalanced classes.

In this project, the synthetic data generates roughly 21% churners (set by the `-1.3` intercept in the logistic model). Accuracy is logged but should not be the primary metric.

**In the code (`train.py`):**

```python
"accuracy": accuracy_score(test_data[TARGET_COLUMN], predictions),
```

`accuracy_score` computes `(TP + TN) / total`.

---

## 3. Precision

**Formula:**

```
Precision = TP / (TP + FP)
```

Using the example:

```
Precision = 23 / (23 + 13) = 23 / 36 ≈ 0.639
```

**What it means:** Of all customers the model flagged as churners, 63.9% actually churned. The rest were false alarms.

**Business interpretation:** Low precision means the retention team wastes effort on customers who were not going to churn anyway. High precision means the team's interventions are well-targeted.

**In the code (`train.py`):**

```python
"precision": precision_score(test_data[TARGET_COLUMN], predictions, zero_division=0),
```

`zero_division=0` returns 0 instead of raising a warning if the model predicts no positive cases at all (TP + FP = 0).

---

## 4. Recall (Sensitivity)

**Formula:**

```
Recall = TP / (TP + FN)
```

Using the example:

```
Recall = 23 / (23 + 17) = 23 / 40 = 0.575
```

**What it means:** Of all customers who actually churned, the model caught 57.5% of them. The other 42.5% were missed.

**Business interpretation:** Low recall means many churners slip through undetected and cancel without any intervention. For churn prediction, missing a churner (FN) is typically more costly than a false alarm (FP), so recall is usually the more important metric.

**Also called:** Sensitivity, True Positive Rate (TPR), Hit Rate.

**In the code (`train.py`):**

```python
"recall": recall_score(test_data[TARGET_COLUMN], predictions, zero_division=0),
```

---

## 5. F1-Score

Precision and recall trade off against each other: you can increase precision by only predicting churn when very confident (but you'll miss more churners), or increase recall by predicting churn more aggressively (but you'll have more false alarms).

The **F1-score** is the harmonic mean of precision and recall, providing a single balanced metric:

**Formula:**

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Or equivalently:

```
F1 = 2TP / (2TP + FP + FN)
```

Using the example:

```
F1 = 2 × (0.639 × 0.575) / (0.639 + 0.575)
   = 2 × 0.367 / 1.214
   ≈ 0.605
```

**Why harmonic mean, not arithmetic mean?** The harmonic mean penalises extreme imbalances. If precision = 1.0 and recall = 0.0, the arithmetic mean is 0.5 (misleadingly high), but the harmonic mean is 0 (correctly indicating the model is useless).

**F1 is not logged by `train.py`** — only accuracy, precision, and recall are. The full classification report (which includes F1) is printed by `evaluate.py`.

---

## 6. The Precision-Recall Trade-off

The model's threshold for predicting "churn" is 0.5 by default: if `churn_probability ≥ 0.5`, predict churn. Changing this threshold shifts the precision-recall balance:

| Threshold | Effect |
|-----------|--------|
| Lower (e.g., 0.3) | More customers flagged as churners → higher recall, lower precision |
| Higher (e.g., 0.7) | Fewer customers flagged → lower recall, higher precision |

The project uses the default 0.5 threshold. In a real deployment, the optimal threshold would be chosen based on the relative cost of false positives vs. false negatives.

---

## 7. Macro vs. Weighted Averages

The classification report from `evaluate.py` shows two types of averages across classes:

**Macro average:** Unweighted mean across classes.

```
Macro Precision = (Precision_class_0 + Precision_class_1) / 2
```

This treats both classes equally regardless of how many samples each has.

**Weighted average:** Mean weighted by the number of samples in each class (the `support`).

```
Weighted Precision = (Precision_class_0 × support_0 + Precision_class_1 × support_1) / total
```

With 160 non-churners and 40 churners in the test set, the weighted average is dominated by class 0 performance.

**Which to use?** For imbalanced classes, the weighted average reflects overall performance on the actual distribution. The macro average is better when you care equally about both classes.

---

## 8. Class Imbalance in This Project

The synthetic data generates approximately 21% churners. This means the test set has roughly:
- 160 non-churners (class 0)
- 40 churners (class 1)

This is a **mild class imbalance**. The Random Forest handles it reasonably well without any special treatment. For more severe imbalances (e.g., 1% churn rate), techniques like:
- **Class weighting** (`class_weight='balanced'` in scikit-learn)
- **Oversampling** (SMOTE)
- **Undersampling**

would be needed. This project does not implement any of these, which is appropriate for the ~21% churn rate.

---

## 9. Where Metrics Appear in the Code

| Metric | Logged to MLflow | Printed by evaluate.py | Notes |
|--------|-----------------|------------------------|-------|
| Accuracy | Yes (`train.py`) | Yes (in report) | Misleading for imbalanced data |
| Precision | Yes (`train.py`) | Yes (per class + avg) | For class 1 (churn) |
| Recall | Yes (`train.py`) | Yes (per class + avg) | For class 1 (churn) |
| F1-score | No | Yes (per class + avg) | Not logged to MLflow |
| Support | No | Yes | Count of actual samples per class |

**`train.py` logs precision and recall for the positive class (churn=1) only**, using scikit-learn's default `average='binary'` setting. `evaluate.py`'s `classification_report` shows per-class metrics for both classes.

---

## 10. Interpreting a Sample Report

```
              precision    recall  f1-score   support

           0       0.92      0.92      0.92       160
           1       0.69      0.70      0.70        40

    accuracy                           0.87       200
   macro avg       0.81      0.81      0.81       200
weighted avg       0.87      0.87      0.87       200
```

Reading this report:

- **Class 0 (stay):** 92% precision, 92% recall. The model is very good at identifying customers who will stay.
- **Class 1 (churn):** 69% precision, 70% recall. Of 40 actual churners, the model catches ~28 (70% recall). Of all predicted churners, ~69% actually churned.
- **Accuracy 87%:** 174 out of 200 correct. Looks good, but is partly inflated by the easy majority class.
- **Macro avg 81%:** Equal-weight average. The model performs meaningfully worse on the minority class.
- **Weighted avg 87%:** Dominated by class 0 performance due to 160 vs. 40 samples.

The key business metric is **class 1 recall**: are we catching enough churners to make retention efforts worthwhile?

---

Previous: [Random Forest Deep Dive ←](A-random-forest-deep-dive.md) | Next: [Synthetic Data Generation →](C-synthetic-data-math.md)
