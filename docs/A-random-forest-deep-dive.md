# A — Random Forest: Theory and Implementation

This document explains the Random Forest algorithm from first principles — decision trees, ensemble methods, bootstrap sampling, and the voting mechanism — and then shows exactly how the project implements it.

Related documents: [Model Training](07-model-training.md) | [Model Evaluation](08-model-evaluation.md) | [Classification Metrics](B-classification-metrics.md)

---

## 1. The Building Block: Decision Trees

A **decision tree** is a flowchart-like model that makes predictions by asking a sequence of yes/no questions about the input features.

### How a Tree Makes a Decision

Given a customer with `account_age=12`, `monthly_spend=89.5`, `support_tickets=3`, a tree might look like:

```
Is support_tickets > 2?
├── YES → Is account_age < 24?
│         ├── YES → CHURN (leaf node, prediction = 1)
│         └── NO  → STAY  (leaf node, prediction = 0)
└── NO  → STAY (leaf node, prediction = 0)
```

Each internal node is a **split**: a threshold on one feature. Each leaf node is a **prediction**.

### How a Tree Is Built: Recursive Splitting

Training a decision tree means finding the best splits. At each node, the algorithm searches over all features and all possible thresholds to find the split that best separates the classes.

**What "best" means — Gini Impurity:**

The most common criterion for classification trees is **Gini impurity**, which measures how often a randomly chosen element from a node would be incorrectly classified if it were randomly labelled according to the class distribution at that node.

For a node containing samples from `K` classes, where `p_k` is the proportion of samples belonging to class `k`:

```
Gini(node) = 1 - Σ(p_k²)   for k = 1 to K
```

For binary classification (churn = 0 or 1):

```
Gini(node) = 1 - (p_0² + p_1²)
```

- A **pure node** (all samples are the same class) has Gini = 0. Example: all 50 samples are churners → `1 - (0² + 1²) = 0`.
- A **maximally impure node** (50/50 split) has Gini = 0.5. Example: 25 churners, 25 non-churners → `1 - (0.5² + 0.5²) = 0.5`.

**Gini gain from a split:**

When a node with `N` samples is split into a left child (N_L samples) and a right child (N_R samples):

```
Gini_gain = Gini(parent) - (N_L/N) × Gini(left) - (N_R/N) × Gini(right)
```

The algorithm picks the feature and threshold that **maximises** Gini gain at each node.

### Stopping Conditions

A tree keeps splitting until one of these conditions is met:
- The node is pure (Gini = 0).
- The node has fewer samples than `min_samples_split` (default: 2).
- The tree has reached `max_depth`.

In this project, `max_depth=8` is set in `config.yaml`, which limits each tree to at most 8 levels of splits.

### The Overfitting Problem

A single deep decision tree tends to **overfit**: it memorises the training data perfectly but generalises poorly to new data. A tree with no depth limit can create a unique leaf for every training sample, achieving 100% training accuracy but near-random test accuracy.

This is the core motivation for Random Forests.

---

## 2. From One Tree to a Forest: Ensemble Learning

**Ensemble learning** combines multiple models to produce a better prediction than any single model. The key insight is that if individual models make *different* errors, averaging their predictions cancels out those errors.

Random Forests use two sources of randomness to ensure the trees are different from each other:

1. **Bootstrap sampling** — each tree is trained on a different random sample of the data.
2. **Feature subsampling** — each split considers only a random subset of features.

### Bootstrap Sampling (Bagging)

**Bootstrap sampling** (also called **bagging** — Bootstrap AGGregating) works as follows:

Given a training set of `N` samples:
1. Draw `N` samples **with replacement** from the training set.
2. Train one decision tree on this bootstrap sample.
3. Repeat for each of the `n_estimators` trees.

"With replacement" means the same sample can appear multiple times in a bootstrap sample, and some samples will not appear at all. On average, each bootstrap sample contains about **63.2%** of the unique original samples (the rest are duplicates). The ~36.8% of samples not selected are called **out-of-bag (OOB)** samples.

Why does this help? Each tree sees a slightly different dataset, so each tree makes different errors. When you average the predictions, the errors tend to cancel out.

### Feature Subsampling (Random Subspace Method)

At each split in each tree, instead of considering all features, the algorithm considers only a random subset of `max_features` features.

For classification, scikit-learn's default is `max_features = sqrt(n_features)`. With 3 features in this project, `sqrt(3) ≈ 1.73`, so each split considers either 1 or 2 features.

This further decorrelates the trees: even if one feature is very strong (like `support_tickets`), not every split will use it, forcing other features to contribute.

---

## 3. Making a Prediction: Majority Vote

Once all `n_estimators` trees are trained, a prediction for a new sample works as follows:

**For `predict()` (hard label):**
1. Each tree independently predicts a class (0 or 1).
2. The final prediction is the **majority vote**: the class predicted by more than half the trees.

```
Tree 1: CHURN (1)
Tree 2: CHURN (1)
Tree 3: STAY  (0)
Tree 4: CHURN (1)
Tree 5: STAY  (0)
...
Tree 200: CHURN (1)

Majority vote → CHURN (1)
```

**For `predict_proba()` (soft probability):**
1. Each tree predicts a class probability (the fraction of training samples in the leaf node that belong to each class).
2. The final probability is the **average** across all trees.

```
churn_probability = (1/n_estimators) × Σ P_tree_i(churn=1)
```

This is what `api.py` uses: `model.predict_proba(features)[0][positive_class_index]`.

---

## 4. Why Random Forests Work: Bias-Variance Trade-off

Every model makes two types of errors:

- **Bias**: systematic error from wrong assumptions. A model that always predicts "no churn" has high bias.
- **Variance**: sensitivity to small fluctuations in the training data. A deep decision tree has high variance — change a few training samples and the tree structure changes dramatically.

The **bias-variance trade-off** says that reducing one often increases the other.

Random Forests reduce variance without significantly increasing bias:
- Individual deep trees have low bias but high variance.
- Averaging many uncorrelated trees reduces variance (errors cancel out) while keeping bias low.

Mathematically, if each tree has variance `σ²` and the trees are uncorrelated, the variance of the average of `n` trees is `σ²/n`. In practice, trees are correlated (they share training data and features), so the reduction is less dramatic, but still substantial.

---

## 5. Hyperparameters in This Project

The three hyperparameters set in `configs/config.yaml` and logged to MLflow:

### `n_estimators = 200`

The number of trees in the forest. More trees generally improve performance up to a point, after which returns diminish. 200 is a reasonable default for a dataset of 1,000 samples.

**Effect:** More trees → lower variance → more stable predictions. The cost is longer training time (linear in `n_estimators`).

### `max_depth = 8`

The maximum depth of each tree. This is the primary regularisation parameter.

**Effect:** Shallower trees → higher bias, lower variance. Deeper trees → lower bias, higher variance. `max_depth=8` allows trees to capture complex interactions while preventing extreme overfitting on 800 training samples.

**Why 8?** With 3 features and 800 samples, a depth-8 tree can have up to `2^8 = 256` leaf nodes. Since the training set has 800 samples, this means an average of ~3 samples per leaf — a reasonable balance.

### `random_state = 42`

The seed for all random operations: bootstrap sampling, feature subsampling, and tie-breaking. Setting this ensures that running `train.py` twice produces identical models.

---

## 6. Feature Importance

A useful by-product of Random Forest training is **feature importance**: a score for each feature indicating how much it contributed to reducing impurity across all trees.

For each feature `f`, its importance is:

```
Importance(f) = Σ (N_node / N_total) × Gini_gain(node)
```

summed over all nodes in all trees where feature `f` was used for the split, then normalised so all importances sum to 1.

In this project, `support_tickets` is expected to have the highest importance because it has the largest coefficient (`0.45`) in the logistic model used to generate the churn labels.

> The project does not currently log or display feature importances. They can be accessed after training with `model.feature_importances_`.

---

## 7. Implementation in This Project

### Training (`train.py`)

```python
model_parameters = {
    "n_estimators": config["model"]["n_estimators"],   # 200
    "max_depth": config["model"]["max_depth"],          # 8
    "random_state": config["model"]["random_state"],    # 42
}
model = RandomForestClassifier(**model_parameters)
model.fit(train_data[FEATURE_COLUMNS], train_data[TARGET_COLUMN])
```

`model.fit(X, y)` triggers the full training process:
1. For each of the 200 trees: draw a bootstrap sample, build a tree using Gini impurity with `max_depth=8` and `max_features=sqrt(3)`.
2. Store all 200 fitted trees in `model.estimators_`.

### Prediction (`api.py`)

```python
features = pd.DataFrame([customer.model_dump()], columns=FEATURE_COLUMNS)
churn_label = int(model.predict(features)[0])
positive_class_index = list(model.classes_).index(1)
churn_probability = float(model.predict_proba(features)[0][positive_class_index])
```

`model.predict(features)` runs the majority vote across all 200 trees.  
`model.predict_proba(features)` averages the per-leaf class fractions across all 200 trees.

### Model Serialisation (`train.py`)

```python
mlflow.sklearn.log_model(
    model,
    name=MODEL_ARTIFACT_NAME,
    skops_trusted_types=["sklearn.tree._tree.Tree"],
)
```

MLflow serialises the `RandomForestClassifier` object (including all 200 trees) using **skops** — a secure serialisation format for scikit-learn models. The `skops_trusted_types` argument explicitly whitelists the internal C-extension type `sklearn.tree._tree.Tree`, which is required because skops performs security checks on deserialisation to prevent arbitrary code execution via malicious pickle files.

---

## 8. Advantages and Limitations

### Advantages for This Use Case

| Advantage | Relevance |
|-----------|-----------|
| No feature scaling required | `account_age` (1–72), `monthly_spend` (20–250), and `support_tickets` (0–10+) are on very different scales. Trees split on thresholds, not distances, so scaling is irrelevant. |
| Handles mixed feature types | Integer and float features work without preprocessing. |
| Robust to outliers | A single outlier in `monthly_spend` affects at most one split in one tree. |
| Provides probability estimates | `predict_proba()` gives a calibrated probability, not just a binary label. |
| Interpretable feature importance | Can explain which features drive predictions. |
| Resistant to overfitting | Ensemble averaging reduces variance compared to a single tree. |

### Limitations

| Limitation | Impact |
|-----------|--------|
| Black box | Individual tree paths can be inspected, but 200 trees are hard to interpret holistically. |
| Memory usage | Storing 200 trees with up to 256 leaf nodes each requires significant memory for large datasets. |
| Slow on very large datasets | Training is `O(n_estimators × N × log(N) × sqrt(p))` where `N` is samples and `p` is features. |
| Not ideal for very sparse data | Gradient boosting methods (XGBoost, LightGBM) often outperform Random Forests on sparse tabular data. |

### Alternatives Considered

| Algorithm | Why Not Used Here |
|-----------|------------------|
| Logistic Regression | Simpler and more interpretable, but assumes linear decision boundaries. The Random Forest can capture non-linear interactions. |
| Gradient Boosting (XGBoost) | Often achieves higher accuracy, but more hyperparameters to tune and slower to train. |
| Neural Network | Overkill for 3 features and 1,000 samples; requires feature scaling and much more data. |
| Single Decision Tree | High variance; Random Forest is strictly better with minimal added complexity. |

---

Previous: [Model Training ←](07-model-training.md) | Next: [Classification Metrics →](B-classification-metrics.md)
