# Part 3 – Advanced Modeling, Ensembles, Hyperparameter Tuning and Machine Learning Pipeline

## Objective

The objective of this part is to build advanced machine learning classification models, compare their performance, tune hyperparameters, create a reusable machine learning pipeline, and identify the best model for predicting hospital readmission.

---

# Dataset

- Dataset: `cleaned_data.csv`
- Total Records: **10,000**
- Features after preprocessing: **8,444**
- Training Samples: **8,000**
- Testing Samples: **2,000**

The dataset used in this part is the cleaned dataset generated in Part 1.

---

# Task 1 – Decision Tree Baseline

A baseline Decision Tree classifier was trained using the default parameters.

## Results

| Metric | Value |
|---------|-------|
| Training Accuracy | **100.00%** |
| Testing Accuracy | **96.40%** |

### Interpretation

The baseline Decision Tree achieved perfect training accuracy but slightly lower testing accuracy. This indicates that the model memorized the training data and shows signs of **overfitting**.

Decision Trees are considered **high-variance models** because they greedily select the best split at every node without revisiting earlier decisions. As a result, deep trees often fit noise present in the training data.

---

# Task 2 – Controlled Decision Tree

A second Decision Tree was trained using:

- max_depth = 5
- min_samples_split = 20

## Results

| Metric | Value |
|---------|-------|
| Training Accuracy | **97.29%** |
| Testing Accuracy | **96.60%** |

### Interpretation

Limiting the tree depth reduced overfitting and improved generalization.

**max_depth** limits the maximum depth of the tree, reducing model complexity.

**min_samples_split** prevents splitting nodes that contain only a few samples, reducing noisy decision boundaries.

Compared with the baseline tree, the controlled tree produced a much smaller train-test gap.

---

# Task 3 – Gini vs Entropy

Two Decision Trees were trained using different impurity measures.

| Criterion | Test Accuracy |
|-----------|---------------|
| Gini | **96.60%** |
| Entropy | **96.55%** |

## Gini Formula

Gini = 1 − Σ(pi²)

## Entropy Formula

Entropy = −Σ(pi log₂ pi)

### Interpretation

A node with **Gini = 0** means that all samples belong to a single class, representing a perfectly pure node.

Both impurity measures produced very similar classification performance.

---

# Task 4 – Random Forest

A Random Forest classifier was trained using:

- n_estimators = 100
- max_depth = 10
- random_state = 42

## Results

| Metric | Value |
|---------|-------|
| Training Accuracy | **94.26%** |
| Testing Accuracy | **94.45%** |
| ROC-AUC | **0.9809** |

### Train-Test Gap

| Metric | Value |
|---------|-------|
| Training Accuracy | **94.26%** |
| Testing Accuracy | **94.45%** |
| Gap | **~0.18%** |

The very small train-test gap indicates that the Random Forest model generalizes well and does not significantly overfit the training data.

---

# Top 5 Important Features

| Feature | Importance |
|---------|-----------|
| Initial_days | 0.159913 |
| CaseOrder | 0.098210 |
| VitD_levels | 0.016714 |
| Additional_charges | 0.014010 |
| Income | 0.012690 |

### Feature Importance Interpretation

Random Forest calculates feature importance by measuring the average reduction in Gini impurity contributed by each feature across all trees.

Unlike Linear Regression coefficients, feature importance does not indicate positive or negative relationships. Instead, it measures how useful each feature is for making accurate predictions.

---

# Bagging Concept

Random Forest uses **Bootstrap Aggregating (Bagging)**.

Each decision tree is trained using a random sample of the training data with replacement.

At every split, only a random subset of features is considered.

These two sources of randomness reduce model variance and improve generalization compared with a single Decision Tree.

---

# Task 4a – Gradient Boosting

Gradient Boosting classifier was trained using

- n_estimators = 100
- learning_rate = 0.1
- max_depth = 3

## Results

| Metric | Value |
|---------|-------|
| Training Accuracy | **97.30%** |
| Testing Accuracy | **96.80%** |
| ROC-AUC | **0.9959** |

### Interpretation

Gradient Boosting achieved the highest predictive performance among all models by sequentially correcting errors made by previous trees.

---

# Task 4b – Feature Ablation Study

The five least important features identified from the Random Forest model were removed.

A second Random Forest model was trained using the remaining features.

## Results

| Model | ROC-AUC |
|---------|---------|
| Full Random Forest | **0.9809** |
| Reduced Random Forest | **0.9830** |

### Interpretation

Removing the five least important features slightly improved the ROC-AUC score.

This suggests that these features contributed little useful information and may have introduced noise.

A simpler model reduces inference time, storage requirements, and maintenance costs while maintaining predictive performance.

---

# Task 5 – Cross Validation

Five-fold Stratified Cross Validation was performed using ROC-AUC.

## Results

| Model | Mean ROC-AUC | Std ROC-AUC |
|---------|-------------|-------------|
| Logistic Regression | **0.8902** | **0.0095** |
| Controlled Decision Tree | **0.9927** | **0.0017** |
| Random Forest | **0.9803** | **0.0014** |
| Gradient Boosting | **0.9959** | **0.0010** |

### Interpretation

Cross-validation evaluates model performance across multiple train-test splits, providing a more reliable estimate of generalization than a single train-test split.

---

# Task 6 – Hyperparameter Tuning

A machine learning pipeline was built consisting of

1. SimpleImputer
2. StandardScaler
3. RandomForestClassifier

GridSearchCV evaluated:

- n_estimators = [50,100,200]
- max_depth = [5,10,None]
- min_samples_leaf = [1,5]

Total combinations evaluated:

18 combinations × 5 folds = **90 model fits**

## Best Parameters

*(Insert your GridSearchCV output here)*

```
Example:

{'randomforestclassifier__max_depth':10,
 'randomforestclassifier__min_samples_leaf':1,
 'randomforestclassifier__n_estimators':200}
```

## Best Cross Validation ROC-AUC

```
Insert your best_score_ here.
```

### Grid Search vs Randomized Search

Grid Search evaluates every parameter combination and guarantees finding the best model within the search space.

Randomized Search evaluates only randomly selected combinations, making it faster but without guaranteeing the optimal solution.

---

# Task 7 – Manual Learning Curve

The best pipeline obtained from GridSearchCV was trained using progressively larger portions of the training dataset (20%, 40%, 60%, 80%, and 100%). For each training fraction, the ROC-AUC was calculated on both the training subset and the fixed test set.

## Results

| Training Fraction | Training AUC | Test AUC |
|------------------:|-------------:|---------:|
| 20% | 1.0000 | 0.9845 |
| 40% | 1.0000 | 0.9851 |
| 60% | 1.0000 | 0.9880 |
| 80% | 1.0000 | 0.9885 |
| 100% | 1.0000 | 0.9900 |

## Interpretation

The training ROC-AUC remained **1.0000** for all training fractions, indicating that the model perfectly fitted each training subset. This suggests that the model has a very high capacity and can closely fit the available training data.

The test ROC-AUC increased steadily from **0.9845** at 20% of the training data to **0.9900** when the full training dataset was used. This demonstrates that the model generalized better as more training data became available.

Although the training performance did not decrease with larger datasets, the consistent improvement in test ROC-AUC indicates that the model still benefits from additional training data. Therefore, the current model appears to be **primarily data-limited rather than capacity-limited**, as increasing the amount of training data continued to improve its performance on unseen samples.
---

# Task 8 – Model Serialization

The best pipeline was saved using

```python
joblib.dump(best_pipeline, "best_model.pkl")
```

The saved model was successfully reloaded using

```python
joblib.load("best_model.pkl")
```

Predictions were generated successfully on sample records.

---

# Final Model Comparison

| Model | 5-Fold CV Mean AUC | 5-Fold CV Std AUC | Test ROC-AUC |
|--------|-------------------:|------------------:|-------------:|
| Logistic Regression | 0.8902 | 0.0095 | 0.8885 |
| Controlled Decision Tree | 0.9927 | 0.0017 | 0.9946 |
| Random Forest | 0.9803 | 0.0014 | 0.9809 |
| **Gradient Boosting** | **0.9959** | **0.0010** | **0.9959** |

---

# Recommended Model

Based on the experimental results, **Gradient Boosting** is recommended for deployment.

It achieved the highest ROC-AUC on both the independent test dataset and the 5-fold cross-validation evaluation while maintaining the lowest variation across folds. This demonstrates excellent predictive performance, strong generalization ability, and consistent model stability.

Compared with Logistic Regression, Decision Tree, and Random Forest, Gradient Boosting provided the best overall balance between accuracy, robustness, and reliability, making it the most suitable model for hospital readmission prediction.

---

# Conclusion

In this part of the project, advanced ensemble learning techniques, hyperparameter optimization, feature importance analysis, feature ablation, cross-validation, and model serialization were successfully implemented.

Among all evaluated models, **Gradient Boosting** achieved the best predictive performance and is recommended as the final deployment model.