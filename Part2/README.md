# Part 2 – Supervised Machine Learning: Build, Train, and Evaluate Models

## Project Overview

In this part of the project, supervised machine learning techniques were applied to the cleaned hospital dataset generated in **Part 1**. The objective was to build both regression and classification models, evaluate their performance, compare different algorithms, and interpret the results.

Two prediction tasks were performed:

- **Regression:** Predict the patient's **TotalCharge**.
- **Classification:** Predict whether a patient will be **readmitted to the hospital (ReAdmis)**.

The entire workflow was implemented using **Python, Pandas, NumPy, Scikit-learn, and Matplotlib**.

---

# Dataset

- **Dataset:** `cleaned_data.csv`
- **Source:** Generated from Part 1
- **Total Records:** 10,000
- **Missing Values:** 0
- **Duplicate Records:** 0

The dataset was verified before model development to ensure that all preprocessing from Part 1 had been completed successfully.

---

# Target Variables

## Regression Target

**TotalCharge**

TotalCharge is a continuous numerical variable representing the total hospital charges incurred by a patient. It was selected as the regression target because regression algorithms predict continuous numerical values.

---

## Classification Target

**ReAdmis**

The ReAdmis column was converted into binary values:

- Yes → 1
- No → 0

This binary variable was selected because Logistic Regression requires binary target labels.

---

# Feature Engineering

The feature matrix (**X**) was created by removing the target columns from the dataset.

## Ordinal Encoding

The following columns contain categories with natural ordering and were encoded using Label Encoding:

- Education
- Complication_risk

Since these variables have meaningful order, ordinal encoding preserves that relationship.

---

## One-Hot Encoding

The remaining categorical variables were encoded using One-Hot Encoding.

Examples include:

- City
- State
- County
- Area
- Job
- Gender
- Employment
- Marital
- Services

The first dummy column was dropped to avoid multicollinearity.

One-Hot Encoding prevents the model from assuming an artificial numerical order among categories.

---

# Train-Test Split

The dataset was divided into:

- **Training Set:** 80%
- **Testing Set:** 20%

Random State:

```python
random_state = 42
```

This ensures reproducibility.

---

# Feature Scaling

Feature scaling was performed using **StandardScaler**.

The scaler was fitted **only on the training dataset** and then used to transform both the training and testing datasets.

This prevents **data leakage**, because fitting the scaler on the complete dataset would allow information from the testing data to influence the training process.

---

# Regression Model

## Linear Regression

The Linear Regression model was trained to predict **TotalCharge**.

### Performance

| Metric | Value |
|---------|-------:|
| Mean Squared Error (MSE) | 6,715,194.74 |
| R² Score | 0.4114 |

The model explains approximately **41.14%** of the variation in TotalCharge.

---

## Top 3 Important Features

| Feature | Coefficient |
|----------|------------:|
| VitD_levels | 2465.867 |
| Initial_days | 1667.847 |
| Zip | -1131.613 |

### Interpretation

- Positive coefficients increase the predicted TotalCharge.
- Negative coefficients decrease the predicted TotalCharge.

---

# Ridge Regression

A Ridge Regression model was trained using:

```python
alpha = 1.0
```

### Model Comparison

| Model | MSE | R² Score |
|------|------------:|--------:|
| Linear Regression | 6,715,194.74 | 0.4114 |
| Ridge Regression | 3,691,419.00 | 0.6765 |

### Interpretation

Ridge Regression reduced prediction error and significantly improved the R² score.

Ridge Regression applies **L2 Regularization**, which shrinks large coefficients and reduces overfitting.

The **alpha** parameter controls the strength of regularization.

---

# Classification Model

## Logistic Regression

The Logistic Regression model predicts hospital readmission.

### Class Distribution

| Class | Percentage |
|-------|-----------:|
| No | 63% |
| Yes | 37% |

Since the minority class exceeded **35%**, no imbalance handling technique (SMOTE or class weighting) was required.

---

# Classification Performance

## Confusion Matrix

| | Predicted No | Predicted Yes |
|---|-------------:|--------------:|
| Actual No | 1102 | 189 |
| Actual Yes | 148 | 561 |

---

## Evaluation Metrics

| Metric | Value |
|---------|-------:|
| Accuracy | 83.15% |
| Precision | 74.80% |
| Recall | 79.13% |
| F1 Score | 76.90% |

### Interpretation

The Logistic Regression model correctly classified **83.15%** of all patients.

For hospital readmission prediction, **Recall** is particularly important because failing to identify patients who are likely to be readmitted could delay follow-up care.

---

# ROC Curve

The Receiver Operating Characteristic (ROC) curve was generated using predicted probabilities.

## AUC Score

**0.898**

### Interpretation

An AUC score of **0.898** indicates excellent classification performance.

The ROC curve lies well above the diagonal reference line, showing that the classifier distinguishes effectively between patients who are readmitted and those who are not.

---

# Decision Threshold Analysis

Different probability thresholds were evaluated.

| Threshold | Precision | Recall | F1 Score |
|----------:|----------:|--------:|---------:|
| 0.30 | 0.7244 | 0.8265 | **0.7721** |
| 0.40 | 0.7403 | 0.8039 | 0.7708 |
| 0.50 | 0.7480 | 0.7913 | 0.7690 |
| 0.60 | 0.7558 | 0.7772 | 0.7663 |
| 0.70 | 0.7652 | 0.7630 | 0.7641 |

### Best Threshold

The highest **F1-score (0.7721)** was achieved at a threshold of **0.30**.

As the threshold increased:

- Precision increased.
- Recall decreased.

For hospital readmission prediction, **Recall** is more important because identifying patients at risk is more valuable than minimizing false positives.

---

# Precision and Recall

### Precision

```text
Precision = TP / (TP + FP)
```

### Recall

```text
Recall = TP / (TP + FN)
```

---

# Logistic Regression Regularization

Two Logistic Regression models were compared.

| Model | Precision | Recall | AUC |
|------|----------:|--------:|----:|
| Logistic Regression (C = 1.0) | 0.7652 | 0.7630 | 0.8978 |
| Logistic Regression (C = 0.01) | 0.7720 | 0.7786 | 0.9073 |

### Interpretation

The model with **C = 0.01** produced higher Precision, Recall, and AUC.

The parameter **C** controls the inverse strength of regularization.

- Larger C → Less regularization
- Smaller C → Stronger regularization

Reducing **C** improved the model's ability to generalize to unseen data.

---

# Bootstrap Confidence Interval

A bootstrap experiment with **500 resampled datasets** was performed.

For each bootstrap sample, the difference in AUC between the two Logistic Regression models was calculated.

The resulting **95% confidence interval excluded zero**, indicating that the performance improvement obtained using **C = 0.01** is statistically reliable and unlikely to be due to random variation.

---

# Libraries Used

- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

# Output Files

This project generates:

- Linear Regression model
- Ridge Regression model
- Logistic Regression model
- Confusion Matrix
- ROC Curve
- Threshold Comparison Table
- Bootstrap Confidence Interval Results

---

# Conclusion

This project successfully implemented supervised machine learning techniques for both regression and classification tasks.

Among the regression models, **Ridge Regression** outperformed Linear Regression by reducing prediction error and improving the R² score.

For classification, **Logistic Regression** achieved an **Accuracy of 83.15%** and an **AUC of 0.898**, demonstrating excellent predictive performance. The strongly regularized Logistic Regression model (**C = 0.01**) further improved Precision, Recall, and AUC. Bootstrap analysis confirmed that this improvement is statistically reliable.

Overall, the project demonstrates a complete supervised machine learning workflow, including preprocessing, feature engineering, model training, evaluation, threshold analysis, regularization, and statistical validation.