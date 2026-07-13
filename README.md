# Medical Insurance Readmission Prediction using Machine Learning and LLM

## Project Overview

This project demonstrates a complete end-to-end Machine Learning workflow for predicting hospital readmissions using the Medical Insurance dataset. The project covers data preprocessing, exploratory data analysis, supervised learning, advanced ensemble models, hyperparameter tuning, model deployment preparation, and Large Language Model (LLM) integration for prediction explanations.

The implementation is divided into four parts, each focusing on a different stage of the machine learning lifecycle.

---

# Project Objectives

- Perform data cleaning and preprocessing.
- Explore and visualize the dataset.
- Build supervised machine learning models.
- Compare multiple classification algorithms.
- Apply ensemble learning techniques.
- Perform hyperparameter tuning using GridSearchCV.
- Build reusable Scikit-learn pipelines.
- Serialize the best-performing model.
- Integrate a Large Language Model (LLM) for prediction explanations.
- Validate structured JSON responses using JSON Schema.

---

# Repository Structure

```
Medical-Insurance-Readmission-Prediction/
│
├── README.md
│
├── Part1/
│   ├── Part1.ipynb
│   └── README.md
│
├── Part2/
│   ├── Part2.ipynb
│   └── README.md
│
├── Part3/
│   ├── Part3.ipynb
│   ├── README.md
│   └── best_model.pkl
│
└── Part4/
    ├── Part4_LLM_Pipeline.ipynb
    ├── README.md
    └── best_model.pkl
```

---

# Project Components

## Part 1 – Data Preparation and Exploratory Data Analysis

Topics covered:

- Data loading
- Missing value handling
- Duplicate removal
- Feature engineering
- Encoding categorical variables
- Exploratory Data Analysis (EDA)
- Correlation analysis
- Data visualization

---

## Part 2 – Supervised Machine Learning

Models implemented:

- Logistic Regression
- Classification Metrics
- ROC Curve
- Precision-Recall Analysis
- Threshold Analysis
- Statistical Confidence Interval

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## Part 3 – Advanced Machine Learning

Topics covered:

- Decision Tree
- Controlled Decision Tree
- Gini vs Entropy Comparison
- Random Forest
- Gradient Boosting
- Feature Importance
- Feature Ablation Study
- Cross Validation
- GridSearchCV
- Learning Curve
- Model Serialization using Joblib
- Pipeline Construction

Best model saved as:

```
best_model.pkl
```

---

## Part 4 – LLM-Powered Prediction Explanation

Track Selected:

**Track C – Model Prediction Explanation Pipeline**

Features implemented:

- OpenRouter API Integration
- Prompt Engineering
- JSON Schema Validation
- Prediction Explanation
- PII Guardrail
- Temperature Comparison
- Structured JSON Output
- End-to-End LLM Pipeline

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Requests
- JSON
- JSON Schema
- OpenRouter API
- Regular Expressions (Regex)

---

# Machine Learning Workflow

```
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Feature Engineering
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Hyperparameter Tuning
   │
   ▼
Best Model Selection
   │
   ▼
Model Serialization
   │
   ▼
LLM Prediction Explanation
```

---

# Skills Demonstrated

- Data Cleaning
- Data Visualization
- Feature Engineering
- Classification
- Ensemble Learning
- Cross Validation
- Hyperparameter Tuning
- Model Evaluation
- Pipeline Construction
- Model Serialization
- Prompt Engineering
- JSON Validation
- LLM Integration

---

# Results Summary

The project successfully:

- Built multiple supervised learning models.
- Compared ensemble learning techniques.
- Selected the best-performing model using cross-validation.
- Serialized the final pipeline for deployment.
- Integrated a Large Language Model to explain machine learning predictions.
- Validated structured JSON responses using JSON Schema.
- Implemented guardrails to prevent Personally Identifiable Information (PII) from being transmitted.

---

# Author

**Parameswari Manthiramoorthi**



---

# License

This project is developed for educational and academic purposes.