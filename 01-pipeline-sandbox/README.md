# sklearn-pipelines-test

Classification project using the Heart Disease dataset.

Dataset:
https://www.kaggle.com/datasets/neurocipher/heartdisease/data

---

# Objective

Build simple end-to-end scikit-learn pipelines to compare different classification algorithms using ROC AUC and F1-score as evaluation metrics.

---

# Pipeline

Four independent pipelines were created, one for each model:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting

Workflow:
1. Exploratory Data Analysis (EDA)
2. Data preprocessing
3. Model training
4. Hyperparameter tuning with Grid Search
5. Cross-validation
6. Model selection
7. Final evaluation on test data

---

# Results

Best model:
- Support Vector Machine (SVM)

Metrics:
- ROC AUC: 0.854
- F1-score: 0.85

---

# Main Learnings

This was the first project in the portfolio and introduced the fundamentals of:
- scikit-learn pipelines,
- model comparison,
- hyperparameter tuning,
- and validation workflows.

Key lessons learned:
- EDA was too limited and should be expanded in future projects.
- Feature engineering was missing.
- Data wrangling was not properly addressed.
- Model interpretation was insufficient.
- Some code blocks were redundant and could be modularized.

---

# Timeline

- Start date: 27/12
- End date: 04/01
