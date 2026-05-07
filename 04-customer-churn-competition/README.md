# customer-churn-competition

Customer churn prediction project focused on MLOps and deployment workflows.

Competition:
https://www.kaggle.com/competitions/playground-series-s6e3

---

# Objective

Develop a production-oriented Machine Learning workflow including:
- experiment tracking,
- model serving,
- and containerization.

Main goals:
- Implement MLflow for experiment tracking
- Serve models with FastAPI
- Containerize the application using Docker

---

# Pipeline

## Modeling
- Model testing
- Feature engineering
- Hyperparameter tuning

## Experiment Tracking
- MLflow registry integration

## Deployment
- API serving with FastAPI
- Docker containerization

---

# Results

Best model:
- XGBoost Classifier

Metrics:
- Test ROC AUC: 0.91422
- Kaggle placement: 2451 / 4142

---

# Improvements Compared to Previous Projects

- Stronger feature engineering workflow
- Better use of MLflow
- Custom scikit-learn transformers
- Prevention of data leakage
- Deployment-oriented environment
- API serving implementation
- Containerized workflow

---

# Main Learnings

- MLOps tools should be integrated from the beginning of the project lifecycle.
- Feature engineering remains one of the most impactful stages for tabular data.
- Competition ranking can still be improved with stronger experimentation strategies.

---

# Technologies

- Python
- Scikit-learn
- XGBoost
- MLflow
- FastAPI
- Docker

---

# Timeline

- Start date: 17/03
- Work days: ~7 days
- End date: 01/04
