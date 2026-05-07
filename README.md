# Data Science and Machine Learning Portfolio

Personal portfolio focused on Data Science, Machine Learning, Deep Learning, and MLOps projects.

This repository documents the evolution of practical skills in:
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Machine Learning modeling
- Neural Networks with Keras
- Model evaluation and interpretation
- MLOps workflows
- API serving and containerization

The projects were developed progressively, with each project introducing new concepts, tools, and improvements over previous implementations.

---

# Repository Structure

| Project | Main Topic | Techniques | Status |
|---|---|---|---|
| `sklearn-pipelines-test` | Classification pipelines | Scikit-learn Pipelines, Grid Search, Cross Validation | Completed |
| `student-test-scores` | Regression pipeline | EDA, Feature Engineering, Ridge Regression | Completed |
| `customer-churn-nn` | Neural Networks for tabular data | Keras, MLP, Wide & Deep, TabTransformer | Completed |
| `customer-churn-competition` | MLOps and deployment | MLflow, FastAPI, Docker, XGBoost | Completed |

---

# Projects

## 1. sklearn-pipelines-test

Binary classification project using the Heart Disease dataset.

Main goals:
- Build end-to-end scikit-learn pipelines
- Compare multiple classification algorithms
- Evaluate models using ROC AUC and F1-score

Models tested:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting

Best result:
- SVM
  - ROC AUC: 0.854
  - F1-score: 0.85

Main learnings:
- Importance of proper EDA
- Need for feature engineering
- Pipeline standardization
- Model comparison methodology

---

## 2. student-test-scores

Regression project focused on predicting student test scores.

Main goals:
- Perform a more structured EDA
- Build regression pipelines
- Improve model interpretation and validation

Pipeline stages:
- Data inspection
- Data quality analysis
- Univariate and bivariate analysis
- Feature engineering
- Modeling and evaluation

Best model:
- Ridge Regression
  - RMSE: 8.894
  - MAE: 7.101
  - R²: 0.778

Main learnings:
- Better project organization
- Cleaner code structure
- Feature engineering workflow
- Model interpretation techniques

---

## 3. customer-churn-nn

Deep Learning project focused on customer churn prediction.

Main goals:
- Build Artificial Neural Networks using Keras
- Compare neural architectures for tabular classification tasks
- Benchmark neural networks against XGBoost

Architectures tested:
- Multi-Layer Perceptron (MLP)
- Wide & Deep
- TabTransformer

Main learnings:
- Limitations of neural networks on tabular datasets
- Comparison between boosting models and deep learning
- Keras workflow for structured data

---

## 4. customer-churn-competition

Customer churn prediction project with focus on MLOps and deployment.

Main goals:
- Create a production-oriented ML workflow
- Implement experiment tracking
- Serve models through APIs
- Containerize the application

Technologies used:
- XGBoost
- MLflow
- FastAPI
- Docker

Results:
- ROC AUC: 0.91422
- Kaggle placement: 2451 / 4142

Main learnings:
- Experiment tracking with MLflow
- Feature engineering using custom transformers
- Preventing data leakage
- API deployment
- Containerized ML environments

---

# Technologies

## Data Science and Machine Learning
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Keras
- TensorFlow

## Visualization
- Matplotlib
- Seaborn

## MLOps and Deployment
- MLflow
- FastAPI
- Docker

---

# Evolution Across Projects

This portfolio was intentionally structured as a learning progression.

The first projects focus on:
- basic pipelines,
- model comparison,
- and experimentation practices.

The later projects introduce:
- neural networks,
- feature engineering improvements,
- experiment tracking,
- API serving,
- and containerized deployment workflows.

Each project documents:
- objectives,
- implementation choices,
- results,
- limitations,
- and lessons learned.

---

# Author

Vitor Liu

GitHub: https://github.com/vitorliu-55
