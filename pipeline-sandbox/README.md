# sklearn-pipelines-test
Quick data visualization and model training using Heart Disease Dataset https://www.kaggle.com/datasets/neurocipher/heartdisease/data .

### Objective
Create simple end-to-end scikit-learn pipelines to compare different classification models, using AUC and F1 metrics as scores.

### Pipeline
4 pipelines, one for each different model: Logistic Regression, SVM, Random Forest and Gradient Boosting.

- (quick) EDA
- (quick) Preprocess for each model
- Train model
  - Grid Search
- Cross validation
  - AUC as score
- Model selection
  - AUC and F1
- Test

### Result
SVM with AUC = 0.854 and F1 = 0.85
