# sklearn-pipelines-test
Quick data visualization and model training using Heart Disease Dataset https://www.kaggle.com/datasets/neurocipher/heartdisease/data .

### Objective
Create simple end-to-end scikit-learn pipelines to compare different classification models, using AUC and F1 metrics as scores.

### Pipeline
4 pipelines, one for each different model: Logistic Regression, SVM, Random Forest and Gradient Boosting.

- (quick) EDA
- Preprocess for each model
- Train model
  - Grid Search
- Cross validation
  - AUC as score
- Model selection
  - AUC and F1
- Test

### Result
SVM with AUC = 0.854 and F1 = 0.85

### Considerations
Things to improve in next projects:
- EDA: was quick and poor
- Data Wrangling: forgot to do Data Wrangling
- Model selection: did not interpretate model. 
- General: used redundant block of codes

#### Start date: 27/12
#### End date: 04/01