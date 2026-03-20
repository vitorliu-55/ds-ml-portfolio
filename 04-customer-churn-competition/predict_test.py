import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")

with open('./data/test.csv', 'r') as f:
    df_test = pd.read_csv(f)

id_col = df_test["id"]
X = df_test.drop("id", axis=1)
X = pd.concat([
        X.select_dtypes([], ['object']),
        X.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
    ], axis=1)


model = mlflow.sklearn.load_model(f"models:/customer-churn@prd")

predictions = pd.Series(model.predict_proba(X)[:, 1], name="Churn")
ret = pd.concat([id_col, predictions], axis=1)

ret.to_csv("./data/prediction.csv", index=False)