from fastapi import FastAPI, File, UploadFile, HTTPException
import mlflow
import pandas as pd
import io

app = FastAPI()

mlflow.set_tracking_uri("http://mlflow:5000")


model_name_stg = "customer-churn@stg"
try:
    model_stg = mlflow.sklearn.load_model(f"models:/{model_name_stg}")
except Exception as e:
    print(f"WARNING: Cannot load {model_name_stg}: {e}")

model_name_prd = "customer-churn@prd"
try:
    model_prd = mlflow.sklearn.load_model(f"models:/{model_name_prd}")
except Exception as e:
    print(f"WARNING: Cannot load {model_name_prd}: {e}")

def transform_data(X):
    X = pd.concat([
        X.select_dtypes([], ['object']),
        X.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
    ], axis=1)

    return X

def make_predictions(model, df):
    id_col = df["id"]
    X = transform_data(df.drop("id", axis=1))
    prediction = pd.Series(model.predict(X)[:, 1], name="Churn")
    ret = pd.concat([id_col, prediction], axis=1)

    return ret.to_json(orient="index")


@app.post("/prediction-stg")
async def predict_stg(file: UploadFile = File(...)):
    global model_stg
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        print(e)
        raise HTTPException(415)

    try:
        predictions = make_predictions(model_stg, df)
    except Exception as e:
        raise HTTPException(500, e)

    return predictions


@app.post("/prediction-prd")
async def predict_prd(file: UploadFile = File(...)):
    global model_prd
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        print(e)
        raise HTTPException(415)

    try:
        predictions = make_predictions(model_prd, df)
    except Exception as e:
        raise HTTPException(500, str(e))

    return predictions