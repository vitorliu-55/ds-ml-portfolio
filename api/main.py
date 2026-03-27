from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool
import mlflow
import pandas as pd
import io
import os
import threading

model_stg_name = "customer-churn@stg"
model_prd_name = "customer-churn@prd"

model_lock = threading.Lock()

model_cache = {}

def setup_mlflow():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    print(f"Tracking: {mlflow.get_tracking_uri()}")


def load_model(alias):
    global model_stg_name, model_prd_name, model_cache
    model_name_map = {"stg": model_stg_name, "prd": model_prd_name}
    model_name = model_name_map[alias]
    if alias not in model_cache.keys():
        try:
            model_cache[alias] = mlflow.sklearn.load_model(f"models:/{model_name}")
        except Exception as e:
            print(f"WARNING: Cannot load {model_name}: {e}")
    
    return model_cache[alias]

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_mlflow()
    yield

app = FastAPI(lifespan=lifespan)

def transform_data(X):
    X = pd.concat([
        X.select_dtypes([], ['object']),
        X.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
    ], axis=1)

    return X

def make_predictions(model, df):
    id_col = df["id"]
    X = transform_data(df.drop("id", axis=1))
    print(model.predict_proba(X))
    prediction = pd.Series(model.predict_proba(X)[:, 1], name="Churn")
    ret = pd.concat([id_col, prediction], axis=1)

    return ret.to_json(orient="index")


@app.post("/prediction-stg")
async def predict_stg(file: UploadFile = File(...)):
    print("Reading data")
    contents = await file.read()
    print("Read data")
    try:
        print("Loading data")
        df = await run_in_threadpool(
            pd.read_csv,
            io.StringIO(contents.decode("utf-8"))
        )
        print("Loaded data")
    except Exception as e:
        print(e)
        raise HTTPException(415)

    print("Loading model")
    model = await run_in_threadpool(load_model, "stg")
    print("Loaded model")
    if model is None:
        raise HTTPException(status_code=503, detail="stg model not available")

    try:
        print("Making predictions")
        predictions = make_predictions(model, df)
        print("Made predictions")
    except Exception as e:
        raise HTTPException(500, e)

    return predictions


@app.post("/prediction-prd")
async def predict_prd(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        df = await run_in_threadpool(
            pd.read_csv,
            io.StringIO(contents.decode("utf-8"))
        )
        print("Data loaded")
    except Exception as e:
        print(e)
        raise HTTPException(415)

    model = await run_in_threadpool(load_model, "prd")
    print("Model loaded")
    if model is None:
        raise HTTPException(status_code=503, detail="prd model not available")

    try:
        predictions = make_predictions(model, df)
    except Exception as e:
        raise HTTPException(500, str(e))

    return predictions