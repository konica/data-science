from fastapi import FastAPI
from pydantic import BaseModel
import mlflow

app = FastAPI()

@app.get("/")
def predict_temperature():
    model = mlflow.sklearn.load_model("./linear_regression_model_v1")
    prediction = model.predict([[15.0, 1012, 3, 10.3]])  # Example input
    return {"predicted": prediction[0] }