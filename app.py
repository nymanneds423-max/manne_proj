# app.py

from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

app = FastAPI()

# Input schema
class InputData(BaseModel):
    features: list


@app.get("/")
def home():
    return {"message": "ML API is running 🚀"}


@app.post("/predict")
def get_prediction(data: InputData):
    result = predict(data.features)
    return {"prediction": result}