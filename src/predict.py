# src/predict.py

import joblib
import numpy as np

# Load model once (important for performance)
model = joblib.load("models/model.pkl")

def predict(features: list):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return int(prediction[0])