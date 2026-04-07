import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise Exception(f"❌ Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

def predict(features: list):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return int(prediction[0])