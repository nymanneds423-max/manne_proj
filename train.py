# src/train.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib
import os

def train_model():
    # Load data
    data = load_iris()
    X, y = data.data, data.target

    # Train model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Create models folder
    os.makedirs("models", exist_ok=True)

    # Save model
    joblib.dump(model, "models/model.pkl")

    print("Model trained and saved!")