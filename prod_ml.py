import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

def train_model():
    data = load_iris()
    X, y = data.data, data.target

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    joblib.dump(model, "models/model.pkl")
    print("Model trained and saved!")

if __name__ == "__main__":
    train_model()