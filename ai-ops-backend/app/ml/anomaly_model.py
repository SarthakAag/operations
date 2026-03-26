import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Train model (dummy data for hackathon)
def train_model():
    data = pd.DataFrame({
        "cpu": [10, 20, 30, 40, 50, 60, 70],
        "memory": [20, 30, 40, 50, 60, 70, 80]
    })

    model = IsolationForest(contamination=0.1)
    model.fit(data)

    joblib.dump(model, "app/ml/anomaly_model.pkl")

def load_model():
    return joblib.load("app/ml/anomaly_model.pkl")

def predict_anomaly(cpu, memory):
    model = load_model()
    prediction = model.predict([[cpu, memory]])

    return "Anomaly" if prediction[0] == -1 else "Normal"