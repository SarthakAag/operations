from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    texts = [
        "database connection failed",
        "cpu usage high",
        "memory leak issue",
        "login failed",
        "server down"
    ]

    labels = [
        "DB Issue",
        "Infrastructure",
        "Infrastructure",
        "Authentication",
        "Infrastructure"
    ]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    joblib.dump((model, vectorizer), "app/ml/ticket_model.pkl")

def load_model():
    return joblib.load("app/ml/ticket_model.pkl")

def predict_ticket(text):
    model, vectorizer = load_model()
    X = vectorizer.transform([text])
    return model.predict(X)[0]