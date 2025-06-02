import joblib

def load_model():
    return joblib.load("app/models/svd_model.pkl")