import mlflow
import numpy as np
from .config import MLFLOW_TRACKING_URI, MODEL_NAME


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = None


def load_model():
    global model
    if model is None:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
    return model

# : List[List[float]]

def predict(feature_list):
    m = load_model()
    X = np.array(feature_list)
    preds = m.predict(X)
    probs = m.predict_proba(X).tolist() if hasattr(m, 'predict_proba') else None
    return {"predictions": preds.tolist(), "probabilities": probs}