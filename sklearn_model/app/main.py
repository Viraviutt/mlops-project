"""
    Archivo: main.py
    Propósito: Servicio para predicciones del modelo de ML clasico con FastAPI
    Autor: Victor Villarreal, Cristian Garcia
    Fecha: 2025-11-18
"""


import mlflow
import pandas as pd
import numpy as np
import logging
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config import MLFLOW_TRACKING_URI, MODEL_NAME, FEATURE_COLUMNS, EXPERIMENT_NAME
from mlflow.exceptions import MlflowException

logging.basicConfig(level=logging.INFO, format='{"time": "%(asctime)s", "level": "%(levelname)s", "service": "ml_classic", "message": %(message)s}')

app = FastAPI(title="Sklearn Model Prediction Service", version="1.0.0")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
model = None
LOADED_MODEL = None

class Features(BaseModel):
    feature_list: list[float]

def load_model():
    global model
    if model is None:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
    return model

@app.get("/status")
def status():
    """Endpoint para verificar el estado del servicio."""
    return {"status": "Sklearn Model Prediction Service is running"}

@app.post("/predict")
def make_prediction(features: Features) -> dict:
    """
    Realiza una predicción usando el modelo ML cargado.
    """

    LOADED_MODEL = load_model()

    with mlflow.start_run(run_name="request_to_ML_Model_From_Gradio"):

        print("features: ", features)
        print("features.feature_list: ", features.feature_list)
        print("LOADED MODEL: ", LOADED_MODEL)

        if LOADED_MODEL is None:
            raise HTTPException(status_code=503, detail="Modelo ML no inicializado.")
        
        mlflow.log_param("model", LOADED_MODEL.__class__.__name__)
        mlflow.log_param("features", features.feature_list)
        
        # Convertir las características Pydantic en formato DataFrame (requerido por el pipeline)
        input_data = pd.DataFrame([features.feature_list], columns=FEATURE_COLUMNS) 

        logging.info(json.dumps({"event": "prediction_request", "features": input_data.iloc[0].tolist()}))

        mlflow.log_param("features", input_data.iloc[0].tolist())
        try:
            preds = LOADED_MODEL.predict(input_data)

            mlflow.log_metric("prediction", preds)

            logging.info(json.dumps({"event": "prediction_success", "result": preds.tolist()}))
        except Exception as e:
            logging.error(json.dumps({"event": "prediction_error", "details": str(e)}))
            raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")
    
    return {"predictions": preds.tolist()}