"""
    Archivo: config.py
    Propósito: Configuración del modelo clasico de ML (Random Forest)
    Autor: Victor Villarreal, Cristian Garcia
    Fecha: 2025-11-18
"""

from dotenv import load_dotenv
import os

load_dotenv(override=True)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "whitewine_random_forest_model")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "whitewine_random_forest_experiment")

FEATURE_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", 
    "residual sugar", "chlorides", "free sulfur dioxide", 
    "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
]
INITIAL_VALUES = {
    "fixed acidity": 7.0, "volatile acidity": 0.27, "citric acid": 0.36, 
    "residual sugar": 20.7, "chlorides": 0.045, "free sulfur dioxide": 45.0, 
    "total sulfur dioxide": 170.0, "density": 1.001, "pH": 3.0, 
    "sulphates": 0.45, "alcohol": 8.8
}