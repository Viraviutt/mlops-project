"""
    Archivo: train.py
    Propósito: Entrenamiento y registro del modelo de ML clasico con mlflow
    Autor: Victor Villarreal, Cristian Garcia
    Fecha: 2025-11-18
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from pipeline import build_pipeline
from sklearn.model_selection import train_test_split
from config import MLFLOW_TRACKING_URI, MODEL_NAME, EXPERIMENT_NAME


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def train_and_log():
    mlflow.set_experiment(EXPERIMENT_NAME)
    print("Justo antes de leer el csv")
    df = pd.read_csv("/sklearn_service/data/winequality-white.csv", sep=";")
    print("Justo despues de leer el csv")
    X = df.drop('quality', axis=1)
    y = df['quality']
    print("Justo antes de hacer el split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    pipe = build_pipeline()
    print("Justo antes de hacer el fit")
    with mlflow.start_run(run_name="whitewine_model_training"):
        mlflow.log_param("model", "RandomForest")
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        mlflow.log_metric("accuracy", float(acc))
        mlflow.sklearn.log_model(pipe, MODEL_NAME, registered_model_name=MODEL_NAME)
        print(f"Trained. accuracy={acc}")

if __name__ == "__main__":
    train_and_log()