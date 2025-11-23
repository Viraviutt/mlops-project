"""
    Archivo: pipeline.py
    Propósito: Construir el pipeline de preprocesamiento y entrenamiento.
    Autor: Victor Villarreal, Cristian Garcia
    Fecha: 2025-11-18
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from typing import Any


def build_pipeline() -> Pipeline:
    pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
        ])
    return pipe