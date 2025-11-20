"""
    Archivo: config.py
    Propósito: Configuración del servicio LLM
    Autor: Victor Villarreal, Cristian Garcia
    Fecha: 2025-11-18
"""
import os
from openai import OpenAI
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise HTTPException(status_code=500, detail="GEMINI_API_KEY no está configurada en las variables de entorno.")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

MODEL_NAME = "gemini-2.5-flash"
MODEL = OpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")