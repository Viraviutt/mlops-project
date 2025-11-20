"""
    Archivo: main.py
    Propósito: Servicio para conectividad con modelo LLM
    Autor: Victor Villarreal, Cristian Garcia
    Fecha: 2025-11-18
"""
import base64
import io
import os
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging, json
from PIL import Image
from dotenv import load_dotenv

from config import MODEL_NAME, MODEL
from utils import validate_input, validate_history, read_file_pdf, read_file_txt, read_file_docx, sent_files

load_dotenv()

# Configuración de Logging en formato JSON
logging.basicConfig(level=logging.INFO, format='{"time": "%(asctime)s", "level": "%(levelname)s", "service": "llm_connector", "message": %(message)s}')
app = FastAPI(title="LLM Connector Service", version="1.0.0")

CNN_API_URL = os.getenv("CNN_API_URL", "http://localhost:8003")

logger = logging.getLogger("llm_connector")
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class FileData(BaseModel):
    filename: str
    data_b64: str
    type: str

class Question(BaseModel):
    """Esquema para la solicitud al LLM."""
    text: str
    history: list[dict]
    file: FileData | None = None

class Response(BaseModel):
    """Esquema para la respuesta del LLM."""
    answer: str

@app.get("/status")
def status():
    """Endpoint para verificar el estado del servicio."""
    return {"status": "LLM Connector Service is running"}

def call_cnn_api(file_data: dict) -> str:
    """
    Llama a la API de la CNN enviando el array NumPy reconstruido.
    """
    try:
        decoded_bytes = base64.b64decode(file_data["data_b64"])

        bytes_io = io.BytesIO(decoded_bytes)

        img_array = np.load(bytes_io)
        logger.info(json.dumps({"level":"INFO","service":"llm","msg":f"Array reconstruido en LLM con forma {img_array.shape}"}))
    except Exception as e:
        logger.error(json.dumps({"level":"ERROR","service":"llm","error":f"Error al reconstruir el array NumPy: {e}"}))
        raise HTTPException(status_code=500, detail=f"Error al reconstruir el array NumPy: {e}")
    
    cnn_image_bytes = img_array.tobytes()

    try:
        response = requests.post(
            f"{CNN_API_URL}/classify_array", # Endpoint de la CNN que recibe bytes
            data=cnn_image_bytes,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Shape": ",".join(map(str, img_array.shape)),
                "X-Dtype": str(img_array.dtype)
            },
            timeout=20
        )
        response.raise_for_status()
        data = response.json()
        
        return (
            f"El modelo CNN clasificó la imagen como {data['predicted_class']} "
            f"con una confianza del {data['confidence']:.2f}."
        )
    except Exception as e:
        return f"Error al comunicarse con la API de CNN: {str(e)}"

sent_files = []

@app.post("/ask", response_model=Response)
def chat(question: Question) -> dict:
    """
    Endpoint para manejar preguntas al LLM con historial y archivos adjuntos.
    """
    logger.info(json.dumps({"level":"INFO","service":"llm","msg":"received_query"}))
    if question.file:
        call_cnn_api()
    is_valid, error_msg = validate_input(question.text)
    if not is_valid:
        logger.error(json.dumps({"level":"ERROR","service":"llm","error":error_msg}))
        return error_msg
    try:
        print(f"Modelo seleccionado: {MODEL}")
        
        if not MODEL:
            error = f"El modelo '{MODEL}' no está disponible. Verifica la configuración de API keys"
            return f"❌ **Error**: {error}."

        print(f"modelo: {MODEL_NAME}")

        system_prompt = "Eres un asistente de IA que responde preguntas y ayuda con tareas."

        messages = [
            {"role": "user", "content": f"{system_prompt}"}
        ]

        for msg in question.history:
            if validate_history(msg, sent_files[0] if sent_files else ""):
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": question.text})

        response_content = None

        resp = MODEL.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        response_content = resp.choices[0].message.content
        logging.info(json.dumps({"event": "response_sent", "answer_length": len(response_content)}))
        print("Respuesta recibida del modelo, ", response_content)

        if not response_content:
            error = f"No se pudo obtener una respuesta válida del modelo '{MODEL_NAME}'."
            return f"**Error**: {error}."
        
    except Exception as e:
        logger.error(json.dumps({"level":"ERROR","service":"llm","error":str(e)}))
        error = f"Error inesperado: {str(e)[:200]}"
        print(f"Error inesperado: {e}")
        raise HTTPException(status_code=500, detail="LLM error")
        #return f"❌ **Error Inesperado**: Ocurrió un problema técnico: {str(e)[:200]}..."
        
    return {"answer": response_content}