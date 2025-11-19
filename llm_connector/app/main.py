"""
FastAPI app para LLM connector
"""
from time import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm_connector.app.utils import chat
import logging, json
import os

from llm_connector.app.config import MODEL_NAME, MODEL
from llm_connector.app.utils import validate_input, validate_history, read_file_pdf, read_file_txt, read_file_docx, sent_files

# Configuración de Logging en formato JSON
logging.basicConfig(level=logging.INFO, format='{"time": "%(asctime)s", "level": "%(levelname)s", "service": "llm_connector", "message": %(message)s}')
app = FastAPI(title="LLM Connector Service", version="1.0.0")

logger = logging.getLogger("llm_connector")
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class Response(BaseModel):
    """Esquema para la respuesta del LLM."""
    answer: str

@app.get("/status")
def status():
    """Endpoint para verificar el estado del servicio."""
    return {"status": "LLM Connector Service is running"}

@app.post("/ask", response_model=Response)
def chat(message: dict, history: list) -> str:
    """
    Endpoint para manejar preguntas al LLM con historial y archivos adjuntos.
    """
    logger.info(json.dumps({"level":"INFO","service":"llm","msg":"received_query"}))
    is_valid, error_msg = validate_input(message["text"])
    if not is_valid:
        return error_msg
    try:
        total_start_time = time.time()
        print(f"Modelo seleccionado: {MODEL}")
        
        if not MODEL:
            error = f"El modelo '{MODEL}' no está disponible. Verifica la configuración de API keys"
            return f"❌ **Error**: {error}."

        print(f"modelo: {MODEL_NAME}")

        system_prompt = "Eres un asistente de IA que responde preguntas y ayuda con tareas."

        messages = [
            {"role": "user", "content": f"{system_prompt}"}
        ]

        if message["files"]:
            sent_files.append(message["files"][0])
            for file in message["files"]:
                if os.path.basename(file).endswith('.pdf'):
                    file_content = read_file_pdf(file)
                elif os.path.basename(file).endswith('.txt'):
                    file_content = read_file_txt(file)
                elif os.path.basename(file).endswith('.docx'):
                    file_content = read_file_docx(file)
                else:
                    file_content = "Tipo de archivo no soportado. Solo se permiten archivos .txt, .pdf y .docx"
                
                # Añadir el contenido del archivo al historial como un mensaje del sistema
                history.append({"role": "user", "content": f"Contenido del archivo \"{os.path.basename(file)}\": {file_content}"})

        for msg in history:
            if validate_history(msg, sent_files[0] if sent_files else ""):
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": message["text"]})

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
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
    except Exception as e:
        logger.error(json.dumps({"level":"ERROR","service":"llm","error":str(e)}))
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        error = f"Error inesperado: {str(e)[:200]}"
        print(f"Error inesperado: {e}")
        raise HTTPException(status_code=500, detail="LLM error")
        #return f"❌ **Error Inesperado**: Ocurrió un problema técnico: {str(e)[:200]}..."
        
    return response_content