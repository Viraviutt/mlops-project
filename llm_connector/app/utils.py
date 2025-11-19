import time
import fitz # PyMuPDF
import docx2txt
import os
from httpx import Timeout
from llm_connector.app.config import MODEL_NAME, MODEL
import base64
import mimetypes

sent_files = []

def validate_input(message) -> bool | str:
    """Valida el mensaje de entrada"""
    if not message:
        return False, "**Error**: No puedes enviar un mensaje vacío. Por favor, escribe algo."
    
    if len(message.strip()) == 0:
        return False, "**Error**: El mensaje solo contiene espacios en blanco. Por favor, escribe un mensaje válido."
    
    if len(message) > 8000:
        return False, "**Error**: El mensaje es demasiado largo (máximo 8000 caracteres). Por favor, acórtalo."
    
    return True, None

def validate_history(history, file) -> bool:
    if len(history["content"]) > 0 and isinstance(history["content"], tuple):
        if history["content"][0] == file:
            return False
    return True
    
def read_file_txt(file) -> str:
    try:
        if file is None:
            return ""
        with open(file, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error al leer el archivo: {e}"
    
def read_file_pdf(file) -> str:
    try:
        with fitz.open(file) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        return f"Error al leer el archivo PDF: {e}"
    
def read_file_docx(file) -> str:
    try:
        texto = docx2txt.process(file)
        return texto
    except Exception as e:
        return f"Error al leer DOCX {file}: {e}"

def chat(message: dict, history) -> str:
    print("Iniciando función de chat... mensaje: ", message)
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
        print("Respuesta recibida del modelo, ", response_content)

        if not response_content:
            error = f"No se pudo obtener una respuesta válida del modelo '{MODEL_NAME}'."
            return f"**Error**: {error}."
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
    except Exception as e:
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        error = f"Error inesperado: {str(e)[:200]}"
        print(f"Error inesperado: {e}")
        return f"❌ **Error Inesperado**: Ocurrió un problema técnico: {str(e)[:200]}..."
        
    return response_content