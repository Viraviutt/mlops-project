"""
    Archivo: utils.py
    Propósito: Funciones auxiliares para el modelo LLM
    Autor: Victor Villarreal, Cristian Garcia
    Fecha: 2025-11-18
"""

import time
import fitz
import docx2txt
import os
from config import MODEL_NAME, MODEL

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
    """Extrae el texto de un archivo de texto"""
    try:
        if file is None:
            return ""
        with open(file, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error al leer el archivo: {e}"
    
def read_file_pdf(file) -> str:
    """Extrae el texto de un archivo pdf"""
    try:
        with fitz.open(file) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        return f"Error al leer el archivo PDF: {e}"
    
def read_file_docx(file) -> str:
    """Extrae el texto de un archivo docx"""
    try:
        texto = docx2txt.process(file)
        return texto
    except Exception as e:
        return f"Error al leer DOCX {file}: {e}"