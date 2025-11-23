"""
    Archivo: app.py
    Propósito: Interfaz grafica realizada en gradio
    Autor: Victor Villarreal, Cristian Garcia
    Fecha: 2025-11-18
"""


import gradio as gr
import requests
import os
import numpy as np
import base64
import io
from dotenv import load_dotenv
from config import FEATURE_COLUMNS, INITIAL_VALUES, LIMITATIONS
from PIL import Image

load_dotenv()

LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8001")
ML_API_URL = os.getenv("ML_API_URL", "http://localhost:8002")
CNN_API_URL = os.getenv("CNN_API_URL", "http://localhost:8003")

def chat_with_llm(message: dict, history) -> str:
    """Realiza una consulta al modelo LLM para recibir una respuesta."""
    try:
        serialized_file = None
        if isinstance(message["files"], str) and os.path.exists(message["files"]):
            file_path = message["files"]
                
                
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = Image.open(file_path)
                    img_array = np.array(img)
                    
                    bytes_io = io.BytesIO()
                    np.save(bytes_io, img_array)
                    
                    b64_data = base64.b64encode(bytes_io.getvalue()).decode('utf-8')
                    
                    serialized_file = {
                        "filename": os.path.basename(file_path),
                        "data_b64": b64_data,
                        "type": "image/numpy"
                    }
                
                except Exception as e:
                    print(f"ERROR: No se pudo serializar la imagen a NumPy/Base64: {e}")
    

        print("Iniciando función de chat... mensaje: ", message, "history: ", history)

        response = requests.post(
            f"{LLM_API_URL}/ask",
            json={
                  "text": message["text"],
                  "history": history,
                  "file": serialized_file,
                 },
            timeout=10
        )
        response.raise_for_status()
        return response.json().get("answer", "Error: No se pudo obtener respuesta del LLM.")
    except requests.exceptions.RequestException as e:
        return f"Error de conexión con el LLM: {e}"

def validate_ml_data(*args):
    """Llama al modelo de ML Clásico."""
    try:
        payload = {
            "feature_list": list(args)
        }
        
        
        response = requests.post(
            f"{ML_API_URL}/predict",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        
        result = (
            f"**Predicción del Modelo ML Clásico (Wine Quality)**\n"
            f"Calidad predicha: **{data['predictions'][0]}**\n"
        )
        return result
    except requests.exceptions.RequestException as e:
        return f"Error de conexión o datos con el Modelo ML: {e}"
    except Exception as e:
        return f"Error inesperado: {e}"

def classify_image_cnn(image_path):
    """Llama al modelo CNN."""
    if image_path is None:
        return "Por favor, sube una imagen."
    

        
    try:
        # Gradio pasa el path temporal del archivo
        with open(image_path, "rb") as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            
            response = requests.post(
                f"{CNN_API_URL}/classify",
                files=files,
                timeout=20 # Dar más tiempo para el procesamiento de la CNN
            )
            response.raise_for_status()
            
            data = response.json()
            
            result = (
                f"**Clasificación CNN**\n"
                f"Clase Predicha: **{data['prediction']}**\n"
                f"Confianza: {data['confidence']:.2f}\n"
            )
            return result
    except requests.exceptions.RequestException as e:
        return f"Error de conexión con la CNN: {e}"
    except Exception as e:
        return f"Error inesperado: {e}"


llm_interface = gr.ChatInterface(
    fn=chat_with_llm,
    type="messages",
    multimodal=True,
    stop_btn=True,
    theme=gr.themes.Ocean(),
    submit_btn=True,
    save_history=True,
    autoscroll=True,
    title="LLM Connector",
    description="Chatea con el LLM para obtener respuestas y contexto."
)

input_components_single = []
for feature in FEATURE_COLUMNS:
    default_value = INITIAL_VALUES.get(feature, 0.5) 
    input_components_single.append(
        gr.Number(
            label=f"{feature} (g/dm³ o valor correspondiente)", 
            value=default_value
        )
    )

ml_interface = gr.Interface(
    fn=validate_ml_data,
    inputs=input_components_single,
    theme=gr.themes.Ocean(),
    outputs="markdown",
    title="Validación ML Clásico (Clasificador Vino)",
    description="Ingresa las características del vino para obtener su calidad."
)

cnn_interface = gr.Interface(
    fn=classify_image_cnn,
    inputs=gr.Image(type="filepath", label="Sube una imagen (Rostro de una persona)"),
    theme=gr.themes.Ocean(),
    outputs="markdown",
    title="Clasificación Visual CNN",
    description=f"Sube una imagen para clasificación. \nNota: {LIMITATIONS}"
)

app = gr.TabbedInterface(
    [llm_interface, ml_interface, cnn_interface],
    ["LLM Chat", "ML Clásico", "Clasificación CNN"],
    title="Pipeline MLOps (LLM + ML + CNN)",
    theme=gr.themes.Ocean(),
)

if __name__ == "__main__":
    # La API de Gradio se lanza en el puerto 7860
    app.launch(server_name="0.0.0.0", server_port=7860)