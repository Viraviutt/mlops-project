"""
    Archivo: main.py
    Propósito: Servicio para clasificación de imágenes con FastAPI
    Autor: Victor Villarreal, Cristian Garcia
    Fecha: 2025-11-18
"""

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import logging
import json
from config import LIMITATIONS
from predict import classify_image

logging.basicConfig(level=logging.INFO, format='{"time": "%(asctime)s", "level": "%(levelname)s", "service": "cnn_model", "message": %(message)s}')

app = FastAPI(title="CNN Image Classification Service")

class ClassificationResponse(BaseModel):
    prediction: str
    confidence: float

@app.get("/")
def index():
    """Endpoint de inicio."""
    return {"message": "Bienvenido al servicio de clasificación de imágenes con FastAPI"}

@app.get("/status")
def status():
    """Endpoint para verificar el estado del servicio."""
    return {"status": "CNN Image Classification Service is running"}

@app.post("/classify-array")
async def classify(request: Request):
    """
    Endpoint para clasificar una imagen subida por el usuario.
    """

    try:
        images_bytes = await request.body()
    except Exception as e:
        return JSONResponse({"error": f"Error al leer el cuerpo binario: {str(e)}"}, status_code=400)
    
    shape_str = request.headers.get("x-shape")
    dtype_str = request.headers.get("x-dtype")

    if not shape_str or not dtype_str:
        return JSONResponse({
            "error": "Faltan metadatos (X-Shape o X-Dtype) necesarios para la deserialización del NumPy Array."
        }, status_code=400)
    
    try:
        # Se hace para convertir la forma (ej. "480,640,3") a tupla de enteros
        img_shape = tuple(map(int, shape_str.split(',')))
        
        img_array = np.frombuffer(image_bytes, dtype=dtype_str).reshape(img_shape)
        
    except Exception as e:
        return JSONResponse({
            "error": f"Error de deserialización o reconstrucción del array: {str(e)}"
        }, status_code=500)
    
    logging.info(json.dumps({"event": "classification_request", "Image": img_array}))

    try:
        prediction, confidence = classify_image(img_array)
        
        logging.info(json.dumps({"type": "info", "event": "classification_done", "msg": "Clasificación completada, prediccion: {}, confianza: {}".format(prediction, confidence)}))
    except Exception as e:
        logging.error(json.dumps({"event": "classification_error", "details": str(e)}))
        raise HTTPException(status_code=500, detail=f"Error durante la clasificación: {e}")
    
    return ClassificationResponse(
        prediction=prediction,
        confidence=float(confidence),
        limitations=LIMITATIONS
    )

    

@app.post("/classify", response_model=ClassificationResponse)
async def classify(file: UploadFile = File(...)):
    """
    Endpoint para clasificar una imagen subida por el usuario.
    """

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo subido no es una imagen válida.")

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen JPEG o PNG.")
    
    logging.info(json.dumps({"event": "classification_request", "filename": file.filename}))

    try:
        image_bytes = await file.read()
        import cv2
        import numpy as np

        # Convertir bytes a matriz de imagen
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        prediction, confidence = classify_image(img)

        return ClassificationResponse(
            prediction=prediction,
            confidence=confidence
        )

    except Exception as e:
        logging.error(json.dumps({"event": "classification_error", "details": str(e)}))
        raise HTTPException(status_code=500, detail=f"Error durante la clasificación: {e}")