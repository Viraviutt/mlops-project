"""
    Archivo: config.py
    Propósito: Configuración del servicio CNN
    Autor: Victor Villarreal, Cristian Garcia
    Fecha: 2025-11-18
"""

import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_CNN_MODEL = "dima806/facial_emotions_image_detection"

LIMITATIONS = """
Limitaciones del modelo:
- Precisión variable dependiendo de la calidad y el contenido de la imagen.

- Puede no reconocer correctamente emociones en rostros con ángulos inusuales o iluminación deficiente.

- No es adecuado para análisis clínicos o diagnósticos profesionales.

- Resultados pueden variar según el contexto cultural y las expresiones faciales individuales.

- No garantiza el reconocimiento de emociones en imagenes con fondo oscuro o iluminación difusa.

- No debe usarse para decisiones críticas sin supervisión humana.

- No debe usarse para el diagnóstico de enfermedades mentales o psiquiatricas.

- No debe usarse en contextos legales o forenses sin validación adicional.

- No debe usarse para evaluar emociones en niños sin supervisión adecuada.

- No se garantiza buena precisión en imágenes con múltiples rostros o seres que no sean humanos.

- No debe usarse para evaluar emociones en animales sin supervisión adecuada.

- No debe usarse para evaluar emociones en personas con discapacidad.

- No debe usarse para evaluar emociones en personas con trastornos mentales.

- No debe usarse para evaluar emociones en personas con trastornos de comportamiento.

- No debe usarse para evaluar emociones en personas con trastornos de aprendizaje.

- No debe usarse para evaluar emociones en personas con trastornos de ansiedad.
"""

KERNEL_EDGES = np.array(
    [
        [-1,-1,-1],
        [-1,8,-1],
        [-1,-1,-1]
    ]
)

KERNEL_SHARPEN = np.array(
    [
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ]
)