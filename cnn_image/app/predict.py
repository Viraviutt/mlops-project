"""
    Archivo: predict.py
    Propósito: Función para clasificar una imagen utilizando un modelo preentrenado de Hugging Face.
    Autor: Victor Villarreal, Cristian Garcia
    Fecha: 2025-11-18
"""

import cv2
from hf_model import hf_cnn_model
from preprocess import smooth, edges, sharpen
import tempfile
def classify_image(image) -> str | float:
    """
    Clasifica una imagen utilizando un modelo preentrenado de Hugging Face.
    """

    print("Tipo de archivo que es imagen: ", type(image))
    print("Dimensiones de la imagen: ", image.shape)
    print("Imagen: ", image)

    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp_file.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    image = smooth(image)
    image = edges(image)
    image = sharpen(image)

    output = hf_cnn_model.predict(temp_file.name)

    return output.get("prediction", "Error en la predicción"), output.get("confidence", 0.0)