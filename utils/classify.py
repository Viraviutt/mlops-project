import cv2
from cnn_image.app.hf_model import hf_cnn_model
from cnn_image.app.preprocess import smooth, edges, sharpen
import tempfile
def classify_image(image_path):
    """Clasifica una imagen utilizando un modelo preentrenado de Hugging Face."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp_file.name, cv2.cvtColor(image_path, cv2.COLOR_RGB2BGR))

    print("Clasificando imagen temporal: ", temp_file)
    print("Imagen guardada temporalmente en: ", temp_file.name)

    print("Clasificando imagen: ", image_path)
    image_path = smooth(image_path)
    image_path = edges(image_path)
    image_path = sharpen(image_path)
    print("Clasificando imagen procesada: ", image_path)
    output = hf_cnn_model.predict(temp_file.name)
    print("Resultado de la clasificación: ", output)
    return output.get("prediction", "Error en la predicción"), output.get("confidence", 0.0)