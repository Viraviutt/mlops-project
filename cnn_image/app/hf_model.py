"""
    Archivo: hf_model.py
    Propósito: Clase para consumir un modelo de clasificación de imágenes en HuggingFace.
    Autor: Victor Villarreal, Cristian Garcia
    Fecha: 2025-11-18
"""

from typing import Dict, Any, List
from huggingface_hub import InferenceClient
from config import HF_TOKEN, HF_CNN_MODEL

class HuggingFaceCNNClassifier:
    """
    Cliente para consumir modelos de clasificación de imágenes en HuggingFace.
    """

    def __init__(self):
        if HF_TOKEN is None:
            raise ValueError("No se encontró la variable de entorno HF_TOKEN")

        self.client = InferenceClient(
            provider="auto",
            api_key=HF_TOKEN
        )

        self.model_name = HF_CNN_MODEL

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Clasifica una imagen usando el modelo remoto de HuggingFace.

        Args:
            image_bytes (bytes): La imagen en bytes.

        Returns:
            dict: Predicción estructurada y limitaciones del sistema.
        """

        try:
            output: List[dict] = self.client.image_classification(
                image_bytes,
                model=self.model_name,
            )

            # Hugging Face devuelve algo como:
            # [{'label': 'happy', 'score': 0.98}, ...]

            if not output:
                return {"error": "Modelo no retornó resultados."}

            best = max(output, key=lambda x: x.get("score", 0))

            return {
                "prediction": best["label"],
                "confidence": float(best["score"]),
                "raw_output": output,
                "limitations": [
                    "Este modelo solo reconoce emociones faciales.",
                    "No funciona bien con imágenes borrosas o muy oscuras.",
                    "No detecta múltiples rostros simultáneamente.",
                ],
                "model": self.model_name,
            }

        except Exception as e:
            return {"error": str(e)}


# Instancia global
hf_cnn_model = HuggingFaceCNNClassifier()
