import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from cnn_image.app.config import HF_TOKEN, HF_CNN_MODEL

load_dotenv()

MODEL_CNN_NAME = HF_CNN_MODEL

client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN,
)

output = client.image_classification("/home/viraviut/HDD/MLOPS/mlops-final-project/cnn_image/cats.jpg", model=MODEL_CNN_NAME)