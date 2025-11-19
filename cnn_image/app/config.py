import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_CNN_MODEL = "dima806/facial_emotions_image_detection"
