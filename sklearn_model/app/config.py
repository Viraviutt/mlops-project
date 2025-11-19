from dotenv import load_dotenv
import os

load_dotenv(override=True)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "whitewine_random_forest_model")