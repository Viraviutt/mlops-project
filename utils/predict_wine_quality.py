import pandas as pd
import mlflow
import os
from dotenv import load_dotenv
from sklearn_model.app.predict import load_model
from utils.predict_wine_config import FEATURE_COLUMNS

load_dotenv(override=True)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def predict_wine_quality(*args):
    """Función de predicción para un único punto de datos (Single Input) con MLflow Tracking."""
    
    # Inicia un nuevo RUN de MLflow para esta predicción
    # Usa un tag para identificar que esta predicción viene de la App Gradio
    with mlflow.start_run(run_name="Single_Prediction", tags={"source": "Gradio_App", "type": "single"}):
        
        input_values = list(args)

        float_values = [float(v) for v in input_values]

        input_df = pd.DataFrame([float_values], columns=FEATURE_COLUMNS)

        prediction = load_model().predict(input_df)

        quality_score = prediction[0]

        # 1. Registrar Parámetros (Inputs)
        input_params = {k: v for k, v in zip(FEATURE_COLUMNS, float_values)}
        mlflow.log_params(input_params)
        
        # 2. Registrar Métrica (Output)
        mlflow.log_metric("predicted_quality", quality_score)
        
        # 4. Formatear y Determinar Insight
        formatted_prediction = f"Predicción de Calidad: {quality_score:.2f} / 10"

        if quality_score >= 7.0:
            insight = "¡Excelente! Un vino de alta calidad."
        elif quality_score >= 5.0:
            insight = "Calidad promedio. El modelo sugiere un vino bebible."
        else:
            insight = "Baja calidad. Se recomienda precaución."
        
        return formatted_prediction, insight