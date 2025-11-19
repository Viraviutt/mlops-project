import pandas as pd
import mlflow
import io
from sklearn_model.app.predict import load_model
from utils.predict_wine_config import FEATURE_COLUMNS

def predict_from_csv(csv_file):
    """Función de predicción para datos en lote (CSV Upload) con MLflow Tracking."""
    summary_default = "Inicie la predicción cargando un archivo CSV."
    if csv_file is None:
        return "Error: No se ha cargado ningún archivo.", pd.DataFrame()

    with mlflow.start_run(run_name="Batch_Prediction") as run:
        try:
            df_input = pd.read_csv(csv_file.name, sep=";")
            print(f"CSV cargado: {df_input}.")
        except Exception as e:
            return f"Error al leer el CSV: {e}", pd.DataFrame()
        
        print(f"Columnas del CSV: {df_input.columns.tolist()}")

        # Validación de columnas
        missing_cols = [col for col in FEATURE_COLUMNS if col not in df_input.columns]

        print(f"Columnas faltantes: {missing_cols}")

        if missing_cols:
            return f"Error: El CSV debe tener 11 columnas. Faltan: {', '.join(missing_cols)}", pd.DataFrame()

        # Registrar el archivo de entrada original como artefacto
        mlflow.log_artifact(csv_file.name) 

        df_predict = df_input[FEATURE_COLUMNS]
        df_predict = df_predict.astype('float64')
        predictions = load_model().predict(df_predict)
        print(f"Predicciones generadas: {predictions}")

        df_output = df_input.copy()
        df_output['Predicted_Quality'] = predictions
        
        # Registrar métricas clave del lote
        avg_quality = df_output['Predicted_Quality'].mean()
        mlflow.log_metric("batch_size", len(df_output))
        mlflow.log_metric("avg_predicted_quality", avg_quality)

        # Opcional: Registrar el CSV de salida con las predicciones
        output_buffer = io.StringIO()
        df_output.to_csv(output_buffer, index=False)
        mlflow.log_text(output_buffer.getvalue(), "batch_predictions_output.csv")

        return avg_quality, df_output