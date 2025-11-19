import gradio as gr
import os
from dotenv import load_dotenv
from llm_connector.app.utils import chat
from utils.classify import classify_image
from utils.predict_wine_config import FEATURE_COLUMNS, INITIAL_VALUES
from utils.predict_wine_quality import predict_wine_quality
from utils.predict_wine_quality_from_csv import predict_from_csv

load_dotenv()




with gr.Blocks(
    title="App",
    theme=gr.themes.Ocean(),
) as app:
    gr.Markdown(
        f"""
        # Prueba diferentes modelos de inteligencia artificial para diferentes tareas.
        """
    )

    with gr.Tab("Chatea con el Chatbot IA sobre lo que quieras."):
        chatbot = gr.Chatbot(type="messages", height=500)

        chat_interface = gr.ChatInterface(
            chat, 
            chatbot=chatbot,
            type="messages", 
            title="Chat con Modelos de IA", 
            multimodal=True,
            save_history=True,
            autoscroll=True,
            stop_btn=True
        )

    with gr.Tab("IA que procesa archivos (predice calidad del vino)"):
        # Componentes de entrada dinámicos (11 números)
        input_components_single = []
        for feature in FEATURE_COLUMNS:
            default_value = INITIAL_VALUES.get(feature, 0.5) 
            input_components_single.append(
                gr.Number(
                    label=f"{feature} (g/dm³ o valor correspondiente)", 
                    value=default_value
                )
            )

        # Botón y outputs
        predict_btn_single = gr.Button("Predecir Calidad", variant="primary")
        output_components_single = [
            gr.Textbox(label="Resultado de la Predicción", key="prediction_output"),
            gr.Textbox(label="Insight de Calidad", key="insight_output"),
        ]
        
        #Conexión de la función al botón
        predict_btn_single.click(
            fn=predict_wine_quality,
            inputs=input_components_single,
            outputs=output_components_single
        )

    with gr.Tab("Predicción por Lote (CSV)"):
        gr.Markdown(
            """
            Sube un archivo **CSV** que contenga las 11 columnas de características. 
            El modelo ejecutará la predicción en todas las filas y devolverá la calidad predicha (`Predicted_Quality`).
            """
        )
        
        # Entrada CSV
        csv_input = gr.File(
            label="Cargar Archivo CSV", 
            file_types=[".csv"]
        )

        # Salidas CSV
        summary_output = gr.Textbox(label="Promedio de Calidad del Lote", interactive=False)
        dataframe_output = gr.Dataframe(
            label="Tabla de Resultados", 
            headers=FEATURE_COLUMNS + ['Predicted_Quality'],
            interactive=False
        )

        # Botón y Conexión de la función
        predict_btn_batch = gr.Button("Predecir Lote CSV", variant="primary")
        
        predict_btn_batch.click(
            fn=predict_from_csv,
            inputs=csv_input,
            outputs=[summary_output, dataframe_output]
        )

    with gr.Tab("IA que procesa imágenes (Clasificación de Imágenes)"):
        gr.Markdown(
            """
            Sube una imagen y el modelo clasificará el contenido de la imagen.
            """
        )
        
        # Entrada de imagen
        image_input = gr.Image(label="Cargar Imagen")

        # Salidas de imagen
        image_label_output = gr.Textbox(label="Etiqueta Predicha", key="image_label_output")
        image_confidence_output = gr.Textbox(label="Confianza de la Predicción", key="image_confidence_output")

        # Botón y Conexión de la función
        classify_btn = gr.Button("Clasificar Imagen", variant="primary")
        
        classify_btn.click(
            fn=classify_image,
            inputs=image_input,
            outputs=[image_label_output, image_confidence_output]
        )

app.launch()