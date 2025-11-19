# Pipeline Inteligente

Repositorio con 4 microservicios: llm_connector, sklearn_model, cnn_image y gradio_frontend.

Requisitos: Docker, Python 3.10 (solo si ejecutas tests localmente).

Ejecutar en desarrollo:

docker-compose -f infra/docker-compose.yml up --build

Visitar:
- MLflow UI: http://localhost:5000
- Frontend (Gradio): http://localhost:7860