# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
# Necesitamos la versión headless de OpenCV, que a veces requiere más librerías en sistemas slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final
FROM python:3.10-slim

WORKDIR /app

# Copiar librerías base necesarias para OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 && rm -rf /var/lib/apt/lists/*

# Copiar dependencias instaladas y el código
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY main.py model_utils.py ./

# Nota: Debes colocar tu archivo 'cnn_model.h5' (entrenado o dummy) aquí
# Si no lo tienes, el main.py creará un dummy al iniciar.
# En producción, esto se gestiona con un volumen o descarga de bucket.
# COPY cnn_model.h5 .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]