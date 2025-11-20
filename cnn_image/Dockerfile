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

WORKDIR /cnn_service

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libfontconfig1 && \
    rm -rf /var/lib/apt/lists/*

# Copiar librerías base necesarias para OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 && rm -rf /var/lib/apt/lists/*

# Copiar dependencias instaladas y el código
# COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./app /cnn_service/app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "app/"]