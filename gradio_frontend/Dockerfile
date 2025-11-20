# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final
FROM python:3.10-slim

WORKDIR /gradio_app

# Copiar dependencias y código
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY requirements.txt .
COPY app.py config.py ./

# Exponer el puerto de Gradio
EXPOSE 7860

# Comando de ejecución
CMD ["python", "app.py"]