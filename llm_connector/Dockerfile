# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final
FROM python:3.10-slim

WORKDIR /llm_service

# Copiar dependencias instaladas y código
# COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./app /llm_service/app

# Exponer el puerto
EXPOSE 8000

# Comando de ejecución con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "app/"]