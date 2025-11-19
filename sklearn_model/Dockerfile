FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8002"]