Este servicio aloja un modelo predictivo entrenado con scikit-learn (Sklearn). Está diseñado para ser un endpoint de predicción ligero.

# 1. Funcionamiento Principal
El servicio se encarga de:
- Carga del Modelo: Al iniciarse, intenta cargar la última versión del modelo Sklearn registrado en MLflow.
- API REST: Expone un endpoint para recibir datos y devolver predicciones.

# 2. 🌐 Endpoints de la API
El servicio es accesible internamente en la red de Swarm a través del nombre de servicio: mlops_stack_sklearn_model.

| Metodo | Endpoint     | Description                |
| :-------- | :------- | :------------------------- |
| `GET` | `/health` | Verifica si el servicio está en ejecución y el modelo cargado. |
| `POST` | `/predict` | Realiza una predicción en base a los datos de entrada. |

# Ejemplo de Petición (POST /predict)
- URL Interna: http://mlops_stack_sklearn_model:8000/predict

- Body (JSON):
```bash
    {
    "features": [
        [5.1, 3.5, 1.4, 0.2] 
    ]
    }
```

- Respuesta (JSON):
```bash
    {
    "prediction": [0]
    }
```

# 3. 🎯 Scripts de Entrenamiento
- train.py: Contiene la lógica para entrenar el modelo y registrarlo en MLflow.
- Uso (Swarm): 

```bash
    docker service run --rm --network mlops_stack_ml_network infra-sklearn_model:latest python train.py
```
