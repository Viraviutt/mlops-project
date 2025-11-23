# Pipeline Inteligente

Repositorio con 4 microservicios: llm_connector, sklearn_model, cnn_image y gradio_frontend.

Requisitos: Docker, Python 3.10 (solo si ejecutas tests localmente).

# 1. Despliegue Rápido (Entorno de Desarrollo)
Para iniciar todos los servicios rápidamente en modo de desarrollo y prueba (sin Swarm), utiliza docker-compose. Requisitos:
- Tener Docker Desktop instalado y funcionando.
- Asegúrate de que el Puerto 5000 (MLflow) y el Puerto 80 (Gradio) no estén siendo utilizados por otra aplicación en tu máquina.

## Comandos de Despliegue

### 1. Construir y levantar la pila:
```bash
  #Asumiendo que has copiado stack.yml a infra/docker-compose.yml
  docker compose -f infra/docker-compose.yml up -d --build
```
Nota: Si ya habías construido las imágenes de tu aplicación (infra-cnn_image, etc.), puedes omitir el --build si los cambios son solo en la infraestructura.

### 2. Verificar el Estado:
```bash
  docker compose -f infra/docker-compose.yml ps
```
## URLs de Acceso

|  Servicio | URL     | Description                |
| :-------- | :------- | :------------------------- |
| `MLFLOW UI` | `http://localhost:5000` | Seguimiento de experimentos y registro de modelos. |
| `Gradio Frontend` | `htpp://localhost:80` | Interfaz de usuario para interactuar con los modelos. |

# 2. Despliegue en Producción (Docker Swarm)
Para el despliegue con orquestación y mayor disponibilidad, utilizamos Docker Swarm y el archivo stack.yml.

#### Requisitos
- 1. Habilitar Docker Swarm en tu nodo manager (si usas Docker Desktop, solo necesitas inicializarlo):

```bash
  docker swarm init
```

- 2. Asegúrate de que todas las imágenes personalizadas (infra-...) hayan sido construidas previamente (Swarm no construye imágenes):
```bash
  docker compose -f infra/stack.yml build
```

## Comandos de despliegue
- 1. Desplegar pila (stack)
```bash
  docker stack deploy -c infra/stack.yml mlops_stack
```

- 2. Verificar el estado y tareas
```bash
  docker stack services mlops_stack
  docker stack ps mlops_stack
```

Las URLs de acceso son las mismas que en el entorno de desarrollo: http://localhost:5000 y http://localhost:80.

# 3. Entrenamiento y registro de modelos a mlflow
Antes de usar los endpoints de los modelos, estos deben ser entrenados y registrados en el servidor de MLflow.

## A. Entrenamiento en entorno local (docker compose)
Puedes ejecutar el script de entrenamiento directamente dentro de un contenedor de servicio. Esto es ideal para el desarrollo local.

### Entrenamiento del modelo de scikit-learn
```bash
  docker compose -f infra/docker-compose.yml exec sklearn_model python train.py
```

## B. Entrenamiento en entorno de swarm
En Swarm, usamos el comando docker service run para ejecutar una tarea única (One-Off Task) que entrena y registra el modelo, aprovechando la misma red (mlops_stack_ml_network).

El formato general es: docker service run --rm --network [NETWORK_NAME] [IMAGE_NAME] python train.py

### Entrenamiento del modelo de scikit-learn
```bash
  docker service run --rm --network mlops_stack_ml_network infra-cnn_image:latest python train.py
```
Nota Importante: El nombre de la red de Swarm sigue el patrón [STACK_NAME]_[NETWORK_NAME]. En nuestro caso, es mlops_stack_ml_network.