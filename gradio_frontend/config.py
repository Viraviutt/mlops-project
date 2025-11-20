"""
    Archivo: config.py
    Propósito: Configuración del servicio frontend de gradio
    Autor: Victor Villarreal, Cristian Garcia
    Fecha: 2025-11-18
"""

FEATURE_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", 
    "residual sugar", "chlorides", "free sulfur dioxide", 
    "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
]

INITIAL_VALUES = {
    "fixed acidity": 7.0, "volatile acidity": 0.27, "citric acid": 0.36, 
    "residual sugar": 20.7, "chlorides": 0.045, "free sulfur dioxide": 45.0, 
    "total sulfur dioxide": 170.0, "density": 1.001, "pH": 3.0, 
    "sulphates": 0.45, "alcohol": 8.8
}

LIMITATIONS = """
Limitaciones del modelo:
- Precisión variable dependiendo de la calidad y el contenido de la imagen.

- Puede no reconocer correctamente emociones en rostros con ángulos inusuales o iluminación deficiente.

- No es adecuado para análisis clínicos o diagnósticos profesionales.

- Resultados pueden variar según el contexto cultural y las expresiones faciales individuales.

- No garantiza el reconocimiento de emociones en imagenes con fondo oscuro o iluminación difusa.

- No debe usarse para decisiones críticas sin supervisión humana.

- No debe usarse para el diagnóstico de enfermedades mentales o psiquiatricas.

- No debe usarse en contextos legales o forenses sin validación adicional.

- No debe usarse para evaluar emociones en niños sin supervisión adecuada.

- No se garantiza buena precisión en imágenes con múltiples rostros o seres que no sean humanos.

- No debe usarse para evaluar emociones en animales sin supervisión adecuada.

- No debe usarse para evaluar emociones en personas con discapacidad.

- No debe usarse para evaluar emociones en personas con trastornos mentales.

- No debe usarse para evaluar emociones en personas con trastornos de comportamiento.

- No debe usarse para evaluar emociones en personas con trastornos de aprendizaje.

- No debe usarse para evaluar emociones en personas con trastornos de ansiedad.
"""