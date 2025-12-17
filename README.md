# Examen de Inteligencia Artificial

## Objetivo del proyecto
Desarrollar un modelo de Machine Learning capaz de predecir si un pasajero del Titanic sobrevivió o no, aplicando técnicas de preprocesamiento de datos y un modelo de clasificación supervisada.

## Dataset
Se utilizó el dataset **Titanic** obtenido desde Kaggle, el cual contiene información demográfica y de viaje de 891 pasajeros.

Dimensiones del dataset:
- 891 registros
- 12 columnas

Tipos de datos:
- Variables numéricas: PassengerId, Survived, Pclass, Age, SibSp, Parch, Fare
- Variables categóricas: Name, Sex, Ticket, Cabin, Embarked

Variable objetivo:
- **Survived** (0 = No sobrevivió, 1 = Sobrevivió)

## Preprocesamiento de datos
Se aplicaron las siguientes técnicas:
- Limpieza de valores nulos:
  - La columna *Age* se completó con la mediana
  - La columna *Embarked* se completó con la moda
- Codificación de variables categóricas mediante One-Hot Encoding
- Normalización de variables numéricas usando *StandardScaler*
- Separación del dataset en 80% para entrenamiento y 20% para prueba

## Modelo de Machine Learning utilizado
**Regresión Logística**

### Justificación del modelo
Se eligió la Regresión Logística porque es adecuada para problemas de clasificación binaria, es fácil de interpretar y ofrece un buen desempeño con datasets de tamaño mediano.

## Evaluación del modelo
El modelo fue evaluado utilizando las siguientes métricas:

- **Accuracy:** 0.8044  
- **Precision:** 0.7746  
- **Recall:** 0.7432  

### Matriz de confusión
[[89 16]
[19 55]]


## Resultados
El modelo logra predecir correctamente aproximadamente 8 de cada 10 casos. La precisión y el recall muestran un buen equilibrio entre la identificación de pasajeros que sobrevivieron y los que no.

## Problemas encontrados
- Presencia de valores nulos en el dataset
- Necesidad de codificar variables categóricas
- Configuración inicial del entorno y librerías

## Posibles mejoras
- Probar otros modelos como Árboles de Decisión o K-Nearest Neighbors
- Ajustar hiperparámetros
- Utilizar validación cruzada
- Incluir más variables o un dataset más grande

## Instrucciones de ejecución
1. Instalar dependencias:
```bash
pip install -r requirements.txt


## ejecutar el programa
cd src
python model.py


## Autor: Santiago Isaias Almazan Chavez