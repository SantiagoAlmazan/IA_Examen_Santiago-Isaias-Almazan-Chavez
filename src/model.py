import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

# Cargar dataset
df = pd.read_csv("../data/titanic.csv")

# Mostrar info básica
print("Dimensiones:", df.shape)
print("\nTipos de datos:\n", df.dtypes)

# Limpieza de nulos
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Codificación de variables categóricas
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

# Selección de variables
X = df.drop(columns=["Survived", "Name", "Ticket", "Cabin"])
y = df["Survived"]

# Normalización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
