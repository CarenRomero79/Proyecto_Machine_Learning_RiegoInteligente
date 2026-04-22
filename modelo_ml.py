import pandas as pd
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier

# Conexión MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["base_datos"]
coleccion = db["mi_coleccion"]

# Obtener datos
data = list(coleccion.find())

df = pd.DataFrame(data)
df = df.dropna()

# Etiqueta (regla base)
df["regar"] = df["humedad"].apply(lambda x: 1 if x < 40 else 0)

# Variables
X = df[["temperatura", "humedad"]]
y = df["regar"]

# Modelo
modelo = RandomForestClassifier(n_estimators=50)
modelo.fit(X, y)

print("✅ Modelo entrenado correctamente")
