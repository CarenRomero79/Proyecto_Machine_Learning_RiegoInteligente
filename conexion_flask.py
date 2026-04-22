from flask import Flask, render_template, jsonify
from pymongo import MongoClient
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# =========================
# 🔌 CONEXIÓN MONGODB
# =========================
client = MongoClient("mongodb://localhost:27017/")
db = client["base_datos"]
coleccion = db["mi_coleccion"]

# =========================
# 🌍 ZONA HORARIA
# =========================
timezone = pytz.timezone('America/Bogota')

# =========================
# 🧠 ENTRENAMIENTO MODELO
# =========================
def entrenar_modelo():
    datos = list(coleccion.find())

    if len(datos) < 10:
        return None

    df = pd.DataFrame(datos)
    df = df.dropna()

    # 🔥 etiqueta de riego (regla base)
    df["regar"] = df["humedad"].apply(lambda x: 1 if x < 40 else 0)

    X = df[["temperatura", "humedad"]]
    y = df["regar"]

    modelo = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo.fit(X, y)

    return modelo

modelo = entrenar_modelo()

# =========================
# 🌐 RUTA PRINCIPAL
# =========================
@app.route('/')
def index():
    return render_template('index.html')

# =========================
# 📊 DATOS PARA GRÁFICA
# =========================
@app.route('/datos')
def datos():
    datos_sensor = []

    for doc in coleccion.find().sort("timestamp", -1).limit(20):
        datos_sensor.append({
            "temperatura": doc.get("temperatura"),
            "humedad": doc.get("humedad"),
            "fecha": doc.get("timestamp").astimezone(timezone).strftime('%Y-%m-%d %H:%M:%S')
        })

    return jsonify(datos_sensor)

# =========================
# 🧠 PREDICCIÓN DE RIEGO (ML)
# =========================
@app.route('/predict')
def predict():
    global modelo

    if modelo is None:
        return jsonify({"error": "No hay suficientes datos para entrenar el modelo"})

    ultimo = coleccion.find().sort("timestamp", -1).limit(1)[0]

    t = ultimo["temperatura"]
    h = ultimo["humedad"]

    pred = modelo.predict(np.array([[t, h]]))[0]

    return jsonify({
        "temperatura": t,
        "humedad": h,
        "regar": int(pred)
    })

# =========================
# 🚀 EJECUCIÓN
# =========================
if __name__ == '__main__':
    app.run(debug=True)
