"""
generar_datos.py
Genera datos sintéticos realistas usando NumPy + SDV
y los guarda en MongoDB para simular los sensores físicos.
"""

import numpy as np
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from pymongo import MongoClient
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings("ignore")

# =========================
# 🔌 CONEXIÓN MONGODB
# =========================
client = MongoClient("mongodb://localhost:27017/")
db     = client["base_datos"]
coleccion = db["mi_coleccion"]

# =========================
# 🌍 ZONA HORARIA
# =========================
timezone = pytz.timezone('America/Bogota')

# =========================
# 📐 PASO 1: NUMPY genera 50 datos base realistas
# =========================
def generar_datos_base(n=50):
    """
    Genera datos base con NumPy respetando patrones
    físicos reales: más calor al mediodía, más seco
    cuando hay más temperatura, etc.
    """
    np.random.seed(42)
    horas = np.random.randint(0, 24, n)

    # Temperatura: más alta en horas pico (10-16)
    temperatura = np.where(
        (horas >= 10) & (horas <= 16),
        np.random.normal(30, 4, n),   # Día caluroso
        np.random.normal(20, 4, n)    # Mañana/noche fresco
    ).clip(10, 45)

    # Humedad suelo: baja cuando hace calor
    humedad = np.where(
        (horas >= 10) & (horas <= 16),
        np.random.normal(30, 8, n),   # Más seco en el día
        np.random.normal(58, 8, n)    # Más húmedo en la noche
    ).clip(5, 100)

    # Humedad aire: inversamente proporcional a temperatura
    humedad_aire = (100 - temperatura * 0.8 + np.random.normal(0, 8, n)).clip(20, 95)

    # Luz: alta en el día, cero en la noche
    luz = np.where(
        (horas >= 6) & (horas <= 18),
        np.random.normal(600, 150, n),
        np.random.normal(30, 15, n)
    ).clip(0, 1000)

    df = pd.DataFrame({
        "temperatura":  np.round(temperatura, 2),
        "humedad":      np.round(humedad, 2),
        "humedad_aire": np.round(humedad_aire, 2),
        "hora":         horas,
        "luz":          np.round(luz, 2),
    })

    print(f"✅ NumPy generó {n} datos base")
    return df

# =========================
# 🤖 PASO 2: SDV aprende patrones y genera 300 datos
# =========================
def generar_datos_sinteticos(datos_base, cantidad=300):
    """
    SDV aprende las relaciones entre variables del
    dataset base y genera datos sintéticos respetando
    esos patrones (correlaciones, distribuciones, rangos).
    """
    print(f"🧠 SDV aprendiendo patrones de los datos base...")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(datos_base)

    sintetizador = GaussianCopulaSynthesizer(metadata)
    sintetizador.fit(datos_base)

    datos_sinteticos = sintetizador.sample(cantidad)

    # Redondear y limpiar
    datos_sinteticos["temperatura"]  = datos_sinteticos["temperatura"].round(2).clip(10, 45)
    datos_sinteticos["humedad"]      = datos_sinteticos["humedad"].round(2).clip(5, 100)
    datos_sinteticos["humedad_aire"] = datos_sinteticos["humedad_aire"].round(2).clip(20, 95)
    datos_sinteticos["hora"]         = datos_sinteticos["hora"].round(0).astype(int).clip(0, 23)
    datos_sinteticos["luz"]          = datos_sinteticos["luz"].round(2).clip(0, 1000)

    print(f"✅ SDV generó {cantidad} datos sintéticos realistas")
    return datos_sinteticos

# =========================
# 🏷️ PASO 3: Etiqueta de riego (regla base)
# =========================
def etiquetar_riego(df):
    """
    Regla base para etiquetar los datos.
    El modelo ML aprende y mejora esta regla con el tiempo.
    """
    condiciones = (
        (df["humedad"] < 35) |
        ((df["temperatura"] > 30) & (df["humedad"] < 50)) |
        ((df["hora"] >= 10) & (df["hora"] <= 16) & (df["humedad"] < 45))
    )
    df["regar"] = condiciones.astype(int)
    return df

# =========================
# 💾 PASO 4: Guardar en MongoDB con timestamps
# =========================
def guardar_en_mongodb(df):
    """
    Guarda los datos en MongoDB con timestamps
    distribuidos en las últimas 24 horas para
    simular lecturas continuas de sensores.
    """
    ahora = datetime.now(timezone)
    registros = []

    for i, fila in df.iterrows():
        # Distribuye los timestamps en las últimas 24h
        minutos_atras = int((len(df) - i) * (24 * 60 / len(df)))
        timestamp = ahora - timedelta(minutes=minutos_atras)

        registro = {
            "temperatura":  float(fila["temperatura"]),
            "humedad":      float(fila["humedad"]),
            "humedad_aire": float(fila["humedad_aire"]),
            "hora":         int(fila["hora"]),
            "luz":          float(fila["luz"]),
            "regar":        int(fila["regar"]),
            "timestamp":    timestamp,
            "fuente":       "simulado"  # Marca que son datos simulados
        }
        registros.append(registro)

    coleccion.insert_many(registros)
    print(f"💾 {len(registros)} registros guardados en MongoDB")

# =========================
# 🚀 EJECUTAR SIMULACIÓN COMPLETA
# =========================
def ejecutar_simulacion(limpiar_antes=False):
    print("\n🌱 INICIANDO SIMULACIÓN DE SENSORES")
    print("=" * 45)

    if limpiar_antes:
        coleccion.delete_many({})
        print("🗑️  Colección limpiada")

    # Paso 1: NumPy genera datos base
    datos_base = generar_datos_base(n=50)

    # Paso 2: SDV genera datos sintéticos realistas
    datos_sinteticos = generar_datos_sinteticos(datos_base, cantidad=300)

    # Paso 3: Etiqueta de riego
    datos_sinteticos = etiquetar_riego(datos_sinteticos)

    # Resumen
    total    = len(datos_sinteticos)
    si_regar = datos_sinteticos["regar"].sum()
    no_regar = total - si_regar
    print(f"\n📊 Distribución de datos:")
    print(f"   💧 SÍ regar : {si_regar} ({si_regar/total*100:.1f}%)")
    print(f"   🚫 NO regar : {no_regar} ({no_regar/total*100:.1f}%)")

    # Paso 4: Guardar en MongoDB
    guardar_en_mongodb(datos_sinteticos)

    print("\n✅ SIMULACIÓN COMPLETA")
    print(f"   Base de datos: base_datos → mi_coleccion")
    print(f"   Total registros: {total}")
    print("=" * 45)

if __name__ == "__main__":
    ejecutar_simulacion(limpiar_antes=True)
