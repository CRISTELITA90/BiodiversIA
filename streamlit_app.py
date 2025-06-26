import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib

# Título del dashboard
st.set_page_config(page_title="BiodiversIA", layout="wide")
st.title("🧬 BiodiversIA – Predicción de especies en Canarias")

# Cargar datos
@st.cache_data
def load_data():
    return pd.read_csv("datos_especies_canarias.csv")

df = load_data()

# Cargar modelo
@st.cache_resource
def load_model():
    return joblib.load("modelo_random_forest.pkl")

model = load_model()

# Mapa interactivo
st.subheader("📍 Mapa de observaciones")
st.write("Visualiza la distribución geográfica de las especies predichas.")

# Filtrado por isla
islas = df["island"].dropna().unique()
isla_seleccionada = st.selectbox("Selecciona una isla", islas)

df_filtrado = df[df["island"] == isla_seleccionada]

m = folium.Map(location=[df_filtrado["latitude"].mean(), df_filtrado["longitude"].mean()], zoom_start=8)
for _, row in df_filtrado.iterrows():
    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=f"Especie: {row['species']}<br>Predicción: {row.get('predicted_species', 'N/A')}",
        icon=folium.Icon(color="green", icon="leaf")
    ).add_to(m)

st_data = st_folium(m, width=700)

# Simulación de predicción
st.subheader("🔮 Simulador de predicción de especie")

with st.form("prediction_form"):
    lat = st.number_input("Latitud", value=28.1)
    lon = st.number_input("Longitud", value=-15.4)
    temp = st.slider("Temperatura (°C)", 10.0, 35.0, 22.0)
    precip = st.slider("Precipitación (mm)", 0.0, 300.0, 50.0)
    ndvi = st.slider("Índice NDVI", -1.0, 1.0, 0.4)
    
    submitted = st.form_submit_button("Predecir especie")

    if submitted:
        input_data = pd.DataFrame([{
            "latitude": lat,
            "longitude": lon,
            "temperature": temp,
            "precipitation": precip,
            "ndvi": ndvi
        }])
        prediction = model.predict(input_data)[0]
        st.success(f"🌱 Especie predicha: **{prediction}**")

# Pie de página
st.markdown("---")
st.markdown("📌 Proyecto desarrollado por **Cristela Moreno García** – [GitHub](https://github.com/CRISTELITA90)")
