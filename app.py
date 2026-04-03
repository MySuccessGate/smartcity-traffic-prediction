import streamlit as st
import pandas as pd
import joblib

# Configuration de la page
st.set_page_config(page_title="Prédiction de Trafic Smart City", layout="centered")

# Titre
st.title("🚦 Prédiction de Trafic Smart City")
st.markdown("Prédisez le volume de trafic pour une heure donnée.")

# Chargement du modèle (avec cache)
@st.cache_resource
def load_model():
    return joblib.load("best_traffic_model.pkl")

model = load_model()

# Interface utilisateur
col1, col2 = st.columns(2)

with col1:
    hour = st.slider("Heure (0-23)", 0, 23, 12)
    day = st.selectbox("Jour", ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"])
    temp = st.number_input("Température (°C)", -20.0, 50.0, 20.0)

with col2:
    rain = st.number_input("Pluie (mm/h)", 0.0, 20.0, 0.0)
    snow = st.number_input("Neige (mm/h)", 0.0, 10.0, 0.0)
    clouds = st.number_input("Nuages (%)", 0, 100, 50)

# Mapping jour -> dayofweek (lundi=0)
jour_map = {
    "Lundi": 0, "Mardi": 1, "Mercredi": 2,
    "Jeudi": 3, "Vendredi": 4, "Samedi": 5, "Dimanche": 6
}
dayofweek = jour_map[day]

# Bouton de prédiction
if st.button("🔮 Prédire le trafic"):
    # Créer un DataFrame avec les colonnes exactes attendues par le modèle
    input_data = pd.DataFrame([[hour, dayofweek, temp, rain, snow, clouds]],
                              columns=['hour', 'dayofweek', 'temp', 'rain_1h', 'snow_1h', 'clouds_all'])
    prediction = model.predict(input_data)[0]
    st.success(f"🚗 Trafic prédit : **{int(prediction)}** véhicules / heure")
