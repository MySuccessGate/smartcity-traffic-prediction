import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration
st.set_page_config(page_title="Smart City Traffic", layout="wide")

# Chargement des données et du modèle
@st.cache_data
def load_data():
    df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    return df

@st.cache_resource
def load_model():
    return joblib.load("best_traffic_model.pkl")

df = load_data()
model = load_model()

# Création des onglets
tab1, tab2 = st.tabs(["🚦 Prédiction de trafic", "📈 Visualisations et analyse"])

# ==================== ONGLET 1 : PRÉDICTION ====================
with tab1:
    st.title("Prédiction du volume de trafic")
    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider("Heure (0-23)", 0, 23, 12)
        day = st.selectbox("Jour", ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"])
        temp = st.number_input("Température (°C)", -20.0, 50.0, 20.0)
    with col2:
        rain = st.number_input("Pluie (mm/h)", 0.0, 20.0, 0.0)
        snow = st.number_input("Neige (mm/h)", 0.0, 10.0, 0.0)
        clouds = st.number_input("Nuages (%)", 0, 100, 50)
    
    jour_map = {"Lundi":0,"Mardi":1,"Mercredi":2,"Jeudi":3,"Vendredi":4,"Samedi":5,"Dimanche":6}
    dayofweek = jour_map[day]
    
    if st.button("Prédire"):
        input_data = pd.DataFrame([[hour, dayofweek, temp, rain, snow, clouds]],
                                  columns=['hour','dayofweek','temp','rain_1h','snow_1h','clouds_all'])
        pred = model.predict(input_data)[0]
        st.success(f"🚗 Trafic prédit : **{int(pred)}** véhicules/heure")

# ==================== ONGLET 2 : VISUALISATIONS ====================
with tab2:
    st.header("Analyse exploratoire des données")
    
    # Série temporelle du trafic (échantillon pour performance)
    st.subheader("Évolution du trafic dans le temps")
    sample = df.sample(min(5000, len(df)))  # éviter de surcharger
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(sample['date_time'], sample['traffic_volume'], alpha=0.5, linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume de trafic")
    st.pyplot(fig)
    
    # Distribution du trafic par heure
    st.subheader("Trafic moyen par heure")
    hourly = df.groupby('hour')['traffic_volume'].mean().reset_index()
    fig2, ax2 = plt.subplots()
    ax2.bar(hourly['hour'], hourly['traffic_volume'], color='skyblue')
    ax2.set_xlabel("Heure")
    ax2.set_ylabel("Trafic moyen")
    st.pyplot(fig2)
    
    # Heatmap des corrélations (entre variables numériques)
    st.subheader("Corrélations entre variables")
    numeric_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'traffic_volume', 'hour', 'dayofweek']
    corr = df[numeric_cols].corr()
    fig3, ax3 = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)
    
    # Impact de la température
    st.subheader("Relation température / trafic")
    fig4, ax4 = plt.subplots()
    ax4.scatter(df['temp'], df['traffic_volume'], alpha=0.3, s=1)
    ax4.set_xlabel("Température (°C)")
    ax4.set_ylabel("Trafic")
    st.pyplot(fig4)
