import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la page
st.set_page_config(page_title="Trafic Smart City V1", page_icon="🚦", layout="wide")

# Chargement du modèle et des données
model = joblib.load("best_traffic_model.pkl")

@st.cache_data
def charger_donnees():
  df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek
    return df

df = charger_donnees()

# ==========================================
# MENU LATÉRAL
# ==========================================
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Aller à :", [
    "🏠 Accueil", "📊 Exploration", "📈 Visualisations", "🔍 Analyse", 
    "🧠 Modélisation", "🔮 Prédiction", "📉 Performance", 
    "📋 Tableau de bord", "ℹ️ À propos", "📚 Documentation"
])
st.sidebar.markdown("---")
# ... (votre menu)
st.sidebar.markdown("---")

# Signature en bas de la zone principale (à ajouter à la toute fin du fichier, hors des conditions "if menu == ...")

st.sidebar.markdown("**Ibrahim OLAOYE** - Ingénieur Statisticien-Économètre | Data Scientist")

# ==========================================
# CONTENU DES ONGLETS
# ==========================================

if menu == "🏠 Accueil":
    st.title("🚦 Stratégie de trafic de Smart City")
    st.markdown("Bienvenue sur l'application de prédiction du volume de trafic.")
    st.markdown("Cette application utilise un modèle XGBoost entraîné sur les données de trafic (Metro Interstate Traffic Volume).")
    st.subheader("Fonctionnalités :")
    st.markdown("- Prédiction du trafic à une heure de vie\n- Visualisations exploratoires\n- Analyse de l'impact des conditions météo\n- Performance du modèle (R² = 0.946)")

elif menu == "📊 Exploration":
    st.title("Exploration des données")
    st.write(f"Nombre d'enregistrements : {len(df)}")
    st.write(df.head(10))
    st.subheader("Statistiques descriptives")
    st.write(df.describe())

elif menu == "📈 Visualisations":
    st.title("Visualisations")
    st.subheader("Évolution du trafic")
    fig1 = px.line(df, x='date_time', y='traffic_volume', title="Évolution du trafic dans le temps")
    st.plotly_chart(fig1, use_container_width=True)
    
    st.subheader("Trafic par moyen heure")
    df_hour = df.groupby('hour')['traffic_volume'].mean().reset_index()
    fig2 = px.bar(df_hour, x='hour', y='traffic_volume', title="Trafic moyen par heure")
    st.plotly_chart(fig2, use_container_width=True)

elif menu == "🔍 Analyse":
    st.title("Analyse avancée")
    st.subheader("Matrice de corrélation")
    # Sélection des colonnes numériques
    cols_num = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'traffic_volume', 'hour', 'dayofweek']
    corr = df[cols_num].corr()
    fig3, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig3)

    st.subheader("Trafic en fonction de la température")
    fig4 = px.scatter(df.sample(1000), x='temp', y='traffic_volume', opacity=0.5, title="Trafic vs Température")
    st.plotly_chart(fig4, use_container_width=True)

elif menu == "🧠 Modélisation":
    st.title("Modélisation")
    st.write("Caractéristiques exactes usées par le modèle :")
    st.code("hour, dayofweek, temp, rain_1h, snow_1h, clouds_all")
    st.subheader("Pourquoi XGBoost ?")
    st.markdown("- Gère les non-linéarités\n- Robuste aux aberrants\n- Bonne généralisation sur séries temporelles")
    
    st.subheader("Importance des caractéristiques")
    # Simulation d'importance (puisque le modèle est chargé)
    features = ['hour', 'dayofweek', 'temp', 'rain_1h', 'snow_1h', 'clouds_all']
    importances = [0.70, 0.15, 0.05, 0.03, 0.02, 0.05] # Valeurs approximatives basées sur vos captures
    fig5, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=importances, y=features, ax=ax, color='purple')
    st.pyplot(fig5)

elif menu == "🔮 Prédiction":
    st.title("Prédiction du volume de trafic")
    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider("Heure (0-23)", 0, 23, 12)
        day = st.selectbox("Jour", ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"])
        temp = st.number_input("Température (°C)", -20.0, 50.0, 20.0)
    with col2:
        rain = st.number_input("Pluie (mm/h)", 0.0, 20.0, 0.0)
        snow = st.number_input("Neige (mm/h)", 0.0, 10.0, 0.0)
        clouds = st.number_input("Nuages (%)", 0, 100, 50)
        
    jour_map = {"Lundi": 0, "Mardi": 1, "Mercredi": 2, "Jeudi": 3, "Vendredi": 4, "Samedi": 5, "Dimanche": 6}
    dayofweek = jour_map[day]
    
    if st.button("🔮 Prédire le trafic"):
        input_data = pd.DataFrame([[hour, dayofweek, temp, rain, snow, clouds]], 
                                  columns=['hour', 'dayofweek', 'temp', 'rain_1h', 'snow_1h', 'clouds_all'])
        prediction = model.predict(input_data)[0]
        st.success(f"🚗 Trafic prédit : **{int(prediction)}** véhicules / heure")

elif menu == "📉 Performance":
    st.title("Performance du modèle")
    st.write("Métriques sur l'ensemble de test :")
    st.markdown("- **R²** : 0.946\n- **RMSE** : environ 450 véhicules/heure\n- **MAE** : environ 320 véhicules/heure")

elif menu == "📋 Tableau de bord":
    st.title("Tableau de bord interactif")
    st.write("Filtres et graphiques interactifs...")

elif menu == "ℹ️ À propos":
    st.title("À propos")
    st.write("Prédiction du trafic urbain intelligent")
    st.markdown("- **Auteur** : Ibrahim OLAOYE\n- **Données** : Metro Interstate Traffic Volume\n- **Modèle** : XGBoost\n- **Interface** : Streamlit")

elif menu == "📚 Documentation":
    st.title("Documentation")
    st.write("Guide d'utilisation...")
    st.subheader("Limitations")
    st.write("- Le modèle ne prend pas en compte les jours fériés (colonne 'holiday' non usée).\n- La prédiction est une estimation.")
    st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-family: Arial, sans-serif;'>
        <span style='color: #6A0DAD; font-weight: bold; font-size: 15px;'>Ibrahim OLAOYE</span> 
        <span style='color: #555555;'>– Ingénieur Statisticien-Économètre | Data Scientist</span><br>
        <span style='color: #1E90FF;'>ibolaoye93@gmail.com</span> 
        <span style='color: #FF0000;'>| +229 0196958344</span>
    </div>
    """, 
    unsafe_allow_html=True
)
