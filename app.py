import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shap

st.title("Traffic Intelligence System - Version 2.0")
st.markdown("### Plateforme d'aide à la décision pour la Smart City")

@st.cache_resource
def charger_et_entrainer():
    df = pd.read_csv("data/Metro_Interstate_Traffic_Volume.csv")
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['is_holiday'] = df['holiday'].notna().astype(int)
    
    features = ['hour', 'dayofweek', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'is_holiday']
    X = df[features]
    y = df['traffic_volume']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    
    model_low = xgb.XGBRegressor(n_estimators=100, objective='reg:quantileerror', quantile_alpha=0.025, random_state=42, verbosity=0)
    model_low.fit(X_train, y_train)
    
    model_high = xgb.XGBRegressor(n_estimators=100, objective='reg:quantileerror', quantile_alpha=0.975, random_state=42, verbosity=0)
    model_high.fit(X_train, y_train)
    
    return model, model_low, model_high, features

with st.spinner("Chargement du modèle et des données..."):
    model, model_low, model_high, features = charger_et_entrainer()

st.sidebar.header("Paramètres de simulation")
hour = st.sidebar.slider("Heure (0-23)", 0, 23, 12)
day = st.sidebar.selectbox("Jour", ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"])
temp = st.sidebar.number_input("Température (°C)", -20.0, 50.0, 20.0)
rain = st.sidebar.number_input("Pluie (mm/h)", 0.0, 20.0, 0.0)
snow = st.sidebar.number_input("Neige (mm/h)", 0.0, 10.0, 0.0)
clouds = st.sidebar.number_input("Nuages (%)", 0, 100, 50)
is_holiday = st.sidebar.checkbox("Est un jour férié ?", value=False)

jour_map = {"Lundi": 0, "Mardi": 1, "Mercredi": 2, "Jeudi": 3, "Vendredi": 4, "Samedi": 5, "Dimanche": 6}
dayofweek = jour_map[day]
input_data = pd.DataFrame([[hour, dayofweek, temp, rain, snow, clouds, int(is_holiday)]], columns=features)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Option A : Prédiction Probabiliste")
    if st.button("Lancer la prédiction"):
        pred = model.predict(input_data)[0]
        low = model_low.predict(input_data)[0]
        high = model_high.predict(input_data)[0]
        st.metric(label="Prédiction médiane", value=f"{int(pred)} véhicules/h")
        st.info(f"**Intervalle de confiance à 95% :**\n\n[{int(low)} ; {int(high)}]\n\n*Marge : ±{int(high - low)} véhicules*")
with col2:
    st.subheader("Option B : Explicabilité (SHAP)")
    if st.button("Expliquer la prédiction"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.bar_plot(shap_values[0], feature_names=features, max_display=10, show=False)
        st.pyplot(fig)

st.markdown("---")
st.subheader("Option C : Simulation de scénarios")
col_sim1, col_sim2 = st.columns(2)
with col_sim1:
    st.markdown("**Scénario actuel**")
    st.write(f"- Heure : {hour}h")
    st.write(f"- Jour : {day}")
    st.write(f"- Température : {temp}°C")
    st.write(f"- Pluie : {rain} mm/h")
with col_sim2:
    st.markdown("**Scénario alternatif**")
    hour_alt = st.slider("Heure alternative", 0, 23, 17, key="hour_alt")
    temp_alt = st.number_input("Température alternative (°C)", -20.0, 50.0, 10.0, key="temp_alt")
    rain_alt = st.number_input("Pluie alternative (mm/h)", 0.0, 20.0, 5.0, key="rain_alt")

if st.button("Comparer les scénarios"):
    input_actuel = pd.DataFrame([[hour, dayofweek, temp, rain, snow, clouds, int(is_holiday)]], columns=features)
    pred_actuel = model.predict(input_actuel)[0]
    input_alt = pd.DataFrame([[hour_alt, dayofweek, temp_alt, rain_alt, snow, clouds, int(is_holiday)]], columns=features)
    pred_alt = model.predict(input_alt)[0]
    
    diff = pred_alt - pred_actuel
    pct = (diff / pred_actuel) * 100
    
    st.markdown("### Résultats de la comparaison")
    col_res1, col_res2, col_res3 = st.columns(3)
    col_res1.metric("Scénario actuel", f"{int(pred_actuel)} véhicules/h")
    col_res2.metric("Scénario alternatif", f"{int(pred_alt)} véhicules/h", delta=f"{int(diff)}")
    col_res3.metric("Variation", f"{pct:.1f}%")

# --- SIGNATURE EN BAS DE PAGE ---
st.markdown("---")
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-family: Arial, sans-serif; font-size: 14px;'>
        <span style='color: #0066CC; font-weight: bold; font-size: 16px;'>Ibrahim OLAOYE</span><br>
        <span style='color: #333333;'>Ingénieur Statisticien-Économètre | Data Scientist</span><br>
        <span style='color: #666666; font-size: 13px;'>ibolaoye93@gmail.com | +229 0196958344</span>
    </div>
    """, 
    unsafe_allow_html=True
)
