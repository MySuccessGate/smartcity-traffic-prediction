# 🚦 Smart City Traffic Prediction

Application web de prédiction du volume de trafic routier pour une smart city, basée sur un modèle XGBoost et déployée avec Streamlit Cloud.

🔗 **Accéder à l'application en ligne** : [Cliquez ici](https://smartcity-traffic-prediction-4s36wqedvdgcqtgwdzcmsc.streamlit.app/)

## 📌 Contexte
Ce projet aide à anticiper le nombre de véhicules par heure en fonction de l'heure, du jour, de la température, des précipitations et de la nébulosité.

## 📊 Données
Metro Interstate Traffic Volume (UCI/Kaggle) : 48 000+ enregistrements horaires de 2012 à 2018.

## 🧠 Modèle
- **Algorithme** : XGBoost
- **Variables** : heure, jour de semaine, température, pluie, neige, couverture nuageuse
- **Performance** : R² = 0.946

## 🖥️ Fonctionnalités (10 onglets)
- Accueil, Exploration, Visualisations, Analyse, Modélisation, Prédiction, Performance, Dashboard, À propos, Documentation

## 🚀 Déploiement
Hébergé sur Streamlit Cloud. Mise à jour automatique via GitHub.

## 📁 Structure
- `app.py` : code principal
- `best_traffic_model.pkl` : modèle entraîné
- `Metro_Interstate_Traffic_Volume.csv` : données
- `requirements.txt` : dépendances
- `README.md` : ce fichier

## 🛠️ Installation locale
```bash
git clone https://github.com/MySuccessGate/smartcity-traffic-prediction.git
cd smartcity-traffic-prediction
pip install -r requirements.txt
streamlit run app.py

## 👤 Auteur

**Ibrahim OLAOYE; Ingénieur Statisticien-Économètre | Data Scientist** – Projet réalisé dans le cadre d'une initiative sur les villes intelligentes.

## 📜 Licence

Ce projet est open source – vous pouvez l'utiliser et l'adapter librement.

---

✨ *N'hésitez pas à tester l'application et à me faire part de vos retours !*
