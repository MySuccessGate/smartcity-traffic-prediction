"""
Module de prédiction probabiliste du trafic horaire.
Auteur : Ibrahim OLAOYE
Date : Octobre 2026
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================

def charger_donnees(chemin_csv):
    """Charge le fichier CSV et prépare les features."""
    df = pd.read_csv(chemin_csv)
    
    # Conversion de la date
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Encodage de holiday (1 si férié, 0 sinon)
    df['is_holiday'] = df['holiday'].notna().astype(int)
    
    return df


# ============================================================
# 2. ENTRAÎNEMENT DES MODÈLES
# ============================================================

def entrainer_modeles(df, features):
    """Entraîne 3 modèles XGBoost : médian, quantile bas, quantile haut."""
    
    X = df[features]
    y = df['traffic_volume']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Modèle principal (prédiction ponctuelle)
    model = xgb.XGBRegressor(
        n_estimators=100,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    
    # Modèle quantile bas (2.5%)
    model_low = xgb.XGBRegressor(
        n_estimators=100,
        objective='reg:quantileerror',
        quantile_alpha=0.025,
        random_state=42,
        verbosity=0
    )
    model_low.fit(X_train, y_train)
    
    # Modèle quantile haut (97.5%)
    model_high = xgb.XGBRegressor(
        n_estimators=100,
        objective='reg:quantileerror',
        quantile_alpha=0.975,
        random_state=42,
        verbosity=0
    )
    model_high.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"R²   : {r2:.4f}")
    print(f"RMSE : {rmse:.2f} véhicules/heure")
    print(f"MAE  : {mae:.2f} véhicules/heure")
    
    return model, model_low, model_high, (r2, rmse, mae)


# ============================================================
# 3. PRÉDICTION PROBABILISTE
# ============================================================

def predire_intervalle(model, model_low, model_high, features, valeurs):
    """Retourne la prédiction ponctuelle et l'intervalle de confiance à 95%."""
    X_new = pd.DataFrame([valeurs], columns=features)
    
    pred = model.predict(X_new)[0]
    low = model_low.predict(X_new)[0]
    high = model_high.predict(X_new)[0]
    
    return {
        'prediction': int(round(pred)),
        'borne_basse': int(round(low)),
        'borne_haute': int(round(high)),
        'marge': int(round(high - low))
    }


# ============================================================
# 4. TEST
# ============================================================

if __name__ == "__main__":
    
    # Chemin vers votre fichier CSV (dans le dossier data)
    CHEMIN_CSV = "data/Metro_Interstate_Traffic_Volume.csv"
    
    print("=" * 60)
    print("MODULE DE PRÉDICTION PROBABILISTE - TEST")
    print("=" * 60)
    
    # Chargement
    df = charger_donnees(CHEMIN_CSV)
    
    # Features (avec is_holiday ajouté)
    features = ['hour', 'dayofweek', 'temp', 'rain_1h', 
                'snow_1h', 'clouds_all', 'is_holiday']
    
    # Entraînement
    print("\n[1] Entraînement des modèles...")
    model, model_low, model_high, metriques = entrainer_modeles(df, features)
    
    # Test de prédiction
    print("\n[2] Test de prédiction probabiliste...")
    print("-" * 60)
    
    # Exemple 1 : Lundi 17h, 20°C, pas de pluie, 40% nuages
    resultat = predire_intervalle(
        model, model_low, model_high, features,
        valeurs=[17, 0, 293.15, 0, 0, 40, 0]
    )
    print(f"Lundi 17h, 20°C, ciel nuageux :")
    print(f"  -> Prediction : {resultat['prediction']} vehicules")
    print(f"  -> Intervalle 95% : [{resultat['borne_basse']} ; {resultat['borne_haute']}]")
    print(f"  -> Marge : +-{resultat['marge']} vehicules")
    
    # Exemple 2 : Dimanche 8h, 15°C, pluie légère
    resultat2 = predire_intervalle(
        model, model_low, model_high, features,
        valeurs=[8, 6, 288.15, 1.0, 0, 75, 0]
    )
    print(f"\nDimanche 8h, 15°C, pluie legere :")
    print(f"  -> Prediction : {resultat2['prediction']} vehicules")
    print(f"  -> Intervalle 95% : [{resultat2['borne_basse']} ; {resultat2['borne_haute']}]")
    print(f"  -> Marge : +-{resultat2['marge']} vehicules")
    
    # Exemple 3 : Jour férié, Mardi 12h
    resultat3 = predire_intervalle(
        model, model_low, model_high, features,
        valeurs=[12, 1, 295.15, 0, 0, 20, 1]
    )
    print(f"\nMardi 12h, jour ferie, ciel clair :")
    print(f"  -> Prediction : {resultat3['prediction']} vehicules")
    print(f"  -> Intervalle 95% : [{resultat3['borne_basse']} ; {resultat3['borne_haute']}]")
    print(f"  -> Marge : +-{resultat3['marge']} vehicules")
    
    print("\n" + "=" * 60)
    print("TEST TERMINE AVEC SUCCES")
    print("=" * 60)