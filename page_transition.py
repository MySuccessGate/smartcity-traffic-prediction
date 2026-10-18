import streamlit as st
import time

st.title("Prédiction de Trafic Smart City - Version 1.0")
st.markdown("Prédisez le volume de trafic pour une heure donnée.")

st.info("Bienvenue sur la Version 1.0. **Pour découvrir la nouvelle version (V2.0), veuillez faire défiler la page vers le bas.**")

st.markdown("---")
st.markdown("### Prêt pour la suite ?")

# Le bouton de transition
if st.button("Découvrir la nouvelle version (V2.0)", type="primary", use_container_width=True):
    
    # --- L'EFFET VISUEL (Barre de progression) ---
    progress_text = "Initialisation du système..."
    my_bar = st.progress(0, text=progress_text)

    time.sleep(1)
    my_bar.progress(30, text="Chargement des modèles d'intelligence artificielle...")
    
    time.sleep(1)
    my_bar.progress(60, text="Connexion aux données de trafic en temps réel...")
    
    time.sleep(1)
    my_bar.progress(90, text="Préparation de l'interface décisionnelle...")
    
    time.sleep(1)
    my_bar.progress(100, text="Système prêt.")
    time.sleep(0.5) # Petite pause finale
    
    # Bascule vers la V2
    st.switch_page("app_v2.py")

# --- SIGNATURE ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-family: Arial, sans-serif; font-size: 14px; color: #555555;'>
        <span style='color: #1E3A8A; font-weight: bold; font-size: 15px;'>Ibrahim OLAOYE</span> 
        <span style='color: #888888;'>| Ingénieur Statisticien-Économètre | Data Scientist</span><br>
        <span style='color: #666666;'>ibolaoye93@gmail.com | +229 0196958344</span>
    </div>
    """, 
    unsafe_allow_html=True
)