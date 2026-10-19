import streamlit as st

# Configuration de la page
st.set_page_config(page_title="Traffic Intelligence System", page_icon="🚦", layout="wide")

# Gestion de la transition via les paramètres de l'URL
if "page" not in st.query_params:
    st.query_params["page"] = "v1"

# ============================================================
# DÉFINITION DES PAGES
# ============================================================

# Page 1 : Ancien modèle (app_v1_ancien.py)
page_v1 = st.Page("app_v1_ancien.py", title="Ancien Modèle (V1)", icon="🚦")

# Page 2 : Transition
page_transition = st.Page("page_transition.py", title="Mise à jour (V2.0)", icon="🔄")

# Page 3 : Nouveau modèle (app_v2.py)
page_v2 = st.Page("app_v2.py", title="Intelligence Décisionnelle (V2.0)", icon="🧠")

# ============================================================
# NAVIGATION
# ============================================================
# On crée un menu de navigation
pg = st.navigation({
    "Version 1.0": [page_v1],
    "Transition": [page_transition],
    "Version 2.0": [page_v2]
})

# On lance la page sélectionnée
pg.run()
