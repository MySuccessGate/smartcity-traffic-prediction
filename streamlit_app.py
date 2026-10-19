import streamlit as st

st.set_page_config(page_title="Traffic Intelligence System", page_icon="🚦", layout="wide")

page_v1 = st.Page("app_v1_ancien.py", title="Ancien Modèle (V1)", icon="🚦")
page_transition = st.Page("page_transition.py", title="Mise à jour (V2.0)", icon="🔄")
page_v2 = st.Page("app_v2.py", title="Intelligence Décisionnelle (V2.0)", icon="🧠")

pg = st.navigation({
    "Version 1.0": [page_v1],
    "Transition": [page_transition],
    "Version 2.0": [page_v2]
})

pg.run()
