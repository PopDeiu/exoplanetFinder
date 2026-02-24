# pages/5_Setări.py

import streamlit as st
from utils import set_galaxy_background, set_sidebar_style

st.set_page_config(
    page_title="Setări",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Aplicare styling IMEDIAT ---
set_sidebar_style()
set_galaxy_background("nebula")

st.header("Setări")

st.subheader("Parametri de căutare")
st.info("Aceste filtre se aplică doar când cauți după nume pe pagina „Caută o stea”.")
st.multiselect("Alege misiunile", options=["TESS", "Kepler", "K2"], key="selected_missions")
st.multiselect("Alege autorii/pipeline-urile de date", options=["SPOC", "Kepler", "K2", "QLP", "TESS-SPOC"], key="selected_authors")

st.divider()

st.subheader("Controale de analiză")

st.caption("Controlează cât de sensibilă este căutarea tranzitelor și cât de mult este netezită curba de lumină.")
st.info("Ajustează acești parametri pentru analiza efectuată pe pagina „Caută o stea”.")

st.slider(
    'Dimensiunea bin-ului (minute)', min_value=1, max_value=60, key='bin_size',
    help="Gruparea punctelor de date în intervale de timp mai mari reduce zgomotul. Valorile mai mari netezesc curba, dar pot ascunde tranzitele foarte scurte."
)
st.slider(
    'Sigma pentru eliminarea valorilor aberante', min_value=1.0, max_value=10.0, step=0.1, key='sigma_val',
    help="Cât de agresiv sunt eliminate valorile aberante. Valorile mai mici elimină mai multe puncte; valorile mai mari păstrează mai multă variabilitate. Punctele care se abat cu mai mult decât acest număr de deviații standard sunt eliminate."
)

st.subheader("Setări periodogramă")
st.slider(
    "Interval de perioade căutat (zile)", min_value=0.5, max_value=100.0, key='period_range',
    help="Intervalul de perioade orbitale în care se caută semnale. Intervalele mai mici duc, de obicei, la calcule mai rapide."
)
