# pages/3_Explore_False_Positives.py

import streamlit as st
from utils import fetch_catalog_targets

st.header("Semnale care NU sunt planete (false pozitive)")

st.caption("Pasul 3 din 4 · Învață cum arată semnalele care pot imita tranzitele de exoplanete.")

with st.expander("ℹ️ De ce sunt importante falsele pozitive?", expanded=False):
    st.markdown(
        """
        Semnalele de tip *fals pozitiv* ajută la antrenarea intuiției (și a algoritmilor) pentru a diferenția planetele reale
        de efecte precum stele binare eclipsante, zgomot instrumental sau alte artefacte.  
        Poți folosi aceste ținte drept exemple „negative” atunci când testezi algoritmi noi de detectare.
        """
    )
st.markdown("Găsește stele cu semnale care arată ca niște planete, dar sunt cauzate de alte fenomene.")
# --- NEW: Explanation for False Positives ---
st.info(
    "Un **fals pozitiv** apare atunci când o scădere a luminozității unei stele este cauzată de altceva decât o planetă în tranzit."
    "Cauze frecvente includ: 

"
    "* **Stele binare eclipsante:** Două stele care se orbitează și se eclipsează una pe cealaltă. 
"
    "* **Zgomot instrumental:** Defecțiuni sau erori ale instrumentelor telescopului. 

"
    "Studierea acestor semnale este esențială pentru a antrena algoritmii să distingă planetele reale de semnalele false."
)

mission_choice_fps = st.selectbox("Alege o misiune:", options=["TESS", "Kepler"], index=0, key="fp_mission_select")

if st.button("Descarcă falsii pozitivi", type="primary"):
    with st.spinner(f"Se descarcă catalogul {mission_choice_fps}... (se actualizează zilnic)"):
        st.session_state.explore_fps_results = fetch_catalog_targets(mission_choice_fps, disposition_type="FALSE_POSITIVES")

if st.session_state.explore_fps_results is not None:
    if st.session_state.explore_fps_results.empty:
        st.error("Nu s‑au găsit ținte pentru această categorie.")
    else:
        st.markdown(f"#### Sample of False Positives from **{mission_choice_fps}**:")
        st.dataframe(st.session_state.explore_fps_results)
        st.info("Poți copia un „ID căutabil” și îl poți lipi în pagina „Caută o stea”.")
