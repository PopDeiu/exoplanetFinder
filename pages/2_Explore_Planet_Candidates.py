# pages/2_Explore_Planet_Candidates.py

import streamlit as st
from utils import fetch_catalog_targets

st.header("Descoperă candidați interesanți la planete")

st.caption("Pasul 2 din 4 · Explorează sisteme în care au fost deja identificate planete sau candidați.")

with st.expander("ℹ️ Ce este această pagină?", expanded=False):
    st.markdown(
        """
        - Alege o misiune (TESS sau Kepler) pentru a descărca o listă de ținte interesante.  
        - Fiecare rând reprezintă o stea cu cel puțin o planetă confirmată sau un candidat la planetă.  
        - Poți copia coloana **„ID căutabil”** și o poți lipi în pagina „Caută o stea” pentru o analiză detaliată a curbei de lumină.
        """
    )
st.markdown("Alege o misiune pentru a obține un eșantion de stele cu planete confirmate sau candidați.")
mission_choice_planets = st.selectbox("Alege o misiune:", options=["TESS", "Kepler"], index=0, key="planet_mission_select")

if st.button("Descarcă candidați la planete", type="primary"):
    with st.spinner(f"Se descarcă catalogul {mission_choice_planets}... (se actualizează zilnic)"):
        st.session_state.explore_planets_results = fetch_catalog_targets(mission_choice_planets, disposition_type="PLANETS")

if st.session_state.explore_planets_results is not None:
    if st.session_state.explore_planets_results.empty:
        st.error("Nu s‑au găsit ținte. Este posibil să existe o problemă temporară cu arhiva de date.")
    else:
        st.markdown(f"#### Sample of Planet Candidates from **{mission_choice_planets}**:")
        st.dataframe(st.session_state.explore_planets_results)
        st.info("Poți copia un „ID căutabil” și îl poți lipi în pagina „Caută o stea”.")
