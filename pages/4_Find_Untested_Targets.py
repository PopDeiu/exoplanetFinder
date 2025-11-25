# pages/4_Find_Untested_Targets.py

import streamlit as st
from utils import fetch_untested_targets

st.header("Găsește ținte posibil netestate 🔭")
st.caption("Descoperă stele interesante care nu au încă planete candidate asociate.")
st.markdown("Acest instrument preia un eșantion aleatoriu de stele luminoase și apropiate care apar în cataloagele TESS, dar nu au încă un candidat cunoscut la planetă. Este un loc foarte bun pentru a căuta semnale noi!")

if st.button("Găsește stele netestate", type="primary", key="fetch_untested"):
    with st.spinner("Se corelează cataloagele TESS... (rezultatele sunt memorate zilnic; prima rulare poate dura puțin)"):
        st.session_state.untested_results = fetch_untested_targets()
        
if st.session_state.untested_results is not None:
    if st.session_state.untested_results.empty:
        st.error("Nu s‑au găsit ținte netestate în eșantionul aleatoriu sau a apărut o eroare.")
    else:
        st.markdown("#### Potentially Un-tested TESS Targets:")
        # --- NEW/MODIFIED: Added help text to dataframe columns ---
        st.dataframe(
            st.session_state.untested_results,
            column_config={
                "TESS Magnitude": st.column_config.NumberColumn(
                    help="The star's brightness in the TESS band. Lower numbers are brighter."
                ),
                "Distance (pc)": st.column_config.NumberColumn(
                    help="The distance to the star in parsecs. 1 parsec = 3.26 light-years."
                ),
            }
        )
        st.info("Copiază un „ID căutabil” și lipește‑l în pagina „Caută o stea” pentru a o analiza.")
