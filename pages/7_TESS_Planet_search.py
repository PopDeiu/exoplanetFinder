# pages/7_TESS_Planet_Search.py

import streamlit as st
from utils import search_toi_catalog, set_galaxy_background, set_sidebar_style
import pandas as pd

st.set_page_config(
    page_title="Căutare de planete TESS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Aplicare styling cosmic ---
set_sidebar_style()
set_galaxy_background("stellar")

st.header("🔍 Căutare de planete TESS")
st.caption("Interfață pentru filtrarea și explorarea catalogului oficial TOI (TESS Objects of Interest) al misiunii NASA.")

st.markdown("""
Folosește filtrele de mai jos pentru a interoga baza de date **ExoFOP**. 
Acest catalog conține semnalele de tranzit detectate de telescopul TESS care sunt în curs de investigare sau deja confirmate.
""")

# --- Filtre de căutare ---
with st.expander("🛠️ Filtre Avansate", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        # Definirea opțiunilor complete de stare (Disposition)
        disposition_options = {
            "CP": "Planetă Confirmată (Confirmed Planet)",
            "PC": "Candidat (Planet Candidate)",
            "KP": "Planetă Cunoscută (Known Planet)",
            "EB": "Binară cu Eclipsă (Eclipsing Binary)",
            "FP": "Fals Pozitiv (False Positive)",
            "FA": "Alarmă Falsă (False Alarm)"
        }

        dispositions = st.multiselect(
            "Starea planetei (Disposition)",
            options=list(disposition_options.keys()),
            default=["CP", "PC"],
            format_func=lambda x: disposition_options[x],
            help="**CP/KP:** Planete sigure. **PC:** Posibile planete. **EB/FP:** Obiecte care s-au dovedit a nu fi planete."
        )

        radius_range = st.slider(
            "Raza Planetei (Raze Terestre)",
            min_value=0.1, max_value=25.0, value=(0.8, 4.0), step=0.1,
            help="Pământul are 1.0. Jupiter are aprox. 11.0."
        )

    with col2:
        tic_id = st.text_input(
            "ID TIC al stelei (Ex: 25155310)",
            placeholder="Introduceți doar cifre...",
            help="Dacă aveți un ID TIC specific, introduceți-l aici pentru a vedea toți candidații acelei stele."
        )

        period_range = st.slider(
            "Perioadă orbitală (zile)",
            min_value=0.1, max_value=500.0, value=(1.0, 30.0), step=0.1,
            help="Durata unui 'an' pe acea planetă."
        )

# --- Execuție Căutare ---
if 'toi_search_results' not in st.session_state:
    st.session_state.toi_search_results = None

if st.button("🚀 Caută în catalogul TESS", type="primary", use_container_width=True):
    search_tic = None
    if tic_id and tic_id.strip().isdigit():
        search_tic = int(tic_id.strip())

    with st.spinner("Se interoghează baza de date ExoFOP..."):
        # Apelăm funcția robustă din utils/data_fetchers.py
        results_df = search_toi_catalog(
            dispositions=dispositions,
            radius_range=radius_range,
            period_range=period_range,
            tic_id=search_tic
        )
        st.session_state.toi_search_results = results_df

# --- Afișare Rezultate ---
if st.session_state.toi_search_results is not None:
    st.divider()
    df = st.session_state.toi_search_results

    if df.empty:
        st.warning("Nu am găsit niciun obiect care să corespundă criteriilor. Încearcă să lărgești intervalele de căutare.")
    else:
        st.success(f"Am găsit **{len(df)} obiecte (TOIs)** conform filtrelor selectate.")

        # Configurare coloane pentru un aspect profi
        st.dataframe(
            df,
            column_config={
                "TIC ID": st.column_config.TextColumn("ID TIC"),
                "TOI": st.column_config.NumberColumn("Nume TOI", format="%d"),
                "TFOPWG Disposition": "Stare",
                "Planet Radius (R_earth)": st.column_config.NumberColumn("Rază (R⊕)", format="%.2f"),
                "Orbital Period (days)": st.column_config.NumberColumn("Perioadă (zile)", format="%.3f"),
                "Planet Temp (K)": st.column_config.NumberColumn("Temp. Planetă", format="%d K"),
                "Stellar Radius (R_sun)": st.column_config.NumberColumn("Rază Stea", format="%.2f R⊙"),
                "Stellar Teff (K)": st.column_config.NumberColumn("Temp. Stea", format="%d K"),
            },
            use_container_width=True,
            hide_index=True
        )

        st.info("💡 **Sfat:** Copiază un **ID TIC** din tabel și mergi la pagina **'Caută o stea'** pentru a genera curba de lumină și a vedea tranzitul real!")
