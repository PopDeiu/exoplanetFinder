# pages/7_TESS_Planet_Search.py

import streamlit as st
import numpy as np
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
            "CP": "Planetă Confirmată",
            "PC": "Candidat la Planetă",
            "KP": "Planetă Cunoscută",
            "EB": "Binară cu Eclipsă",
            "FP": "Fals Pozitiv",
            "FA": "Alarmă Falsă"
        }

        dispositions = st.multiselect(
            "Starea planetei (Disposition)",
            options=list(disposition_options.keys()),
            default=["CP", "PC"],
            format_func=lambda x: disposition_options[x],
            help="**CP/KP:** Planete sigure. **PC:** Posibile planete. **EB/FP:** Obiecte care s-au dovedit a nu fi planete."
        )

        mass_range = st.slider(
            "Masa Planetei (Mase Terestre)",
            min_value=0.1, max_value=30.0, value=(1.0, 10.0), step=0.1,
            help="Pământul are 1.0 M⊕. Jupiter are ~318 M⊕."
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
            mass_range=mass_range,
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

        # Adăugăm coloane calculate
        df = df.copy()
        if 'Stellar Distance (pc)' in df.columns:
            df['Distanta (ani-lumina)'] = pd.to_numeric(df['Stellar Distance (pc)'], errors='coerce') * 3.26156

        # Calcul semiaxa mare (AU) din perioadă și masa stelei (Kepler 3: a³ = M * P²)
        if 'Period (days)' in df.columns and 'Stellar Mass (M_Sun)' in df.columns:
            P = pd.to_numeric(df['Period (days)'], errors='coerce')
            M = pd.to_numeric(df['Stellar Mass (M_Sun)'], errors='coerce')
            df['Semiaxa (AU)'] = (M * (P / 365.25) ** 2) ** (1/3)

        # Calcul zona locuibilă a stelei și verificare
        if 'Stellar Radius (R_Sun)' in df.columns and 'Stellar Eff Temp (K)' in df.columns:
            R = pd.to_numeric(df['Stellar Radius (R_Sun)'], errors='coerce')
            T = pd.to_numeric(df['Stellar Eff Temp (K)'], errors='coerce')
            L = (R ** 2) * ((T / 5778) ** 4)
            df['HZ_int (AU)'] = np.sqrt(L / 1.1)
            df['HZ_ext (AU)'] = np.sqrt(L / 0.53)

            if 'Semiaxa (AU)' in df.columns:
                a = df['Semiaxa (AU)']
                in_hz = (a >= df['HZ_int (AU)']) & (a <= df['HZ_ext (AU)'])
                df['In Zona Locuibila'] = in_hz.map({True: '🌍 Da', False: '❌ Nu'})

        # Configurare coloane relevante
        col_config = {
            "TOI": st.column_config.NumberColumn("TOI", format="%d"),
            "TIC ID": st.column_config.TextColumn("ID TIC"),
            "TFOPWG Disposition": "Stare",
            "Period (days)": st.column_config.NumberColumn("Perioadă (zile)", format="%.3f"),
            "Semiaxa (AU)": st.column_config.NumberColumn("Distanță de stea (AU)", format="%.3f"),
            "Predicted Mass (M_Earth)": st.column_config.NumberColumn("Masă (M⊕)", format="%.2f"),
            "Planet Equil Temp (K)": st.column_config.NumberColumn("Temp. echilibru (K)", format="%d"),
            "Planet Insolation (Earth Flux)": st.column_config.NumberColumn("Iradiație (×Pământ)", format="%.2f"),
            "HZ_int (AU)": st.column_config.NumberColumn("Zonă locuibilă (int)", format="%.2f"),
            "HZ_ext (AU)": st.column_config.NumberColumn("Zonă locuibilă (ext)", format="%.2f"),
            "In Zona Locuibila": "În Zona Locuibilă?",
            "Distanta (ani-lumina)": st.column_config.NumberColumn("Distanță (ani-lumină)", format="%.1f"),
        }
        coloane_de_afisat = [c for c in col_config if c in df.columns]

        st.dataframe(
            df[coloane_de_afisat],
            column_config=col_config,
            use_container_width=True,
            hide_index=True
        )

        st.info("💡 **Sfat:** Copiază un **ID TIC** din tabel și mergi la pagina **'Caută o stea'** pentru a genera curba de lumină și a vedea tranzitul real!")
