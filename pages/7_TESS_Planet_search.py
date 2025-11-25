# pages/7_TESS_Planet_Search.py

import streamlit as st
from utils import search_toi_catalog
import pandas as pd

st.set_page_config(page_title="Căutare de planete TESS")

st.header("🛰️ Căutare de planete TESS")
st.caption("Interfață pentru filtrarea și explorarea catalogului oficial TOI al misiunii TESS.")

st.markdown("Folosește filtrele de mai jos pentru a căuta în catalogul oficial **TESS Objects of Interest (TOI)** din ExoFOP.")

# --- Filtre de căutare ---
st.subheader("Filtre de căutare")

col1, col2 = st.columns(2)

with col1:
    # Filter by disposition status
    dispositions = st.multiselect(
        "Starea planetei",
        options=["CP", "PC"],
        default=["CP"],
        help="**CP:** Planetă confirmată. **PC:** Candidat la planetă. Poți selecta una sau ambele."
    )
    # Filter by Planet Radius
    radius_range = st.slider(
        "Planet Radius (Earth Radii)",
        min_value=0.1, max_value=25.0, value=(0.5, 4.0), step=0.1,
        help="Search for planets within a specific size range."
    )

with col2:
    # Filter by Host Star ID TIC
    tic_id = st.text_input(
        "Host Star ID TIC (optional)",
        help="Enter a specific ID TIC to see all candidates for that star."
    )
    # Filter by Orbital Period
    period_range = st.slider(
        "Perioadă orbitală (zile)",
        min_value=0.1, max_value=500.0, value=(0.5, 50.0), step=0.1,
        help="Caută planete într-un anumit interval de perioadă orbitală."
    )

# --- Search Execution ---
if 'toi_search_results' not in st.session_state:
    st.session_state.toi_search_results = None

if st.button("Caută în catalogul TESS", type="primary"):
    # Clean up ID TIC input if provided
    search_tic = None
    if tic_id and tic_id.strip().isdigit():
        search_tic = int(tic_id.strip())
        
    with st.spinner("Se caută în catalogul TESS TOI..."):
        results_df = search_toi_catalog(
            dispositions=dispositions,
            radius_range=radius_range,
            period_range=period_range,
            tic_id=search_tic
        )
        st.session_state.toi_search_results = results_df

# --- Display Results ---
if st.session_state.toi_search_results is not None:
    st.divider()
    results_df = st.session_state.toi_search_results
    
    if results_df.empty:
        st.warning("No TESS Objects of Interest (TOIs) found matching your criteria. Please broaden your search.")
    else:
        st.success(f"Found **{len(results_df)} TOIs** matching your criteria.")
        
        st.dataframe(
            results_df,
            column_config={
                "Planet Radius (R_earth)": st.column_config.NumberColumn(format="%.2f"),
                "Perioadă orbitală (zile)": st.column_config.NumberColumn(format="%.2f"),
                "Planet Temp (K)": st.column_config.NumberColumn(format="%d K"),
                "Stellar Radius (R_sun)": st.column_config.NumberColumn(format="%.2f R⊙"),
                "Stellar Teff (K)": st.column_config.NumberColumn(format="%d K"),
            },
            use_container_width=True,
            hide_index=True
        )
        
        st.info("Poți copia un „ID TIC” și îl poți folosi în pagina „Caută o stea” pentru o analiză completă.")
