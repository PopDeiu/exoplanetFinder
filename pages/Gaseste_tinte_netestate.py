import streamlit as st
from utils import fetch_untested_targets, set_galaxy_background, set_sidebar_style
from utils.auth import init_auth, render_sidebar_auth
import pandas as pd
from PIL import Image
import os
import base64
from io import BytesIO

init_auth()

st.set_page_config(
    page_title="Ținte netestate",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Aplicare styling IMEDIAT ---
set_sidebar_style()
set_galaxy_background("stellar")

st.logo("assets/ExoLogo_noBg.png", size="large")

with st.sidebar:
    render_sidebar_auth()

st.header("Găsește ținte posibil netestate")
st.caption("Descoperă stele interesante care nu au încă planete candidate oficiale asociate.")
st.markdown("""
Acest instrument scanează catalogul **TESS Input Catalog (TIC)** pentru stele luminoase care **nu apar** în lista oficială de obiecte de interes (TOI). 
Este "terenul de vânătoare" perfect pentru a găsi tranzite pe care algoritmii automați le-au omis.
""")

# Inițializare session_state pentru rezultate și paginare
if 'untested_results' not in st.session_state:
    st.session_state.untested_results = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0
if 'items_per_page' not in st.session_state:
    st.session_state.items_per_page = 100

if st.button("Găsește stele netestate", type="primary", key="fetch_untested", use_container_width=True):
    with st.spinner("Se corelează cataloagele TESS și se filtrează TOI-urile cunoscute..."):
        # Această funcție din utils/data_fetchers.py acum exclude automat TOI-urile
        st.session_state.untested_results = fetch_untested_targets(num_to_sample=200)
        st.session_state.current_page = 0  # Reset la prima pagină

if st.session_state.untested_results is not None:
    df = st.session_state.untested_results
    if df.empty:
        st.error("Nu s‑au găsit ținte netestate în acest eșantion. Încearcă din nou.")
    else:
        st.subheader("Stele candidate pentru analiză")
        
        # --- CONTROALE DE PAGINARE ---
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            st.session_state.items_per_page = st.selectbox(
                "Elemente per pagină:",
                options=[10, 20, 50, 100],
                index=1,  # Default: 100
                key="items_select"
            )
        
        # Calcularea numărului de pagini
        total_items = len(df)
        total_pages = (total_items + st.session_state.items_per_page - 1) // st.session_state.items_per_page
        
        # Asigurare că pagina curentă este validă
        if st.session_state.current_page >= total_pages:
            st.session_state.current_page = total_pages - 1
        
        with col2:
            page_options = [f"Pagina {i+1}" for i in range(total_pages)]
            selected_page_text = st.selectbox(
                "Selectează pagina:",
                options=page_options,
                index=st.session_state.current_page,
                key="page_select"
            )
            st.session_state.current_page = page_options.index(selected_page_text)
        
        
        with col4:
            st.metric("Total stele", total_items)
        
        st.markdown(f"**Pagina {st.session_state.current_page + 1} din {total_pages}**")
        
        # Calcularea indexurilor pentru pagina curentă
        start_idx = st.session_state.current_page * st.session_state.items_per_page
        end_idx = start_idx + st.session_state.items_per_page
        df_paginated = df.iloc[start_idx:end_idx]
        
        # Ajustăm column_config pentru a se potrivi cu datele returnate de fetch_untested_targets
        st.dataframe(
            df_paginated,
            column_config={
                "Searchable ID": st.column_config.TextColumn("ID Căutare", help="Copiază acest ID (ex: TIC 12345)"),
                "Tmag": st.column_config.NumberColumn(
                    "Magnitudine TESS", 
                    format="%.2f",
                    help="Strălucirea stelei. Numerele mici (sub 10) sunt stele foarte strălucitoare."
                ),
                "rad": st.column_config.NumberColumn("Rază Stea (R⊙)", format="%.2f"),
                "mass": st.column_config.NumberColumn("Masă Stea (M⊙)", format="%.2f"),
                "ra": st.column_config.NumberColumn("Ascensie Dreaptă", format="%.4f"),
                "dec": st.column_config.NumberColumn("Declinație", format="%.4f"),
            },
            hide_index=True,
            use_container_width=True
        )
        
        st.success("✅ Sfat: Copiază un ID din prima coloană și mergi la pagina 'Caută o stea'!")

        # --- GHID VIZUAL PENTRU DECOPERIRE ---
        st.divider()
        st.subheader("📖 Cum identifici o planetă nouă?")
        g1, g2 = st.columns(2)
        
        with g1:
            st.info("**✅ Tranzit Real (Căutăm asta):**")
            st.markdown("""
            - **Pasul 2:** Un vârf (peak) foarte clar și înalt în periodogramă.
            - **Pasul 3:** Punctele albastre formează o formă de **'U'** sau **'V'** clară.
            - **Modelul:** Linia roșie se suprapune bine peste puncte.
            """)
            
        with g2:
            st.warning("**❌ Zgomot / Eroare (Ignorăm asta):**")
            st.markdown("""
            - **Pasul 2:** Multe vârfuri mici de aceeași înălțime (haos).
            - **Pasul 3:** Punctele albastre sunt împrăștiate peste tot fără o formă clară.
            - **Modelul:** Linia roșie trece prin zone goale.
            """)

# --- Mesaj de încurajare ---
if st.session_state.untested_results is None:
    st.info("Apasă butonul de mai sus pentru a genera o listă de stele proaspete din baza de date MAST.")
