import streamlit as st
from utils.data_fetchers import get_stars_from_simbad
from utils.database import clear_all_naked_eye_stars, bulk_save_stars, get_real_stars_by_bortle
from utils.ui_styles import set_galaxy_background, set_sidebar_style

st.set_page_config(page_title="Planetariu VR")
set_sidebar_style()
set_galaxy_background("stellar")

st.title("Planetariu VR")
st.markdown("Selectează nivelul de poluare luminoasă și locația. Baza ta de date va fi actualizată doar cu stelele vizibile în aceste condiții.")

# --- ORASE PREDEFINITE ---
ORASE = {
    "Personalizat": (46.1866, 21.3123),
    "New York (NYC)": (40.7128, -74.0060),
    "Tokyo": (35.6762, 139.6503),
    "Londra": (51.5074, -0.1278),
    "Paris": (48.8566, 2.3522),
    "Sydney": (-33.8688, 151.2093),
}

# Initializează session state pentru widget-uri
if "lat_1" not in st.session_state:
    st.session_state.lat_1 = 46.1866
if "lon_1" not in st.session_state:
    st.session_state.lon_1 = 21.3123
if "lat_2" not in st.session_state:
    st.session_state.lat_2 = 46.1866
if "lon_2" not in st.session_state:
    st.session_state.lon_2 = 21.3123

def on_city_change_1():
    oras = st.session_state.city_selector_1
    if oras in ORASE and oras != "Personalizat":
        st.session_state.lat_1, st.session_state.lon_1 = ORASE[oras]

def on_city_change_2():
    oras = st.session_state.city_selector_2
    if oras in ORASE and oras != "Personalizat":
        st.session_state.lat_2, st.session_state.lon_2 = ORASE[oras]

# ========== FORMULARUL 1 ==========
st.subheader("Setări Observație")
st.markdown("Default, locația este setată la Arad, România. Poți ajusta latitudinea și longitudinea pentru a obține stele relevante pentru zona ta.")

oras_selectat_1 = st.selectbox(
    "Alege un oraș presetat:",
    options=list(ORASE.keys()),
    index=0,
    key="city_selector_1",
    on_change=on_city_change_1,
)

col1, col2 = st.columns(2)
with col1:
    latitudine = st.number_input("Latitudine (°N)", min_value=-90.0, max_value=90.0,
        value=st.session_state.lat_1, step=0.1, format="%.4f", key="lat_1")
with col2:
    longitudine = st.number_input("Longitudine (°E)", min_value=-180.0, max_value=180.0,
        value=st.session_state.lon_1, step=0.1, format="%.4f", key="lon_1")

bortle_scale = st.selectbox(
    "Selectează Scara Bortle din locația ta:",
    options=list(range(1, 10)),
    index=3,
    format_func=lambda x: f"Bortle {x} - {'Cer excelent' if x < 3 else 'Cer urban' if x > 6 else 'Cer mediu'}"
)

st.divider()

if st.button("Sincronizează cu Baza de Date", type="primary", use_container_width=True):
    with st.spinner(f"Aducem stelele pentru Bortle {bortle_scale} din locația selectată..."):
        
        # 1. Obținem stelele (am adăugat lat și lon ca parametri)
        # Atenție: Va trebui să actualizăm funcția în utils.py ca să folosească aceste coordonate!
        #stars_to_save = get_real_stars_by_bortle(bortle_scale, lat=latitudine, lon=longitudine)
        stars_to_save = get_stars_from_simbad(bortle_scale, lat=latitudine, lon=longitudine)
        
        if not stars_to_save:
            st.warning("Nu s-au găsit stele sau a apărut o problemă la descărcare.")
        else:
            clear_success = clear_all_naked_eye_stars()

            if clear_success:
                insert_success = bulk_save_stars(stars_to_save)

                if insert_success:
                    st.success(f"✅ Succes! Am șters datele vechi și am salvat {len(stars_to_save)} stele noi pentru Bortle {bortle_scale}.")
                    st.balloons()
                else:
                    st.error("Eroare la salvarea noilor stele în baza de date.")
            else:
                st.error("Eroare la ștergerea stelelor vechi. Sincronizarea a fost oprită.")

# ========== FORMULARUL 2 (derulant) ==========
st.markdown("---")
st.subheader("Setări Observație #2")
st.markdown("Un al doilea formular pentru a compara setări diferite.")

oras_selectat_2 = st.selectbox(
    "Alege un oraș presetat:",
    options=list(ORASE.keys()),
    index=0,
    key="city_selector_2",
    on_change=on_city_change_2,
)

col3, col4 = st.columns(2)
with col3:
    latitudine2 = st.number_input("Latitudine (°N)", min_value=-90.0, max_value=90.0,
        value=st.session_state.lat_2, step=0.1, format="%.4f", key="lat_2")
with col4:
    longitudine2 = st.number_input("Longitudine (°E)", min_value=-180.0, max_value=180.0,
        value=st.session_state.lon_2, step=0.1, format="%.4f", key="lon_2")

bortle_scale2 = st.selectbox(
    "Selectează Scara Bortle din locația ta:",
    options=list(range(1, 10)),
    index=3,
    format_func=lambda x: f"Bortle {x} - {'Cer excelent' if x < 3 else 'Cer urban' if x > 6 else 'Cer mediu'}",
    key="bortle_2"
)

st.divider()

if st.button("Sincronizează cu Baza de Date #2", type="primary", use_container_width=True, key="sync_2"):
    with st.spinner(f"Aducem stelele pentru Bortle {bortle_scale2} din locația selectată..."):
        stars_to_save = get_stars_from_simbad(bortle_scale2, lat=latitudine2, lon=longitudine2)

        if not stars_to_save:
            st.warning("Nu s-au găsit stele sau a apărut o problemă la descărcare.")
        else:
            clear_success = clear_all_naked_eye_stars()

            if clear_success:
                insert_success = bulk_save_stars(stars_to_save)

                if insert_success:
                    st.success(f"✅ Succes! Am șters datele vechi și am salvat {len(stars_to_save)} stele noi pentru Bortle {bortle_scale2}.")
                    st.balloons()
                else:
                    st.error("Eroare la salvarea noilor stele în baza de date.")
            else:
                st.error("Eroare la ștergerea stelelor vechi. Sincronizarea a fost oprită.")
