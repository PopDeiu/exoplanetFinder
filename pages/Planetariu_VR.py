import streamlit as st
from datetime import datetime
from utils.data_fetchers import get_stars_from_simbad
from utils.database import clear_all_naked_eye_stars, bulk_save_stars
from utils.ui_styles import set_galaxy_background, set_sidebar_style

st.set_page_config(page_title="Planetariu VR")
set_sidebar_style()
set_galaxy_background("stellar")

st.title("Planetariu VR")

# --- DATE PREDEFINITE ---
ORASE = {
    "Personalizat": (46.1866, 21.3123),
    "Arad": (46.1866, 21.3123),
    "București": (44.4268, 26.1025),
    "New York (NYC)": (40.7128, -74.0060),
    "Tokyo": (35.6762, 139.6503),
    "Londra": (51.5074, -0.1278),
}

# Initializare Session State pentru coordonate
if "lat" not in st.session_state:
    st.session_state.lat = 46.1866
if "lon" not in st.session_state:
    st.session_state.lon = 21.3123

def on_city_change():
    oras = st.session_state.city_selector
    if oras in ORASE and oras != "Personalizat":
        st.session_state.lat, st.session_state.lon = ORASE[oras]

# ========== FORMULARUL 1: POLUARE LUMINOASĂ ==========
with st.expander("1. Sincronizare după Poluare Luminoasă (Bortle)", expanded=True):
    st.markdown("Folosește acest formular pentru a filtra stelele doar în funcție de vizibilitatea oferită de scara Bortle.")
    
    bortle_scale = st.select_slider(
        "Selectează nivelul Bortle:",
        options=list(range(1, 10)),
        value=4,
        key="bortle_slider",
        format_func=lambda x: f"Bortle {x}"
    )

    if st.button("Sincronizează doar Bortle", type="primary", use_container_width=True):
        with st.spinner("Se actualizează baza de date..."):
            # Aici trimitem coordonatele default sau cele din state, dar focusul e pe Bortle
            stars = get_stars_from_simbad(bortle_scale, lat=st.session_state.lat, lon=st.session_state.lon)
            if stars:
                clear_all_naked_eye_stars()
                bulk_save_stars(stars)
                st.success(f"✅ S-au salvat {len(stars)} stele (Bortle {bortle_scale})")
                st.balloons()
            else:
                st.error("Nu s-au putut prelua datele.")

st.write("")

# ========== FORMULARUL 2: LOCALIZARE ȘI TIMP ==========
with st.expander("2. Sincronizare după Locație și Timp", expanded=True):
    st.markdown("Setează coordonatele geografice și momentul observației.")
    
    # Opțiune Timp
    use_current_time = st.checkbox("Folosește data și ora curentă", value=True, key="check_time")
    
    if not use_current_time:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            d = st.date_input("Data", datetime.now())
        with col_t2:
            t = st.time_input("Ora", datetime.now().time())
        final_time = datetime.combine(d, t)
    else:
        final_time = datetime.now()
        st.info(f"Ora curentă: {final_time.strftime('%H:%M:%S')}")

    # Opțiune Locație
    st.selectbox("Oraș preset:", options=list(ORASE.keys()), index=1, key="city_selector", on_change=on_city_change)

    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitudine (°N)", -90.0, 90.0, value=st.session_state.lat, key="lat_input")
    with col2:
        lon = st.number_input("Longitudine (°E)", -180.0, 180.0, value=st.session_state.lon, key="lon_input")

    if st.button("Sincronizează Locație & Timp", type="primary", use_container_width=True):
        with st.spinner("Se calculează stelele vizibile pentru locația selectată..."):
            # Folosim bortle_scale din primul formular sau o valoare default
            current_bortle = st.session_state.get("bortle_slider", 4)
            stars = get_stars_from_simbad(current_bortle, lat=lat, lon=lon, time=final_time)
            
            if stars:
                clear_all_naked_eye_stars()
                bulk_save_stars(stars)
                st.success(f"✅ Sincronizat pentru {lat}, {lon} la data de {final_time.strftime('%d/%m/%Y')}")
                st.balloons()
            else:
                st.warning("Nu s-au găsit stele vizibile pentru aceste setări.")