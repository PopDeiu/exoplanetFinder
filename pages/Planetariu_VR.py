import streamlit as st
from datetime import datetime
import pytz
from utils.data_fetchers import get_stars_from_simbad
from utils.database import clear_all_naked_eye_stars, bulk_save_stars, get_all_settings, update_app_setting
from utils.ui_styles import set_galaxy_background, set_sidebar_style

# Configurare zonă orară România
RO_TZ = pytz.timezone('Europe/Bucharest')
nume_oras = st.session_state.city_selector if "city_selector" in st.session_state else "Arad"
# --- FUNCȚIE ÎNCĂRCARE SETĂRI ---
def load_settings_into_session():
    """Citește din DB și pune în session_state dacă nu există deja"""
    db_settings = get_all_settings()
    
    if db_settings:
        # Mapăm valorile din DB în session_state
        if "lat" not in st.session_state:
            st.session_state.lat = float(db_settings.get("latitudine", 46.1866))
        if "lon" not in st.session_state:
            st.session_state.lon = float(db_settings.get("longitudine", 21.3123))
        if "viteza_slider" not in st.session_state:
            st.session_state.viteza_slider = int(db_settings.get("viteza", 0))
        if "city_selector" not in st.session_state:
            st.session_state.city_selector = db_settings.get("oras", "Arad")
        if "bortle_slider" not in st.session_state:
            st.session_state.bortle_slider = int(db_settings.get("bortle", 4))

# Apelăm funcția de încărcare imediat după importuri
load_settings_into_session()

st.set_page_config(page_title="Planetariu VR")

# --- ORASE PREDEFINITE (Fără "Personalizat") ---
ORASE = {
    "Arad": (46.1866, 21.3123),
    "București": (44.4268, 26.1025),
    "New York (NYC)": (40.7128, -74.0060),
    "Tokyo": (35.6762, 139.6503),
    "Londra": (51.5074, -0.1278),
}

# Initializare Session State
if "lat" not in st.session_state:
    st.session_state.lat = 46.1866
if "lon" not in st.session_state:
    st.session_state.lon = 21.3123

def on_city_change():
    """Actualizează coordonatele instant când orașul este schimbat"""
    oras = st.session_state.city_selector
    if oras in ORASE:
        st.session_state.lat = ORASE[oras][0]
        st.session_state.lon = ORASE[oras][1]

# ========== FORMULARUL 1: BORTLE ==========
with st.expander("1. Sincronizare după Poluare Luminoasă (Bortle)", expanded=True):
    bortle_scale = st.select_slider(
        "Selectează nivelul Bortle:",
        options=list(range(1, 10)),
        key="bortle_slider" # Va prelua automat valoarea din session_state încărcată din DB
    )
    # Când se apasă butonul, salvăm și bortle în DB
    if st.button("Sincronizează doar Bortle", type="primary", use_container_width=True):
        with st.spinner("Se actualizează..."):
            now_ro = datetime.now(RO_TZ)
            stars = get_stars_from_simbad(bortle_scale, lat=st.session_state.lat, lon=st.session_state.lon, time=now_ro)
            if stars:
                clear_all_naked_eye_stars()
                bulk_save_stars(stars)
                st.success(f"✅ S-au salvat {len(stars)} stele.")
                st.balloons()
        update_app_setting("bortle", bortle_scale)

# ========== FORMULARUL 2: LOCAȚIE ȘI TIMP ==========
with st.expander("2. Sincronizare după Locație și Timp", expanded=True):
    now_ro = datetime.now(RO_TZ)
    
    # Cheia "check_time" previne resetarea stării checkbox-ului
    use_current_time = st.checkbox("Folosește data și ora curentă (România)", value=True, key="use_current_time_check")
    
    if not use_current_time:
        col_t1, col_t2 = st.columns(2)
        # Adăugarea cheilor "manual_date" și "manual_time" permite modificarea lunii/anului fără blocaje
        d = col_t1.date_input("Data", value=now_ro.date(), key="manual_date")
        t = col_t2.time_input("Ora", value=now_ro.time(), key="manual_time")
        final_time = RO_TZ.localize(datetime.combine(d, t))
    else:
        final_time = now_ro
        st.info(f"Ora curentă (RO): {final_time.strftime('%H:%M:%S')}")

    viteza_simulare = st.slider(
        "Viteza simulării:",
        -100, 100, 
        key="viteza_slider" 
    )

    st.selectbox(
        "Alege un oraș preset:", 
        options=list(ORASE.keys()), 
        key="city_selector", 
        on_change=on_city_change
    )

    col1, col2 = st.columns(2)
    # Folosim direct value=st.session_state.lat/lon
    lat_val = col1.number_input("Latitudine (°N)", -90.0, 90.0, key="lat", format="%.4f")
    lon_val = col2.number_input("Longitudine (°E)", -180.0, 180.0, key="lon", format="%.4f")


    if st.button("Sincronizează Locație & Timp", type="primary", use_container_width=True):
        with st.spinner("Sincronizare în curs..."):
            current_bortle = st.session_state.get("bortle_slider", 4)
            stars = get_stars_from_simbad(current_bortle, lat=lat_val, lon=lon_val, time=final_time)
            
            if stars:
                clear_all_naked_eye_stars()
                bulk_save_stars(stars)
                
                # --- AICI FACEM MODIFICAREA ÎN DB ---
                from utils.database import update_app_setting
                
                update_app_setting("latitudine", lat_val)
                update_app_setting("longitudine", lon_val)
                update_app_setting("oras", nume_oras)
                update_app_setting("viteza", viteza_simulare)
                update_app_setting("foloseste_data_curenta", "da" if use_current_time else "nu")
                update_app_setting("data_si_ora_obs", final_time.strftime("%Y-%m-%d %H:%M:%S"))
                # ------------------------------------

                st.success(f"✅ Date salvate în DB pentru {lat_val}, {lon_val}")
                st.balloons()
    