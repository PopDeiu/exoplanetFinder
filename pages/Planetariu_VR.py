import streamlit as st
from utils.data_fetchers import get_stars_from_simbad
from utils.database import clear_all_naked_eye_stars, bulk_save_stars, get_real_stars_by_bortle
from utils.ui_styles import set_galaxy_background, set_sidebar_style

st.set_page_config(page_title="Planetariu VR")
set_sidebar_style()
set_galaxy_background("stellar")

st.title("Planetariu VR")
st.markdown("Selectează nivelul de poluare luminoasă și locația. Baza ta de date va fi actualizată doar cu stelele vizibile în aceste condiții.")

# --- SETĂRI LOCAȚIE ȘI CER ---
st.subheader("Setări Observație")
st.markdown("Default, locația este setată la Arad, România. Poți ajusta latitudinea și longitudinea pentru a obține stele relevante pentru zona ta.")

# Coloane pentru Latitudine și Longitudine
col1, col2 = st.columns(2)
with col1:
    # Coordonate implicite pentru Arad, România
    latitudine = st.number_input("Latitudine (°N)", min_value=-90.0, max_value=90.0, value=46.1866, step=0.1, format="%.4f")
with col2:
    longitudine = st.number_input("Longitudine (°E)", min_value=-180.0, max_value=180.0, value=21.3123, step=0.1, format="%.4f")

# Interfața pentru Bortle
bortle_scale = st.selectbox(
    "Selectează Scara Bortle din locația ta:",
    options=list(range(1, 10)),
    index=3, # Default la Bortle 4
    format_func=lambda x: f"Bortle {x} - {'Cer excelent' if x < 3 else 'Cer urban' if x > 6 else 'Cer mediu'}"
)

st.divider()

# --- BUTONUL DE SINCRONIZARE ---
if st.button("Sincronizează cu Baza de Date", type="primary", use_container_width=True):
    with st.spinner(f"Aducem stelele pentru Bortle {bortle_scale} din locația selectată..."):
        
        # 1. Obținem stelele (am adăugat lat și lon ca parametri)
        # Atenție: Va trebui să actualizăm funcția în utils.py ca să folosească aceste coordonate!
        #stars_to_save = get_real_stars_by_bortle(bortle_scale, lat=latitudine, lon=longitudine)
        stars_to_save = get_stars_from_simbad(bortle_scale, lat=latitudine, lon=longitudine)
        
        if not stars_to_save:
            st.warning("Nu s-au găsit stele sau a apărut o problemă la descărcare.")
        else:
            # 2. Ștergem datele vechi din DB
            clear_success = clear_all_naked_eye_stars()
            
            if clear_success:
                # 3. Salvăm noile date
                insert_success = bulk_save_stars(stars_to_save)
                
                if insert_success:
                    st.success(f"✅ Succes! Am șters datele vechi și am salvat {len(stars_to_save)} stele noi pentru Bortle {bortle_scale}.")
                    st.balloons()
                else:
                    st.error("Eroare la salvarea noilor stele în baza de date.")
            else:
                st.error("Eroare la ștergerea stelelor vechi. Sincronizarea a fost oprită.")