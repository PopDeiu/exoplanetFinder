import streamlit as st
from utils.database import clear_all_naked_eye_stars, bulk_save_stars, get_stars_by_bortle_mock
from utils.ui_styles import set_galaxy_background, set_sidebar_style

st.set_page_config(page_title="Planetariu VR")
set_sidebar_style()
set_galaxy_background("stellar")
st.title("Planetariu VR")
st.markdown("Selectează nivelul de poluare luminoasă. Baza ta de date va fi actualizată doar cu stelele vizibile în aceste condiții.")

# Interfața minimalistă
bortle_scale = st.selectbox(
    "Selectează Scara Bortle din locația ta:",
    options=list(range(1, 10)),
    index=3, # Default la Bortle 4
    format_func=lambda x: f"Bortle {x} - {'Cer excelent' if x < 3 else 'Cer urban' if x > 6 else 'Cer mediu'}"
)

# Butonul magic
if st.button("Sincronizează cu Baza de Date", type="primary", use_container_width=True):
    with st.spinner(f"Aducem stelele pentru Bortle {bortle_scale}..."):
        
        # 1. Obținem stelele (aici folosești funcția mock, sau un API real pe viitor)
        stars_to_save = get_stars_by_bortle_mock(bortle_scale)
        
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