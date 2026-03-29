import streamlit as st
import pandas as pd
from utils.database import get_user_observations, delete_observation
from utils.ui_styles import set_galaxy_background, set_sidebar_style

st.set_page_config(page_title="Observațiile Mele", layout="wide")

set_sidebar_style()
set_galaxy_background("default")

st.title("Jurnalul meu Astronomic")

# Verificăm dacă utilizatorul este logat
if not st.session_state.get('logged_in', False):
    st.warning("Te rugăm să te autentifici din pagina principală pentru a vedea observațiile salvate.")
    st.stop()

user_id = st.session_state.user_info['ID']
observations = get_user_observations(user_id)

if not observations:
    st.info("Încă nu ai nicio observație salvată. Mergi la pagina de Căutare și salvează prima ta descoperire!")
else:
    # Convertim în Pandas DataFrame pentru o afișare frumoasă
    df = pd.DataFrame(observations)
    
    # Redenumim coloanele pentru tabel
    df.columns = ["ID", "Stea", "Perioadă (zile)", "Adâncime", "Rază (R_jup)", "Note", "Data Salvării"]

    # Afișăm tabelul
    st.subheader(f"Ai {len(observations)} observații salvate")
    
    # Folosim st.data_editor pentru a permite vizualizarea ușoară (fără editare directă în DB aici)
    st.dataframe(
        df, 
        use_container_width=True,
        column_config={
            "Data Salvării": st.column_config.DatetimeColumn(format="D MMM YYYY, HH:mm"),
            "ID": None # Ascundem coloana ID
        }
    )

    # --- Secțiune de Management (Ștergere) ---
    st.divider()
    with st.expander("Administrează observațiile"):
        obs_to_delete = st.selectbox(
            "Selectează ID-ul observației pe care vrei să o ștergi:",
            options=[obs['ID'] for obs in observations],
            format_func=lambda x: f"ID {x} - {next(item['star_id'] for item in observations if item['ID'] == x)}"
        )
        
        if st.button("Șterge definitiv", type="secondary"):
            if delete_observation(obs_to_delete, user_id):
                st.success(f"Observația {obs_to_delete} a fost ștearsă.")
                st.rerun()
            else:
                st.error("Nu s-a putut șterge observația.")