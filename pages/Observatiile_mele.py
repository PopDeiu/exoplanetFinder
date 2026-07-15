import streamlit as st
import pandas as pd
from utils.database import get_user_observations, delete_observation, update_observation_notes
from utils.ui_styles import set_galaxy_background, set_sidebar_style
from utils.auth import init_auth, render_sidebar_auth

init_auth()

st.set_page_config(page_title="Observațiile Mele", layout="wide")

set_sidebar_style()
set_galaxy_background("default")

st.logo("assets/ExoLogo_noBg.png", size="large")

with st.sidebar:
    render_sidebar_auth()

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
        obs_id_selected = st.selectbox(
            "Selectează ID-ul observației pentru modificări:",
            options=[obs['ID'] for obs in observations],
            format_func=lambda x: f"ID {x} - {next(item['star_id'] for item in observations if item['ID'] == x)}"
        )
        
        # Preluăm nota actuală pentru a o pre-completa în editor
        current_note = next(item['observations'] for item in observations if item['ID'] == obs_id_selected)

        # Creăm două coloane pentru butoane
        col_edit, col_del = st.columns([1, 1])

        with col_edit:
            # Folosim popover pentru a nu aglomera interfața
            with st.popover("📝 Editează Notele", use_container_width=True):
                st.write(f"Modifică observațiile pentru ID {obs_id_selected}")
                new_notes = st.text_area("Note noi:", value=current_note)
                if st.button("Salvează Modificările", type="primary"):
                    if update_observation_notes(obs_id_selected, user_id, new_notes):
                        st.success("Observație actualizată!")
                        st.rerun()
                    else:
                        st.error("Eroare la actualizare.")

        with col_del:
            if st.button("🗑️ Șterge definitiv", type="secondary", use_container_width=True):
                if delete_observation(obs_id_selected, user_id):
                    st.success(f"Observația {obs_id_selected} a fost ștearsă.")
                    st.rerun()
                else:
                    st.error("Nu s-a putut șterge.")