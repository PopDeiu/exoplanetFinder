import streamlit as st
from utils.ui_styles import set_galaxy_background, set_sidebar_style
from utils.settings_manager import load_settings
from utils.database import verify_credentials 

# --- Configurare Pagină ---
st.set_page_config(
    page_title="Vânătorul de exoplanete AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Aplicare styling ---
set_sidebar_style()
set_galaxy_background("default")

# --- Session State Initialization ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

# --- LOGICĂ LOGIN OPȚIONALĂ ÎN SIDEBAR ---
with st.sidebar:
    if not st.session_state.logged_in:
        with st.expander("🔐 Login (Opțional pentru salvare setări)"):
            with st.form("sidebar_login"):
                user_input = st.text_input("Utilizator")
                pass_input = st.text_input("Parolă", type="password")
                if st.form_submit_button("Log In", use_container_width=True):
                    user_data = verify_credentials(user_input, pass_input)
                    if user_data:
                        st.session_state.logged_in = True
                        st.session_state.user_info = user_data
                        st.success(f"Salut, {user_input}!")
                        st.rerun()
                    else:
                        st.error("Credentiale invalide")
    else:
        st.write(f"Conectat ca: **{st.session_state.user_info['username']}**")
        if st.button("Log Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_info = None
            st.rerun()
    
    st.divider()
    st.success("Alege o pagină din meniu.")

# --- ÎNCĂRCARE SETĂRI (Persistent sau Default) ---
# Dacă e logat, ai putea încărca setări specifice din DB aici. 
# Momentan folosim managerul tău de setări existent.
persisted_settings = load_settings()

# Inițializare variabile de sesiune (FĂRĂ st.stop())
if 'search_result' not in st.session_state:
    st.session_state.search_result = None
# ... restul inițializărilor tale ...
if 'selected_missions' not in st.session_state:
    st.session_state.selected_missions = persisted_settings['selected_missions']
if 'period_range' not in st.session_state:
    st.session_state.period_range = tuple(persisted_settings['period_range'])

# --- Main Page Content (Vizibil pentru TOȚI) ---
st.title("Exoplanet Hunter")

if st.session_state.logged_in:
    st.info(f"Sesiune activă pentru {st.session_state.user_info['username']}. Căutările tale pot fi salvate.")
else:
    st.caption("Mod Vizitator: Te poți loga din sidebar pentru a-ți salva preferințele.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ce poți face aici")
    st.markdown("- **Caută o stea** după nume sau ID...")
    # ... restul conținutului tău ...

with col2:
    st.subheader("Sfaturi utile")
    st.info("Începe cu stele bine studiate (Kepler, TOI-uri populare).")