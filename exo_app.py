import streamlit as st
from utils.ui_styles import set_galaxy_background, set_sidebar_style
from utils.settings_manager import load_settings
from utils.database import verify_credentials  # Importăm logica de DB

# --- Configurare Pagină ---
st.set_page_config(
    page_title="Vânătorul de exoplanete AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Aplicare styling IMEDIAT ---
set_sidebar_style()
set_galaxy_background("default")

# --- Session State Initialization (Auth & Persistent Settings) ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

# Încarcă setările salvate din fișier (Settings Manager)
persisted_settings = load_settings()

# --- SIDEBAR: LOGIN FORM ---
with st.sidebar:
    st.title("🔐 Acces Sistem")
    
    if not st.session_state.logged_in:
        with st.form("sidebar_login"):
            st.subheader("Autentificare")
            user_input = st.text_input("Utilizator")
            pass_input = st.text_input("Parolă", type="password")
            submit_login = st.form_submit_button("Log In", use_container_width=True)
            
            if submit_login:
                user_data = verify_credentials(user_input, pass_input)
                if user_data:
                    st.session_state.logged_in = True
                    st.session_state.user_info = user_data
                    st.success(f"Bun venit, {user_input}!")
                    st.rerun()
                else:
                    st.error("Credentiale invalide")
    else:
        st.write(f"Utilizator: **{st.session_state.user_info['username']}**")
        if st.button("Log Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_info = None
            st.rerun()
    
    st.divider()
    st.success("Alege o pagină din meniu pentru a începe explorarea.")
    st.header("Despre aplicație")
    st.info(
        "„Vânătorul de exoplanete AI” folosește date de la misiunile spațiale TESS și Kepler, accesate prin pachetul `lightkurve` și diverse arhive astronomice (MAST, ExoFOP, TIC)."
    )

# --- PROTECȚIE CONȚINUT ---
if not st.session_state.logged_in:
    st.title("Exoplanet Hunter")
    st.warning("⚠️ Acces restricționat. Vă rugăm să folosiți formularul din bara laterală pentru a vă autentifica.")
    st.image("https://img.freepik.com/free-vector/access-control-system-abstract-concept_335657-3180.jpg", width=400)
    st.stop()  # Oprește execuția aici dacă nu este logat

# --- CODUL ORIGINAL (Executat doar după Login) ---

# Inițializare rezultate în Session State
if 'search_result' not in st.session_state:
    st.session_state.search_result = None
if 'explore_planets_results' not in st.session_state:
    st.session_state.explore_planets_results = None
if 'explore_fps_results' not in st.session_state:
    st.session_state.explore_fps_results = None
if 'untested_results' not in st.session_state:
    st.session_state.untested_results = None

# Initialize settings in session state
if 'selected_missions' not in st.session_state:
    st.session_state.selected_missions = persisted_settings['selected_missions']
if 'selected_authors' not in st.session_state:
    st.session_state.selected_authors = persisted_settings['selected_authors']
if 'bin_size' not in st.session_state:
    st.session_state.bin_size = persisted_settings['bin_size']
if 'sigma_val' not in st.session_state:
    st.session_state.sigma_val = persisted_settings['sigma_val']
if 'period_range' not in st.session_state:
    st.session_state.period_range = tuple(persisted_settings['period_range'])

# --- Main Page Content ---
st.title("Exoplanet Hunter")
st.caption("Aplicație interactivă pentru explorarea datelor TESS și Kepler și căutarea de exoplanete prin metoda tranzitului.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ce poți face aici")
    st.markdown(
        """
        - **Caută o stea** după nume sau ID (TIC / TOI / KIC) și descarcă curba ei de lumină.
        - **Explorează sisteme cu planete** – candidați și planete confirmate.
        - **Analizează false pozitive** – semnale care arată ca o planetă, dar sunt cauzate de alte fenomene.
        - **Găsește ținte netestate** care merită investigate în detaliu.
        - **Ajustează parametrii de analiză** (binning, sigma, interval de perioade) din pagina *Setări*.
        """
    )

    st.subheader("Cum funcționează metoda tranzitului")
    st.markdown(
        """
        Când o exoplanetă trece prin fața stelei sale, luminozitatea aparentă a stelei scade foarte puțin pentru
        o perioadă scurtă de timp. Dacă aceste scăderi se repetă periodic, ele pot indica prezența unei planete
        care orbitează steaua.
        """
    )

    st.subheader("Pornire rapidă")
    st.markdown(
        """
        1. Deschide pagina **„Caută o stea după nume sau ID”** din bara laterală.  
        2. Introdu un obiect cunoscut, de exemplu `Kepler-10` sau `TIC 261136679`.  
        3. Alege seturile de date (sectoare / misiuni) și apasă **„Procesează fișierele selectate”**.
        """
    )
    if st.button("Folosește un exemplu (Kepler-10)", type="primary"):
        st.session_state["quick_start_example"] = "Kepler-10"
        st.success("Exemplul a fost salvat. Deschide pagina „Caută o stea după nume sau ID” pentru a-l folosi.")

with col2:
    st.subheader("Sfaturi utile")
    st.info(
        """
        - Începe cu stele bine studiate (Kepler, TOI-uri populare) pentru a vedea cum arată semnalele reale.  
        - Dacă nu apare niciun rezultat, verifică pagina **Setări** – poate intervalul de perioade este prea îngust.  
        - Ține cont că aceasta este o unealtă *educațională* și *exploratorie*.
        """
    )
    st.subheader("Despre date")
    st.markdown(
        """
        Datele provin din arhive publice NASA (TESS, Kepler) accesate prin biblioteca `lightkurve` și servicii
        precum MAST și ExoFOP.
        """
    )