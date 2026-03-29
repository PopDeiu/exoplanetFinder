import streamlit as st
from utils.ui_styles import set_galaxy_background, set_sidebar_style
from utils.settings_manager import load_settings
from utils.database import verify_credentials, register_user # Funcția creată anterior în utils

st.set_page_config(
    page_title="Vânătorul de exoplanete AI",
    layout="wide",
    initial_sidebar_state="expanded"
)
def set_custom_input_style():
    st.markdown("""
        <style>
        /* Schimbă border-ul la focus (când dai click în el) */
        div[data-baseweb="input"] > div:focus-within {
            border-color: #9b59b6 !important;
            box-shadow: 0 0 0 2px rgba(155, 89, 182, 0.5) !important;
        }
        /* Schimbă border-ul implicit pentru a fi mai vizibil */
        div[data-baseweb="input"] {
            border: 1px solid #9b59b6 !important;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
# --- Aplicare styling IMEDIAT ---
set_sidebar_style()
set_galaxy_background("default")
set_custom_input_style()

# --- Session State Initialization ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

if 'search_result' not in st.session_state:
    st.session_state.search_result = None
if 'explore_planets_results' not in st.session_state:
    st.session_state.explore_planets_results = None
if 'explore_fps_results' not in st.session_state:
    st.session_state.explore_fps_results = None
if 'untested_results' not in st.session_state:
    st.session_state.untested_results = None

# Încarcă setările salvate din fișier
persisted_settings = load_settings()

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

# --- Sidebar Content ---
with st.sidebar:
    if not st.session_state.logged_in:
        # Alegem ce vrem să facem: Login sau Cont Nou
        menu = st.radio("Navigare Cont", ["Login", "Înregistrare"], horizontal=True)
        
        if menu == "Login":
            with st.form("login_form"):
                st.subheader("Autentificare")
                user_in = st.text_input("Username")
                pass_in = st.text_input("Password", type="password")
                if st.form_submit_button("Log In", use_container_width=True):
                    user_data = verify_credentials(user_in, pass_in)
                    if user_data:
                        st.session_state.logged_in = True
                        st.session_state.user_info = user_data
                        st.success(f"Salut, {user_in}!")
                        st.rerun()
                    else:
                        st.error("Credentiale incorecte")
        
        else: # Secțiunea de Înregistrare
            with st.form("register_form"):
                st.subheader("Creare Cont Nou")
                new_user = st.text_input("Alege Username")
                new_pass = st.text_input("Alege Parolă", type="password")
                confirm_pass = st.text_input("Confirmă Parolă", type="password")
                
                if st.form_submit_button("Înregistrează-te", use_container_width=True):
                    if new_pass != confirm_pass:
                        st.error("Parolele nu coincid!")
                    elif len(new_user) < 3:
                        st.error("Username-ul este prea scurt!")
                    else:
                        success, message = register_user(new_user, new_pass)
                        if success:
                            st.success(message)
                            st.info("Acum te poți loga din meniul de Login.")
                        else:
                            st.error(message)
    else:
        # Afișare când este logat
        st.write(f"✅ Logat ca: **{st.session_state.user_info['username']}**")
        if st.button("Log out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_info = None
            st.rerun()

    st.sidebar.success("Alege o pagină din meniu pentru a începe explorarea.")
    st.sidebar.divider()
    st.sidebar.header("Despre aplicație")
    st.sidebar.info(
        "„Vânătorul de exoplanete AI” folosește date de la misiunile spațiale TESS și Kepler, accesate prin pachetul `lightkurve` și diverse arhive astronomice (MAST, ExoFOP, TIC). Aplicația descarcă și curăță curbele de lumină ale stelelor și aplică algoritmul Box Least Squares (BLS) pentru a identifica potențiale tranzite planetare. Rezultatele sunt prezentate într-o formă interactivă, ușor de explorat."
    )

# --- Main Page Content (CONȚINUTUL TĂU ORIGINAL) ---
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

        Această aplicație descarcă curbele de lumină, le curăță și aplică un algoritm de tip **Box Least Squares (BLS)**
        pentru a găsi astfel de scăderi periodice.
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
        - Ține cont că aceasta este o unealtă *educațională* și *exploratorie*, nu un pipeline oficial de confirmare a planetelor.
        """
    )
    st.subheader("Despre date")
    st.markdown(
        """
        Datele provin din arhive publice NASA (TESS, Kepler) accesate prin biblioteca `lightkurve` și servicii
        precum MAST și ExoFOP. Ai nevoie de conexiune la internet pentru a rula interogările.
        """
    )