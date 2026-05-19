import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
from utils.ui_styles import set_galaxy_background, set_sidebar_style
from utils.settings_manager import load_settings
from utils.database import verify_credentials, register_user, get_user_by_id
import os

# --- Gestionare Cookie-uri ---
# Folosește o parolă sigură pentru criptarea cookie-urilor
cookies = EncryptedCookieManager(
    password=os.getenv("COOKIE_PASSWORD", "parola-secreta-exoplanet-2026"),
)

if not cookies.ready():
    st.stop() # Așteptăm încărcarea cookie-urilor din browser

st.set_page_config(
    page_title="Vânătorul de exoplanete AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Aplicare styling ---
set_sidebar_style()
set_galaxy_background("default")

# --- Logică Auto-Login (Refresh Persistence) ---
if 'logged_in' not in st.session_state:
    saved_user_id = cookies.get('user_id')
    if saved_user_id:
        user_data = get_user_by_id(saved_user_id)
        if user_data:
            st.session_state.logged_in = True
            st.session_state.user_info = user_data
        else:
            st.session_state.logged_in = False
    else:
        st.session_state.logged_in = False

# --- Sidebar Content (Codul tău intact + Remember Me) ---
with st.sidebar:
    if not st.session_state.logged_in:
        # Alegem ce vrem să facem: Login sau Cont Nou
        menu = st.radio("Navigare Cont", ["Login", "Înregistrare"], horizontal=True)
        
        if menu == "Login":
            with st.form("login_form"):
                st.subheader("Autentificare")
                user_in = st.text_input("Username")
                pass_in = st.text_input("Password", type="password")
                remember_me = st.checkbox("Ține-mă minte (Rămâi logat)")
                
                if st.form_submit_button("Log In", use_container_width=True):
                    user_data = verify_credentials(user_in, pass_in)
                    if user_data:
                        st.session_state.logged_in = True
                        st.session_state.user_info = user_data
                        
                        if remember_me:
                            cookies['user_id'] = str(user_data['ID'])
                            cookies.save() # Salvează ID-ul în browser
                        
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
        # Folosesc 'username' în loc de 'user' deoarece așa este de obicei în DB
        username_display = st.session_state.user_info.get('user', 'Utilizator')
        st.write(f"✅ Logat ca: **{username_display}**")
        if st.button("Deconectare", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_info = None
            if 'user_id' in cookies:
                del cookies['user_id']
                cookies.save()
            st.rerun()

    st.sidebar.success("Alege o pagină din meniu pentru a începe explorarea.")
    st.sidebar.divider()
    st.sidebar.header("Despre aplicație")
    st.sidebar.info(
        "„Vânătorul de exoplanete AI” folosește date de la misiunile spațiale TESS și Kepler..."
    )

# --- Main Page Content (CONȚINUTUL TĂU ORIGINAL) ---
st.title("Exoplanethunter")
st.subheader("Echipa Exohunt")
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

    st.image("assets/transit_method.gif", caption="Animația metodei tranzitului – planeta întunecă parțial steaua, generând o scădere în curba de lumină.", use_container_width=True)

    st.subheader("METODA TRANZITULUI")
    st.markdown(
            """
            1. Preprocesarea curbelor de lumină
            2. Implementarea algoritmilor de detecție  
            3. Modelarea fizică a tranzitului 
            4. Analiza statistică și validarea
            """
        )

    st.subheader("ETAPE")
    st.markdown(
        """
        1. Selectia datelor observaționale 
        2. Preprocesarea datelor și reducerea zgomotului (Data Cleaning)  
        3. Detectarea periodicității (Căutarea tranzitului)
        4. “Împăturirea” curbei de lumină (Phase Folding)
        """
    )
    st.subheader("Preprocesarea datelor și reducerea zgomotului (Data Cleaning)")
    st.markdown(
            """
            - Eliminarea valorilor neimportante : date eronate sau erori de senzor.
            - Aplatizarea curbei : eliminarea petelor stelare, rotația stelei, etc
            
            """
        )
    st.subheader("Detectarea periodicității (Căutarea tranzitului)")
    st.markdown(
        """
         - Algoritmul BLS (Box Least Squares) – grupeaza datele la un interval de timp si face o medie .
         - Periodograma - Un grafic care arată puterea semnalului si cauta varful cel mai mare ce indică cei mai probabili candidați .
        """
    )
    st.subheader("“Împăturirea” curbei de lumină (Phase Folding)")
    st.markdown(
        """
         - Suprapunerea tuturor tranzitelor detectate
        """
    )
    st.image("assets/phase_folding.gif", caption="Phase folding – suprapunerea datelor de tranzit pentru a evidenția periodicitatea.", use_container_width=True)
    
    if st.button("Folosește un exemplu (Kepler-10)", type="primary"):
        st.session_state["quick_start_example"] = "Kepler-10"
        st.success("Exemplul a fost salvat. Deschide pagina „Caută o stea după nume sau ID” pentru a-l folosi.")

with col2:
    st.subheader("Despre date")
    st.markdown(
        """
        Datele provin din arhive publice NASA (TESS, Kepler) accesate prin biblioteca `lightkurve` și servicii
        precum MAST și ExoFOP. Ai nevoie de conexiune la internet pentru a rula interogările.
        """
    )
    st.subheader("Tehnologii utilizate")
    st.markdown(
            """
            - Biblioteca Python Streamlit;
            - Biblioteca Python Lightkurve;
            - Biblioteca Python Astroquery.
            - Unity / C#
            - Meta Quest 3
            - FastAPI (Python)
            - Docker pentru containerizare
            - Server Linux pentru hosting
            - Baze de date SQL (MySQL)
                - Sisteme de autentificare și gestionare a utilizatorilor
                - Stocare securizată a datelor și cookie-uri criptate
            """
        )
    