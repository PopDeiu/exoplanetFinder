import streamlit as st
from utils.ui_styles import set_galaxy_background, set_sidebar_style
from utils.settings_manager import load_settings
from utils.auth import init_auth, render_sidebar_auth

init_auth()

st.set_page_config(
    page_title="Vânătorul de exoplanete AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Aplicare styling ---
set_sidebar_style()
set_galaxy_background("default")
st.logo("assets/ExoLogo_noBg.png", size="large")

# --- Sidebar Content ---
with st.sidebar:
    render_sidebar_auth()

    st.success("Alege o pagină din meniu pentru a începe explorarea.")
    st.divider()
    st.header("Despre aplicație")
    st.info(
        "„Vânătorul de exoplanete AI” folosește date de la misiunile spațiale TESS și Kepler..."
    )

# --- Main Page Content (CONȚINUTUL TĂU ORIGINAL) ---
st.title("Exoplanet Hunter (ExoHunt)")
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
    st.image("assets/Etapa 3.jpeg", use_container_width=True)
    st.subheader("Detectarea periodicității (Căutarea tranzitului)")
    st.markdown(
        """
         - Algoritmul BLS (Box Least Squares) – grupeaza datele la un interval de timp si face o medie .
         - Periodograma - Un grafic care arată puterea semnalului si cauta varful cel mai mare ce indică cei mai probabili candidați .
        """
    )
    st.image("assets/periodograma neagra.jpg.jpeg", use_container_width=True)
    st.subheader("“Împăturirea” curbei de lumină (Phase Folding)")
    st.markdown(
        """
         - Suprapunerea tuturor tranzitelor detectate
        """
    )
    st.image("assets/phase_folding.gif", use_container_width=True)
    
    

with col2:
    st.image("assets/Banner_ExoHunt.png", use_container_width=True)
    st.subheader("Despre date")
    st.markdown(
        """
        Datele provin din arhive publice NASA (TESS, Kepler) accesate prin biblioteca `lightkurve` și servicii
        precum MAST și ExoFOP. Ai nevoie de conexiune la internet pentru a rula interogările
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
    