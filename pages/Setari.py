import streamlit as st
from utils import set_galaxy_background, set_sidebar_style, init_session_state, save_settings
from utils.auth import init_auth, render_sidebar_auth

init_auth()

st.set_page_config(
    page_title="Setări Analiză",
    layout="wide",
    initial_sidebar_state="expanded"
)

set_sidebar_style()
set_galaxy_background("nebula")
st.logo("assets/ExoLogo_noBg.png", size="large")

# --- Inițializare session state din fișierul persistent ---
init_session_state()

# Funcție pentru salvarea automată
def save_all_settings():
    """Salvează toate setările în fișier persistent."""
    save_settings({
        "bin_size": st.session_state.bin_size,
        "sigma_val": st.session_state.sigma_val,
        "period_range": list(st.session_state.period_range),
        "selected_missions": st.session_state.selected_missions,
        "selected_authors": st.session_state.selected_authors
    })

# --- LOGICA DE RESET (Reparată: Ștergem și re-inițializăm) ---
with st.sidebar:
    render_sidebar_auth()
    if st.button("🔄 Reset la valori recomandate"):
        for key in ['bin_size', 'sigma_val', 'period_range', 'selected_missions', 'selected_authors']:
            if key in st.session_state:
                del st.session_state[key]
        
        # Re-inițializăm valorile în state pentru a fi folosite ca 'value' mai jos
        st.session_state.bin_size = 10
        st.session_state.sigma_val = 5.0
        st.session_state.period_range = (1.0, 30.0)
        st.session_state.selected_missions = ["TESS", "Kepler", "K2"]
        st.session_state.selected_authors = ["SPOC", "Kepler"]
        
        # Salvează reset-ul în fișier
        save_all_settings()
        st.rerun()

st.header(" Configurare Vânător de Exoplanete")
st.markdown("""
Ajustează parametrii de mai jos pentru a optimiza modul în care algoritmul procesează datele telescopului. 
*Setările sunt salvate automat pentru întreaga sesiune.*
""")

# --- SECȚIUNEA 1: SURSE DE DATE ---
with st.expander(" Surse de Date și Misiuni", expanded=True):
    st.info("Alege de unde să fie descărcate datele brute.")
    
    st.multiselect(
        "Misiuni Spațiale", 
        options=["TESS", "Kepler", "K2"], 
        default=st.session_state.selected_missions,
        key="selected_missions",
        on_change=save_all_settings,
        help="TESS (misiune activă) vs Kepler/K2 (date istorice de mare precizie)."
    )
    
    st.multiselect(
        "Autori / Pipeline-uri", 
        options=["SPOC", "Kepler", "K2", "QLP", "TESS-SPOC"], 
        default=st.session_state.selected_authors,
        key="selected_authors",
        on_change=save_all_settings,
        help="Pipeline-ul reprezintă metoda prin care datele brute au fost procesate inițial de NASA sau universități."
    )

st.divider()

# --- SECȚIUNEA 2: PROCESARE SEMNAL ---
st.subheader("Curățare și Netezire (Preprocessing)")
st.markdown("Acești parametri decid cum „pregătim” curba de lumină înainte de a căuta planete.")

col1, col2 = st.columns(2)

with col1:
    st.slider(
        'Dimensiunea bin-ului (minute)', 
        min_value=1, max_value=60, 
        value=st.session_state.bin_size,
        key='bin_size',
        on_change=save_all_settings,
        help="Gruparea punctelor reduce zgomotul instrumental (zgomotul alb)."
    )
    st.caption("""
    **Ghid:**
    - **10-20 min:** Ideal pentru majoritatea planetelor.
    - **Peste 30 min:** Netezește prea mult; poți pierde planete mici (mărimea Pământului).
    """)

with col2:
    st.slider(
        'Sigma Clipping (Prag erori)', 
        min_value=1.0, max_value=10.0, step=0.1, 
        value=st.session_state.sigma_val,
        key='sigma_val',
        on_change=save_all_settings,
        help="Identifică și elimină punctele care deviază prea mult de la medie (erori de senzor)."
    )
    st.caption("""
    **Ghid:**
    - **Sub 3.0:** Foarte agresiv (poate tăia din tranzitul real).
    - **5.0:** Standardul echilibrat.
    - **Peste 7.0:** Permisiv (păstrează mai mult zgomot).
    """)

# --- SECȚIUNEA 3: CĂUTARE PERIODICĂ ---
st.divider()
st.subheader("Parametri Algoritm BLS (Box Least Squares)")
st.markdown("Definește limitele de timp în care algoritmul caută orbitele planetelor.")



st.slider(
    "Interval de perioade orbitale (zile)", 
    min_value=0.5, max_value=100.0, 
    value=st.session_state.period_range,
    on_change=save_all_settings,
    key='period_range',
    help="Definește durata minimă și maximă a unui 'an' pe planeta căutată."
)

# Ghid vizual rapid bazat pe selecție
p_min, p_max = st.session_state.get('period_range', (1.0, 30.0))
if p_max <= 10:
    st.warning(f"⚠️ Căutare limitată la planete extrem de apropiate (orbite sub {p_max} zile).")
elif p_max >= 50:
    st.info(f"Căutare extinsă până la {p_max} zile. Procesarea va dura mai mult.")
else:
    st.success("Interval optim pentru detectarea candidaților TESS/Kepler.")

# Vizualizare pentru Debug (Opțional, poți să-l ștergi dacă te încurcă)
with st.expander("Verifică Starea Tehnică"):
    st.json({
        "bin_size": st.session_state.bin_size,
        "sigma": st.session_state.sigma_val,
        "period_range": st.session_state.period_range
    })
