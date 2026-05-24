import streamlit as st
import matplotlib.pyplot as plt

def apply_pro_plotting_style():
    """Configurează Matplotlib pentru un aspect profesional (Dark Mode)."""
    plt.style.use('dark_background')
    params = {
        "axes.facecolor": "#0E1117",
        "figure.facecolor": "#0E1117",
        "axes.edgecolor": "#444",
        "grid.color": "#222",
        "font.family": "sans-serif",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9
    }
    plt.rcParams.update(params)

def set_galaxy_background(gradient_type="default"):
    """Aplică fundalul cu temă cosmică."""
    gradients = {
        "default": "linear-gradient(135deg, rgba(10, 5, 20, 0.9) 0%, rgba(40, 20, 70, 0.85) 50%, rgba(5, 2, 15, 0.92) 100%)",
        "nebula": "linear-gradient(120deg, rgba(50, 25, 90, 0.88) 0%, rgba(70, 30, 120, 0.85) 40%, rgba(15, 8, 30, 0.9) 100%)",
        "void": "radial-gradient(ellipse at 20% 50%, rgba(60, 20, 100, 0.85) 0%, rgba(10, 5, 20, 0.9) 60%, rgba(3, 1, 10, 0.95) 100%)",
        "cosmic": "linear-gradient(45deg, rgba(8, 4, 16, 0.92) 0%, rgba(50, 15, 90, 0.8) 35%, rgba(80, 30, 140, 0.75) 100%)",
        "stellar": "conic-gradient(from 45deg at 30% 50%, rgba(15, 5, 35, 0.9) 0deg, rgba(60, 20, 110, 0.8) 120deg, rgba(40, 15, 80, 0.85) 240deg, rgba(15, 5, 35, 0.9) 360deg)"
    }
    bg = gradients.get(gradient_type, gradients["default"])
    st.markdown(f"""
    <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stBaseViewContainer"] {{
            background: {bg} !important; background-attachment: fixed;
        }}
        .stApp {{ background: {bg} !important; background-attachment: fixed; }}
        [data-testid="stHeader"], [data-testid="stToolbar"] {{ background: transparent !important; }}
        .stMainBlockContainer {{ background: transparent !important; }}
    </style>
    """, unsafe_allow_html=True)

def set_sidebar_style():
    """Aplică stilul sidebar-ului cu stele și asigură interactivitatea butoanelor."""
    st.markdown("""
    <style>
        /* Fundalul principal al sidebar-ului */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(30, 15, 60, 0.95) 0%, rgba(10, 5, 30, 0.95) 100%) !important;
            z-index: 0;
        }
        
        /* Stratul cu stele - POINTER-EVENTS: NONE este cheia succesului */
        [data-testid="stSidebar"]::before {
            content: ''; 
            position: absolute; 
            top: 0; left: 0; right: 0; bottom: 0;
            background-image: 
                radial-gradient(2px 2px at 20px 30px, #e0e0ff, rgba(0,0,0,0)),
                radial-gradient(1px 1px at 100px 250px, #ffffff, rgba(0,0,0,0)),
                radial-gradient(1.5px 1.5px at 50px 80px, #c8b6ff, rgba(0,0,0,0));
            background-repeat: repeat; 
            background-size: 150px 300px; 
            opacity: 0.4;
            z-index: -1; /* Trimite stelele în spatele butoanelor */
            pointer-events: none; /* Ignoră click-urile, lăsându-le să treacă la butoane */
        }

        /* Asigură-te că link-urile meniului sunt deasupra stelelor */
        [data-testid="stSidebarNav"] {
            position: relative;
            z-index: 10;
        }

        /* Logo cu spațiu deasupra */
        [data-testid="stSidebarHeader"] {
            padding-top: 20px !important;
        }
        [data-testid="stSidebarHeader"] img {
            width: 80px !important;
            height: auto !important;
        }
    </style>
    """, unsafe_allow_html=True)
