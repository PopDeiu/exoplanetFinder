"""
Manager pentru salvarea și încărcarea setărilor aplicației persistente.
Valorile sunt salvate în fișierul settings.json și se reîncarcă la fiecare sesiune.
"""

import json
import os
from pathlib import Path

# Calea către directorul proiectului și fișierul de setări
PROJECT_ROOT = Path(__file__).parent.parent
SETTINGS_FILE = PROJECT_ROOT / "settings.json"

# Valorile default
DEFAULT_SETTINGS = {
    "bin_size": 10,
    "sigma_val": 5.0,
    "period_range": [1.0, 30.0],
    "selected_missions": ["TESS", "Kepler", "K2"],
    "selected_authors": ["SPOC", "Kepler"]
}


def load_settings():
    """
    Încarcă setările din fișierul JSON.
    Dacă fișierul nu există, returnează valorile default.
    """
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                # Merge cu default pentru a adăuga orice setări lipsă
                return {**DEFAULT_SETTINGS, **settings}
        except Exception as e:
            print(f"Eroare la încărcarea setărilor: {e}")
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()


def save_settings(settings_dict):
    """
    Salvează setările în fișierul JSON.
    """
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings_dict, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Eroare la salvarea setărilor: {e}")


def get_setting(key, default=None):
    """
    Obține o anumită setare din fișier.
    """
    settings = load_settings()
    return settings.get(key, default)


def update_setting(key, value):
    """
    Actualizează o setare și o salvează în fișier.
    """
    settings = load_settings()
    settings[key] = value
    save_settings(settings)


def init_session_state():
    """
    Inițializează session state cu setările salvate persistent.
    Apelează această funcție la începutul fiecărei pagini pentru a asigura că
    datele din fișier sunt încărcate în session_state.
    
    Exemplu de utilizare:
        import streamlit as st
        from utils.settings_manager import init_session_state
        
        st.set_page_config(...)
        init_session_state()  # La început, înainte de widgets
    """
    import streamlit as st
    
    persisted_settings = load_settings()
    
    if 'bin_size' not in st.session_state:
        st.session_state.bin_size = persisted_settings['bin_size']
    if 'sigma_val' not in st.session_state:
        st.session_state.sigma_val = persisted_settings['sigma_val']
    if 'period_range' not in st.session_state:
        st.session_state.period_range = tuple(persisted_settings['period_range'])
    if 'selected_missions' not in st.session_state:
        st.session_state.selected_missions = persisted_settings['selected_missions']
    if 'selected_authors' not in st.session_state:
        st.session_state.selected_authors = persisted_settings['selected_authors']
