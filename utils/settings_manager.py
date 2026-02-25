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
