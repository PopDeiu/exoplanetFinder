# utils/__init__.py

# Importăm funcțiile din modulele interne pentru a le face accesibile direct din 'utils'
from .ui_styles import (
    set_galaxy_background, 
    set_sidebar_style, 
    apply_pro_plotting_style
)
# Adaugă asta la finalul fișierului utils/__init__.py
from .database import (  # Înlocuiește 'database' cu numele real al fișierului tău
    get_connection,
    get_all_naked_eye_stars,
    get_real_stars_by_bortle,
    clear_all_naked_eye_stars,
    bulk_save_stars,
    get_saved_location,
    update_app_setting
)

from .data_fetchers import (
    get_toi_catalog, 
    search_toi_catalog, 
    fetch_star_data, 
    get_star_parameters, 
    fetch_catalog_targets, 
    fetch_untested_targets,
    get_stars_with_confirmed_planets
)

from .analysis_engine import process_selected_data, generate_pdf_report

from .settings_manager import (
    init_session_state,
    load_settings,
    save_settings,
    update_setting
)
