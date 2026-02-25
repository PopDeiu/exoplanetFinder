# utils/__init__.py

# Importăm funcțiile din modulele interne pentru a le face accesibile direct din 'utils'
from .ui_styles import (
    set_galaxy_background, 
    set_sidebar_style, 
    apply_pro_plotting_style
)

from .data_fetchers import (
    get_toi_catalog, 
    search_toi_catalog, 
    fetch_star_data, 
    get_star_parameters, 
    fetch_catalog_targets, 
    fetch_untested_targets
)

from .analysis_engine import process_selected_data

from .settings_manager import (
    init_session_state,
    load_settings,
    save_settings,
    update_setting
)
