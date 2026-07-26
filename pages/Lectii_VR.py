import streamlit as st
from datetime import datetime
import pytz
from utils.data_fetchers import get_stars_from_simbad
from utils.database import clear_all_naked_eye_stars, bulk_save_stars, get_all_settings, update_app_setting
from utils.database import save_scenariu, get_all_scenarii, get_scenariu_by_id, delete_scenariu, rename_scenariu
from utils.database import init_lectii_table, save_lectie, get_all_lectii, get_lectie_by_id, delete_lectie, rename_lectie
from utils.database import ensure_tables_have_user_id
from utils.database import init_stars_bortle_table, clear_stars_by_bortle, bulk_save_stars_bortle
from utils.tts_utils import generate_scenario_wav
from utils.ui_styles import set_galaxy_background, set_sidebar_style
from utils.auth import init_auth, render_sidebar_auth


RO_TZ = pytz.timezone('Europe/Bucharest')

ORASE = {
    "Arad": (46.1866, 21.3123),
    "București": (44.4268, 26.1025),
    "New York (NYC)": (40.7128, -74.0060),
    "Tokyo": (35.6762, 139.6503),
    "Londra": (51.5074, -0.1278),
    "Sydney": (-33.8688, 151.2093),
    "Buenos Aires": (-34.6037, -58.3816),
    "Capetown": (-33.9249, 18.4241),
    "Mexico City": (19.4326, -99.1332),
    "Iași": (47.1585, 27.6014),
}
LOCATIE_CUSTOM = "Personalizat"

def load_settings_into_session():
    """Citește din DB și actualizează session_state."""
    db_settings = get_all_settings()
    default_time = datetime.now(RO_TZ)
    st.session_state.lat = float(db_settings.get("latitudine", 46.1866))
    st.session_state.lon = float(db_settings.get("longitudine", 21.3123))
    st.session_state.viteza_slider = int(db_settings.get("viteza", 0))
    city_db = db_settings.get("oras", "Arad")
    saved_lat = float(db_settings.get("latitudine", 46.1866))
    saved_lon = float(db_settings.get("longitudine", 21.3123))
    city_matches = False
    for nume_s, (lat_s, lon_s) in ORASE.items():
        if abs(lat_s - saved_lat) < 0.01 and abs(lon_s - saved_lon) < 0.01:
            st.session_state.city_selector = nume_s
            city_matches = True
            break
    if not city_matches:
        st.session_state.city_selector = LOCATIE_CUSTOM
    st.session_state.bortle_slider = int(db_settings.get("bortle", 4))
    st.session_state.use_current_time_check = db_settings.get("foloseste_data_curenta", "da") == "da"
    st.session_state.afisare_constelatii = db_settings.get("afisare_constelatii", "da") == "da"
    data_db = db_settings.get("data_si_ora_obs")
    if data_db:
        try:
            dt_obj = datetime.strptime(data_db, "%Y-%m-%d %H:%M:%S")
            st.session_state.manual_date = dt_obj.date()
            st.session_state.manual_time = dt_obj.time()
        except:
            st.session_state.manual_date = default_time.date()
            st.session_state.manual_time = default_time.time()
    else:
        st.session_state.manual_date = default_time.date()
        st.session_state.manual_time = default_time.time()

# Reîncărcăm setările din DB doar la intrarea pe pagină
if st.session_state.get("_last_page") != "Lectii_VR":
    load_settings_into_session()
init_auth()

st.set_page_config(page_title="Lecții VR")
set_sidebar_style()
set_galaxy_background("stellar")
st.logo("assets/ExoLogo_noBg.png", size="large")

ensure_tables_have_user_id()

# ========== TITLU ==========
st.markdown("# Lecții VR")

# ========== SIDEBAR (mereu vizibil) ==========
with st.sidebar:
    st.markdown("### Descarcă VR")
    with open("assets/planetariuVR.apk", "rb") as f:
        st.download_button(
            label="Planetariu VR APK",
            data=f,
            file_name="planetariuVR.apk",
            mime="application/vnd.android.package-archive",
            use_container_width=True
        )
    render_sidebar_auth()

# Verificăm dacă utilizatorul este logat
if not st.session_state.get('logged_in', False):
    st.warning("Trebuie să fii autentificat (folosește formularul din bara laterală) pentru a accesa această pagină.")
    st.stop()

user_id = st.session_state.user_info['ID']

if "lat" not in st.session_state:
    st.session_state.lat = 46.1866
if "lon" not in st.session_state:
    st.session_state.lon = 21.3123
if "current_lectie_id" not in st.session_state:
    st.session_state.current_lectie_id = None
if "current_lectie_nume" not in st.session_state:
    st.session_state.current_lectie_nume = None
if "descriere_lectie" not in st.session_state:
    st.session_state.descriere_lectie = ""
if "durata_preset" not in st.session_state:
    st.session_state.durata_preset = "30 secunde"
if "durata_custom" not in st.session_state:
    st.session_state.durata_custom = 30
if "lectie_curenta_id" not in st.session_state:
    st.session_state.lectie_curenta_id = None
if "lectie_curenta_nume" not in st.session_state:
    st.session_state.lectie_curenta_nume = None
if "lectie_descriere" not in st.session_state:
    st.session_state.lectie_descriere = ""
if "lectie_scenarii_selectate" not in st.session_state:
    st.session_state.lectie_scenarii_selectate = []
if "adding_scenariu" not in st.session_state:
    st.session_state.adding_scenariu = False
if "adding_lectie" not in st.session_state:
    st.session_state.adding_lectie = False
if "rename_scenariu_id" not in st.session_state:
    st.session_state.rename_scenariu_id = None
if "rename_lectie_id" not in st.session_state:
    st.session_state.rename_lectie_id = None
if "view_lesson_scenarios_id" not in st.session_state:
    st.session_state.view_lesson_scenarios_id = None
if "lesson_rename_scenariu_id" not in st.session_state:
    st.session_state.lesson_rename_scenariu_id = None
if "loaded_in_lesson_tab" not in st.session_state:
    st.session_state.loaded_in_lesson_tab = False

init_lectii_table()
init_stars_bortle_table()

DURATA_PRESET = {
    "10 secunde": 10,
    "20 secunde": 20,
    "30 secunde": 30,
    "1 minut": 60,
    "2 minute": 120,
    "3 minute": 180,
    "6 minute": 360,
}

def on_city_change():
    oras = st.session_state.city_selector
    if oras == LOCATIE_CUSTOM:
        return
    if oras in ORASE:
        st.session_state.lat = ORASE[oras][0]
        st.session_state.lon = ORASE[oras][1]

# ========== FUNCȚIE CONFIGURARE SCENARIU ==========
def _apply_pending_scenario():
    """Aplică valorile unui scenariu încărcat (prin butonul 👁️) în cheile widgeturilor.
    Trebuie apelată ÎNAINTE ca widgeturile să fie instanțiate, altfel Streamlit interzice
    scrierea în st.session_state pentru o cheie de widget deja creată în acest run."""
    scenariu = st.session_state.pop("_pending_scenario", None)
    if not scenariu:
        return
    st.session_state.bortle_slider = int(scenariu["bortle"])
    st.session_state.lat = float(scenariu["latitudine"])
    st.session_state.lon = float(scenariu["longitudine"])
    st.session_state.viteza_slider = int(scenariu["viteza"])
    st.session_state.use_current_time_check = scenariu["foloseste_data_curenta"] in ("da", "True")
    st.session_state.afisare_constelatii = scenariu["afisare_constelatii"] in ("da", "True")
    data_ora = scenariu.get("data_si_ora_obs")
    if data_ora and data_ora != "None":
        try:
            dt_obj = datetime.strptime(str(data_ora), "%Y-%m-%d %H:%M:%S")
            st.session_state.manual_date = dt_obj.date()
            st.session_state.manual_time = dt_obj.time()
        except:
            pass
    for nume_s, (lat_s, lon_s) in ORASE.items():
        if abs(lat_s - float(scenariu["latitudine"])) < 0.01 and abs(lon_s - float(scenariu["longitudine"])) < 0.01:
            st.session_state.city_selector = nume_s
            break
    else:
        st.session_state.city_selector = LOCATIE_CUSTOM
    st.session_state.descriere_lectie = str(scenariu.get("text", ""))
    durata_val = int(scenariu.get("durata", 0))
    st.session_state.durata_custom = durata_val if durata_val > 0 else 30
    st.session_state.durata_preset = next(
        (k for k, v in DURATA_PRESET.items() if v == durata_val),
        "Personalizat"
    )


def render_scenario_config(show_close=True):
    _apply_pending_scenario()
    if show_close:
        st.markdown("---")
        if st.session_state.current_lectie_id and st.session_state.current_lectie_nume:
            st.markdown(f"## Configurare Scenariu: **{st.session_state.current_lectie_nume}**")
        else:
            st.markdown("## Configurare Scenariu Nou")

        if st.button("✖️ Închide configurarea"):
            st.session_state.adding_scenariu = False
            st.session_state.loaded_in_lesson_tab = False
            if st.session_state.current_lectie_id:
                st.session_state.current_lectie_id = None
                st.session_state.current_lectie_nume = None
            st.rerun()

    with st.expander("1. Sincronizare după Poluare Luminoasă (Bortle)", expanded=True):
        st.select_slider(
            "Selectează nivelul Bortle:",
            options=list(range(1, 10)),
            key="bortle_slider"
        )

    with st.expander("2. Sincronizare după Locație și Timp", expanded=True):
        use_current_time = st.checkbox(
            "Folosește data și ora curentă (România)",
            key="use_current_time_check"
        )
        now_ro = datetime.now(RO_TZ)

        if not use_current_time:
            col_t1, col_t2 = st.columns(2)
            d = col_t1.date_input("Data", value=now_ro.date(), key="manual_date")
            t = col_t2.time_input("Ora", value=now_ro.time(), key="manual_time")
            RO_TZ.localize(datetime.combine(d, t))
        else:
            st.info(f"Ora curentă (RO): {now_ro.strftime('%H:%M:%S')}")

        st.slider("Viteza simulării:", -1000, 1000, key="viteza_slider")

        st.selectbox(
            "Alege un oraș preset:",
            options=list(ORASE.keys()) + [LOCATIE_CUSTOM],
            key="city_selector",
            on_change=on_city_change
        )

        col1, col2 = st.columns(2)
        col1.number_input("Latitudine (°N)", -90.0, 90.0, key="lat", format="%.4f")
        col2.number_input("Longitudine (°E)", -180.0, 180.0, key="lon", format="%.4f")

    def on_constelatii_change():
        update_app_setting("afisare_constelatii", "da" if st.session_state.afisare_constelatii else "nu")

    with st.expander("3. Setări afișare", expanded=True):
        st.checkbox(
            "Afișare Constelații",
            key="afisare_constelatii",
            on_change=on_constelatii_change
        )

    st.markdown("---")
    st.markdown("## Salvare Scenariu")

    st.text_area(
        "Descrierea scenariului:",
        key="descriere_lectie",
        placeholder="Scrie o descriere pentru acest scenariu..."
    )

    durata_options = list(DURATA_PRESET.keys()) + ["Personalizat"]
    durata_preset_sel = st.selectbox(
        "Durata scenariului:",
        options=durata_options,
        key="durata_preset"
    )
    if durata_preset_sel == "Personalizat":
        st.number_input(
            "Durata (secunde):",
            min_value=1, value=st.session_state.durata_custom,
            key="durata_custom"
        )

    with st.container():
        if st.session_state.current_lectie_id and st.session_state.current_lectie_nume:
            st.info(f"Editezi scenariul: **{st.session_state.current_lectie_nume}**")
            if st.button("💾 Salvare modificări", type="primary", use_container_width=True):
                nume_lectie = st.session_state.current_lectie_nume
                viteza_val = st.session_state.get("viteza_slider", 0)
                bortle_val = st.session_state.get("bortle_slider", 4)
                lat_val_save = st.session_state.get("lat", 46.1866)
                lon_val_save = st.session_state.get("lon", 21.3123)
                fol_curenta = "da" if st.session_state.get("use_current_time_check", True) else "nu"
                afis_const = "da" if st.session_state.get("afisare_constelatii", True) else "nu"
                data_ora_save = st.session_state.get("manual_date", now_ro.date()).strftime("%Y-%m-%d") + " " + st.session_state.get("manual_time", now_ro.time()).strftime("%H:%M:%S")
                if st.session_state.get("use_current_time_check", True):
                    data_ora_save = now_ro.strftime("%Y-%m-%d %H:%M:%S")
                text_desc = st.session_state.get("descriere_lectie", "")
                preset_curent = st.session_state.get("durata_preset", "Personalizat")
                durata_val = DURATA_PRESET[preset_curent] if preset_curent in DURATA_PRESET else st.session_state.get("durata_custom", 0)
                result = save_scenariu(nume_lectie, viteza_val, bortle_val, lon_val_save, lat_val_save, data_ora_save, fol_curenta, afis_const, text=text_desc, durata=durata_val, user_id=user_id, scenariu_id=st.session_state.current_lectie_id)
                if result:
                    st.success(f"✅ Scenariul „{nume_lectie}” a fost salvat!")
                    st.rerun()
                else:
                    st.error("Nu s-a putut salva scenariul. Verifică conexiunea la baza de date.")
        else:
            nume_lectie = st.text_input("Denumește scenariul:", placeholder="Ex: Observație seară de vară", key="nume_lectie_input")
            if st.button("💾 Salvează Scenariul", type="primary", use_container_width=True):
                if not nume_lectie or nume_lectie.strip() == "":
                    st.warning("Te rog introdu un nume pentru scenariu.")
                else:
                    viteza_val = st.session_state.get("viteza_slider", 0)
                    bortle_val = st.session_state.get("bortle_slider", 4)
                    lat_val_save = st.session_state.get("lat", 46.1866)
                    lon_val_save = st.session_state.get("lon", 21.3123)
                    fol_curenta = "da" if st.session_state.get("use_current_time_check", True) else "nu"
                    afis_const = "da" if st.session_state.get("afisare_constelatii", True) else "nu"
                    data_ora_save = st.session_state.get("manual_date", now_ro.date()).strftime("%Y-%m-%d") + " " + st.session_state.get("manual_time", now_ro.time()).strftime("%H:%M:%S")
                    if st.session_state.get("use_current_time_check", True):
                        data_ora_save = now_ro.strftime("%Y-%m-%d %H:%M:%S")
                    text_desc = st.session_state.get("descriere_lectie", "")
                    preset_curent = st.session_state.get("durata_preset", "Personalizat")
                    durata_val = DURATA_PRESET[preset_curent] if preset_curent in DURATA_PRESET else st.session_state.get("durata_custom", 0)
                    result = save_scenariu(nume_lectie.strip(), viteza_val, bortle_val, lon_val_save, lat_val_save, data_ora_save, fol_curenta, afis_const, text=text_desc, durata=durata_val, user_id=user_id)
                    if result:
                        st.success(f"✅ Scenariul „{nume_lectie.strip()}” a fost salvat!")
                        st.session_state.adding_scenariu = False
                        st.session_state.loaded_in_lesson_tab = False
                        st.session_state.current_lectie_id = None
                        st.session_state.current_lectie_nume = None
                        st.rerun()
                    else:
                        st.error("Nu s-a putut salva scenariul. Verifică conexiunea la baza de date.")

# ========== CSS ==========
st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] {
    gap: 2px !important;
}

button[kind="secondary"][data-testid^="baseButton-rs_"],
button[kind="secondary"][data-testid^="baseButton-ds_"],
button[kind="secondary"][data-testid^="baseButton-rl_"],
button[kind="secondary"][data-testid^="baseButton-dl_"],
button[kind="secondary"][data-testid^="baseButton-rsl_"],
button[kind="secondary"][data-testid^="baseButton-dsl_"] {
    opacity: 0 !important;
    transition: opacity 0.2s !important;
    border: none !important;
    background: transparent !important;
    padding: 2px 4px !important;
    min-width: auto !important;
    width: 28px !important;
    height: 28px !important;
    font-size: 1em !important;
    color: #e0e0ff !important;
    box-shadow: none !important;
    line-height: 1 !important;
}

div[data-testid="stHorizontalBlock"]:hover button[kind="secondary"][data-testid^="baseButton-rs_"],
div[data-testid="stHorizontalBlock"]:hover button[kind="secondary"][data-testid^="baseButton-ds_"],
div[data-testid="stHorizontalBlock"]:hover button[kind="secondary"][data-testid^="baseButton-rl_"],
div[data-testid="stHorizontalBlock"]:hover button[kind="secondary"][data-testid^="baseButton-dl_"],
div[data-testid="stHorizontalBlock"]:hover button[kind="secondary"][data-testid^="baseButton-rsl_"],
div[data-testid="stHorizontalBlock"]:hover button[kind="secondary"][data-testid^="baseButton-dsl_"] {
    opacity: 0.6 !important;
}

button[kind="secondary"][data-testid^="baseButton-rs_"]:hover,
button[kind="secondary"][data-testid^="baseButton-ds_"]:hover,
button[kind="secondary"][data-testid^="baseButton-rl_"]:hover,
button[kind="secondary"][data-testid^="baseButton-dl_"]:hover,
button[kind="secondary"][data-testid^="baseButton-rsl_"]:hover,
button[kind="secondary"][data-testid^="baseButton-dsl_"]:hover {
    opacity: 1 !important;
    background: rgba(255,255,255,0.15) !important;
}

button[kind="secondary"][data-testid^="baseButton-gs_"],
button[kind="secondary"][data-testid^="baseButton-ls_"],
button[kind="secondary"][data-testid^="baseButton-ll_"],
button[kind="secondary"][data-testid^="baseButton-vsl_"],
button[kind="secondary"][data-testid^="baseButton-al_"] {
    border: none !important;
    background: transparent !important;
    padding: 2px 4px !important;
    min-width: auto !important;
    width: 28px !important;
    height: 28px !important;
    font-size: 1em !important;
    color: #e0e0ff !important;
    box-shadow: none !important;
    transition: all 0.2s !important;
    line-height: 1 !important;
}

button[kind="secondary"][data-testid^="baseButton-ls_"]:hover,
button[kind="secondary"][data-testid^="baseButton-ll_"]:hover,
button[kind="secondary"][data-testid^="baseButton-vsl_"]:hover,
button[kind="secondary"][data-testid^="baseButton-al_"]:hover {
    background: rgba(255,255,255,0.1) !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# ========== MESAJE TTS PERSISTENTE ==========
for key in ['_tts_msg', '_tts_btn_msg']:
    if key in st.session_state:
        msg_type, msg = st.session_state.pop(key)
        if msg_type == 'success':
            st.success(msg)
        elif msg_type == 'warning':
            st.warning(msg)
        elif msg_type == 'info':
            st.info(msg)

# ========== TAB-URI PRINCIPALE ==========
tab_scenarii, tab_lectii, tab_setari = st.tabs(["Scenarii", "Lecții", "Setări"])

# =====================================================================
# TAB 1: SCENARII
# =====================================================================
with tab_scenarii:
    st.markdown("### Scenarii")

    scenarii = get_all_scenarii(user_id)

    col_top, col_btn = st.columns([5, 1])
    with col_btn:
        if st.button("➕ Adaugă Scenariu", use_container_width=True):
            st.session_state.bortle_slider = 4
            st.session_state.lat = 46.1866
            st.session_state.lon = 21.3123
            st.session_state.viteza_slider = 0
            st.session_state.use_current_time_check = True
            st.session_state.afisare_constelatii = True
            st.session_state.city_selector = "Arad"
            st.session_state.current_lectie_id = None
            st.session_state.current_lectie_nume = None
            st.session_state.descriere_lectie = ""
            st.session_state.durata_preset = "30 secunde"
            st.session_state.durata_custom = 30
            st.session_state.adding_scenariu = True
            st.session_state.loaded_in_lesson_tab = False
            st.session_state.rename_scenariu_id = None
            st.rerun()

    if scenarii:
        h = st.columns([4, 1, 1, 1, 1], gap="small")
        h[0].markdown("**Nume**")
        h[1].markdown("")
        h[2].markdown("")
        h[3].markdown("")
        h[4].markdown("")

        for s in scenarii:
            cols = st.columns([2, 1, 1, 1, 1], gap="small")
            with cols[0]:
                st.write(s["nume"])
            with cols[1]:
                if st.button("👁️", key=f"ls_{s['ID']}"):
                    scenariu = get_scenariu_by_id(s["ID"], user_id)
                    if scenariu:
                        st.session_state["_pending_scenario"] = scenariu
                        st.session_state.current_lectie_id = scenariu["ID"]
                        st.session_state.current_lectie_nume = scenariu["nume"]
                        st.session_state.adding_scenariu = False
                        st.session_state.loaded_in_lesson_tab = False
                        st.session_state.rename_scenariu_id = None
                        st.rerun()
            with cols[2]:
                if st.button("✏️", key=f"rs_{s['ID']}"):
                    st.session_state.rename_scenariu_id = s["ID"]
                    st.rerun()
            with cols[3]:
                if st.button("🗑️", key=f"ds_{s['ID']}"):
                    if delete_scenariu(s["ID"], user_id):
                        if st.session_state.current_lectie_id == s["ID"]:
                            st.session_state.current_lectie_id = None
                            st.session_state.current_lectie_nume = None
                        st.session_state.rename_scenariu_id = None
                        st.rerun()
            with cols[4]:
                if st.button("🔊", key=f"gs_{s['ID']}"):
                    scenariu = get_scenariu_by_id(s["ID"], user_id)
                    if scenariu and scenariu.get("text", "").strip():
                        wav_result = generate_scenario_wav(s["ID"], scenariu["text"])
                        if wav_result:
                            st.session_state['_tts_btn_msg'] = ("success", f"🔊 Audio regenerat cu succes pentru „{scenariu['nume']}”")
                        else:
                            st.session_state['_tts_btn_msg'] = ("warning", f"Eroare la generarea audio pentru „{scenariu['nume']}”")
                    else:
                        st.session_state['_tts_btn_msg'] = ("info", f"Scenariul „{s['nume']}” nu are text pentru generare audio.")
                    st.rerun()
    else:
        st.info("Nu există scenarii încă. Creează unul nou apăsând „➕ Adaugă Scenariu”.")

    # ========== RENAME SCENARIU ==========
    if st.session_state.get("rename_scenariu_id"):
        ren_id = st.session_state.rename_scenariu_id
        ren_s = next((s for s in scenarii if s["ID"] == ren_id), None)
        if ren_s:
            st.markdown("#### Redenumește Scenariu")
            col_r1, col_r2, col_r3 = st.columns([3, 1, 1])
            with col_r1:
                nume_nou = st.text_input("Nume nou", value=ren_s["nume"], key="ren_scenariu_input")
            with col_r2:
                if st.button("✅ Salvează", use_container_width=True):
                    if nume_nou and nume_nou != ren_s["nume"]:
                        if rename_scenariu(ren_id, nume_nou, user_id):
                            if st.session_state.current_lectie_id == ren_id:
                                st.session_state.current_lectie_nume = nume_nou
                            st.session_state.rename_scenariu_id = None
                            st.rerun()
            with col_r3:
                if st.button("❌ Anulează", use_container_width=True):
                    st.session_state.rename_scenariu_id = None
                    st.rerun()

    # ========== CONFIGURARE SCENARIU (doar dacă adăugăm sau am încărcat, și nu e deja deschis în tab-ul Lecții) ==========
    show_config_scen = not st.session_state.get("loaded_in_lesson_tab", False) and \
                       (st.session_state.get("adding_scenariu", False) or \
                        (st.session_state.current_lectie_id is not None and st.session_state.current_lectie_nume is not None))

    if show_config_scen:
        render_scenario_config()
    else:
        st.caption("Apasă **👁️** pe un scenariu din tabel pentru a-i edita configurația.")

# =====================================================================
# TAB 2: LECȚII
# =====================================================================
with tab_lectii:
    st.markdown("### Lecții")
    st.markdown("Compune o lecție înlănțuind mai multe scenarii.")

    lectii = get_all_lectii(user_id)

    col_top_l, col_btn_l = st.columns([5, 1])
    with col_btn_l:
        if st.button("➕ Adaugă Lecție", key="add_lectie_btn", use_container_width=True):
            st.session_state.lectie_curenta_id = None
            st.session_state.lectie_curenta_nume = None
            st.session_state.lectie_descriere = ""
            st.session_state.lectie_scenarii_selectate = []
            st.session_state.adding_lectie = True
            st.session_state.rename_lectie_id = None
            st.session_state.view_lesson_scenarios_id = None
            st.rerun()

    if lectii:
        lectie_activa_val = int(get_all_settings().get("lectie_activa", 0))

        h2 = st.columns([2, 1, 1, 1, 2], gap="small")
        h2[0].markdown("**Nume**")
        h2[1].markdown("")
        h2[2].markdown("")
        h2[3].markdown("")
        h2[4].markdown("**Status**")

        for l in lectii:
            cols = st.columns([2, 1, 1, 1, 2], gap="small")
            is_active = (lectie_activa_val == l["ID"])
            with cols[0]:
                st.write(l["nume"])
            with cols[1]:
                if st.button("👁️", key=f"ll_{l['ID']}"):
                    lectie = get_lectie_by_id(l["ID"], user_id)
                    if lectie:
                        st.session_state.lectie_curenta_id = lectie["ID"]
                        st.session_state.lectie_curenta_nume = lectie["nume"]
                        st.session_state.lectie_descriere = lectie.get("descriere", "")
                        ids_str = lectie.get("scenarii_ids", "")
                        st.session_state.lectie_scenarii_selectate = [int(x) for x in ids_str.split(",") if x.strip().isdigit()]
                        st.session_state.adding_lectie = False
                        st.session_state.rename_lectie_id = None
                        st.session_state.view_lesson_scenarios_id = l["ID"]
                        st.rerun()
            with cols[2]:
                if st.button("✏️", key=f"rl_{l['ID']}"):
                    st.session_state.rename_lectie_id = l["ID"]
                    st.rerun()
            with cols[3]:
                if st.button("🗑️", key=f"dl_{l['ID']}"):
                    if delete_lectie(l["ID"], user_id):
                        if st.session_state.lectie_curenta_id == l["ID"]:
                            st.session_state.lectie_curenta_id = None
                            st.session_state.lectie_curenta_nume = None
                        st.session_state.rename_lectie_id = None
                        st.rerun()
            with cols[4]:
                col_b, col_t = st.columns([1, 2.5], gap="small")
                with col_b:
                    if st.button("🟢" if is_active else "⚪", key=f"al_{l['ID']}"):
                        if is_active:
                            update_app_setting("lectie_activa", 0)
                        else:
                            update_app_setting("lectie_activa", l["ID"])
                        st.rerun()
                with col_t:
                    st.markdown(
                        f"<span style='color: #4caf50; font-weight: bold; font-size: 0.85em;'>Lectie Activata</span>" if is_active else
                        f"<span style='color: #9e9e9e; font-size: 0.85em;'>Lectia Inactiva</span>",
                        unsafe_allow_html=True
                    )
    else:
        st.info("Nu există lecții încă. Creează una nouă apăsând „➕ Adaugă Lecție”.")

    # ========== RENAME LECȚIE ==========
    if st.session_state.get("rename_lectie_id"):
        ren_l_id = st.session_state.rename_lectie_id
        ren_l = next((l for l in lectii if l["ID"] == ren_l_id), None)
        if ren_l:
            st.markdown("#### Redenumește Lecția")
            col_rl1, col_rl2, col_rl3 = st.columns([3, 1, 1])
            with col_rl1:
                nume_nou_l = st.text_input("Nume nou", value=ren_l["nume"], key="ren_lectie_input")
            with col_rl2:
                if st.button("✅ Salvează", key="confirm_ren_lectie", use_container_width=True):
                    if nume_nou_l and nume_nou_l != ren_l["nume"]:
                        if rename_lectie(ren_l_id, nume_nou_l, user_id):
                            if st.session_state.lectie_curenta_id == ren_l_id:
                                st.session_state.lectie_curenta_nume = nume_nou_l
                            st.session_state.rename_lectie_id = None
                            st.rerun()
            with col_rl3:
                if st.button("❌ Anulează", key="cancel_ren_lectie", use_container_width=True):
                    st.session_state.rename_lectie_id = None
                    st.rerun()

    # ========== CONFIGURARE LECȚIE (doar dacă adăugăm sau am încărcat) ==========
    show_config_lectie = st.session_state.get("adding_lectie", False) or \
                          (st.session_state.lectie_curenta_id is not None and st.session_state.lectie_curenta_nume is not None)

    if show_config_lectie:
        st.markdown("---")
        if st.session_state.lectie_curenta_id and st.session_state.lectie_curenta_nume:
            st.markdown(f"## Configurare Lecție: **{st.session_state.lectie_curenta_nume}**")
        else:
            st.markdown("## Configurare Lecție Nouă")

        if st.button("✖️ Închide configurarea", key="close_lectie_config"):
            st.session_state.adding_lectie = False
            st.session_state.view_lesson_scenarios_id = None
            st.session_state.loaded_in_lesson_tab = False
            if st.session_state.lectie_curenta_id:
                st.session_state.lectie_curenta_id = None
                st.session_state.lectie_curenta_nume = None
            st.rerun()

        scenarii_disponibile = get_all_scenarii(user_id)
        scenarii_options = {s["ID"]: s["nume"] for s in scenarii_disponibile}
        scenarii_durata = {s["ID"]: int(s.get("durata", 0)) for s in scenarii_disponibile}

        st.multiselect(
            "Selectează scenariile (în ordinea dorită):",
            options=list(scenarii_options.keys()),
            format_func=lambda x: scenarii_options[x],
            default=st.session_state.get("lectie_scenarii_selectate", []),
            key="lectie_scenarii_selectate"
        )

        if st.session_state.lectie_scenarii_selectate:
            ids_cu_durata_zero = [
                scenarii_options[sid] for sid in st.session_state.lectie_scenarii_selectate
                if sid in scenarii_durata and scenarii_durata[sid] == 0
            ]
            total_sec = sum(scenarii_durata[sid] for sid in st.session_state.lectie_scenarii_selectate if sid in scenarii_durata)
            if total_sec >= 60:
                total_display = f"{total_sec // 60} min {total_sec % 60}s"
            else:
                total_display = f"{total_sec}s"
            st.caption(f"Durata totală: **{total_display}**  ·  " + " → ".join(
                scenarii_options[sid] for sid in st.session_state.lectie_scenarii_selectate if sid in scenarii_options
            ))
            if ids_cu_durata_zero:
                st.info(
                    f"Unele scenarii nu au durata setată (**{', '.join(ids_cu_durata_zero)}**). "
                    "Mergi în tab-ul **Scenarii**, încarcă fiecare scenariu și setează durata în secțiunea **Salvare Scenariu**."
                )

        # ========== SUB-TABEL SCENARII DIN LECȚIE ==========
        if st.session_state.lectie_scenarii_selectate and st.session_state.get("view_lesson_scenarios_id"):
            st.markdown("---")
            st.markdown("#### Scenarii din această lecție")

            ids_in_lesson = st.session_state.lectie_scenarii_selectate
            scenarii_in_lesson = [s for s in scenarii_disponibile if s["ID"] in ids_in_lesson]
            id_order = {sid: i for i, sid in enumerate(ids_in_lesson)}
            scenarii_in_lesson.sort(key=lambda s: id_order.get(s["ID"], 999))

            if scenarii_in_lesson:
                h3 = st.columns([2, 1, 1, 1], gap="small")
                h3[0].markdown("**Scenariu**")
                h3[1].markdown("")
                h3[2].markdown("")
                h3[3].markdown("")

                for sc in scenarii_in_lesson:
                    sc_cols = st.columns([2, 1, 1, 1], gap="small")
                    with sc_cols[0]:
                        st.write(sc["nume"])
                    with sc_cols[1]:
                        if st.button("👁️", key=f"vsl_{sc['ID']}"):
                            scenariu = get_scenariu_by_id(sc["ID"], user_id)
                            if scenariu:
                                st.session_state["_pending_scenario"] = scenariu
                                st.session_state.current_lectie_id = scenariu["ID"]
                                st.session_state.current_lectie_nume = scenariu["nume"]
                                st.session_state.adding_scenariu = False
                                st.session_state.loaded_in_lesson_tab = True
                                st.rerun()
                    with sc_cols[2]:
                        if st.button("✏️", key=f"rsl_{sc['ID']}"):
                            st.session_state.lesson_rename_scenariu_id = sc["ID"]
                            st.rerun()
                    with sc_cols[3]:
                        if st.button("🗑️", key=f"dsl_{sc['ID']}"):
                            if delete_scenariu(sc["ID"], user_id):
                                new_ids = [sid for sid in st.session_state.lectie_scenarii_selectate if sid != sc["ID"]]
                                st.session_state.lectie_scenarii_selectate = new_ids
                                st.rerun()

            if st.session_state.get("lesson_rename_scenariu_id"):
                ren_s_id = st.session_state.lesson_rename_scenariu_id
                ren_s = next((s for s in scenarii_disponibile if s["ID"] == ren_s_id), None)
                if ren_s:
                    col_sr1, col_sr2, col_sr3 = st.columns([3, 1, 1])
                    with col_sr1:
                        nume_nou_s = st.text_input("Redenumește scenariul", value=ren_s["nume"], key="lesson_ren_scenariu_input")
                    with col_sr2:
                        if st.button("✅ Salvează", key="lesson_confirm_ren_scen", use_container_width=True):
                            if nume_nou_s and nume_nou_s != ren_s["nume"]:
                                if rename_scenariu(ren_s_id, nume_nou_s, user_id):
                                    st.session_state.lesson_rename_scenariu_id = None
                                    st.rerun()
                    with col_sr3:
                        if st.button("❌ Anulează", key="lesson_cancel_ren_scen", use_container_width=True):
                            st.session_state.lesson_rename_scenariu_id = None
                            st.rerun()

        # ========== CONFIGURARE SCENARIU DIRECT ÎN TAB-UL LECȚII ==========
        if st.session_state.get("loaded_in_lesson_tab") and \
           st.session_state.current_lectie_id and st.session_state.current_lectie_nume:
            render_scenario_config()

        st.markdown("---")
        st.markdown("## Salvare Lecție")

        descriere_lectie_input = st.text_area(
            "Descrierea lecției:",
            value=st.session_state.get("lectie_descriere", ""),
            key="lectie_descriere",
            placeholder="Scrie o descriere pentru această lecție..."
        )

        with st.container():
            if st.session_state.lectie_curenta_id and st.session_state.lectie_curenta_nume:
                st.info(f"Editezi lecția: **{st.session_state.lectie_curenta_nume}**")
                if st.button("💾 Salvare modificări", type="primary", use_container_width=True, key="save_lectie_existing"):
                    ids_str = ",".join(str(x) for x in st.session_state.lectie_scenarii_selectate)
                    if not st.session_state.lectie_scenarii_selectate:
                        st.warning("Selectează cel puțin un scenariu.")
                    elif save_lectie(st.session_state.lectie_curenta_nume, descriere_lectie_input, ids_str, user_id=user_id):
                        st.success(f"✅ Lecția „{st.session_state.lectie_curenta_nume}” a fost salvată!")
                        st.rerun()
                    else:
                        st.error("Nu s-a putut salva lecția. Verifică conexiunea la baza de date.")
            else:
                nume_lectie_input = st.text_input(
                    "Denumește lecția:",
                    key="lectie_nume_input"
                )
                if st.button("💾 Salvează Lecția", type="primary", use_container_width=True, key="save_lectie"):
                    if not nume_lectie_input or nume_lectie_input.strip() == "":
                        st.warning("Te rog introdu un nume pentru lecție.")
                    elif not st.session_state.lectie_scenarii_selectate:
                        st.warning("Selectează cel puțin un scenariu.")
                    else:
                        ids_str = ",".join(str(x) for x in st.session_state.lectie_scenarii_selectate)
                        if save_lectie(nume_lectie_input.strip(), descriere_lectie_input, ids_str, user_id=user_id):
                            st.success(f"✅ Lecția „{nume_lectie_input.strip()}” a fost salvată!")
                            st.session_state.lectie_curenta_id = None
                            st.session_state.lectie_curenta_nume = None
                            st.session_state.adding_lectie = False
                            st.rerun()
                        else:
                            st.error("Nu s-a putut salva lecția. Verifică conexiunea la baza de date.")
    else:
        st.caption("Apasă **👁️** pe o lecție din tabel pentru a-i edita configurația.")

# =====================================================================
# TAB 3: SETĂRI
# =====================================================================
with tab_setari:
    st.markdown("### Setări Sincronizare")

    with st.expander("Sincronizare după Poluare Luminoasă (Bortle)", expanded=True):
        bortle_scale = st.select_slider(
            "Selectează nivelul Bortle:",
            options=list(range(1, 10)),
            key="bortle_slider_setari"
        )

        if st.button("Sincronizează Bortle", type="primary", use_container_width=True):
            with st.spinner("Se actualizează..."):
                now_ro = datetime.now(RO_TZ)
                stars = get_stars_from_simbad(bortle_scale, lat=st.session_state.lat, lon=st.session_state.lon, time=now_ro)
                if stars:
                    clear_stars_by_bortle(bortle_scale)
                    bulk_save_stars_bortle(stars, bortle_scale)
                    st.success(f"✅ S-au salvat {len(stars)} stele pentru Bortle {bortle_scale}.")
                    st.balloons()
                else:
                    st.warning("Nu s-au găsit stele pentru acest nivel Bortle.")
            update_app_setting("bortle", bortle_scale)
