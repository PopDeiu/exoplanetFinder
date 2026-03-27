# pages/6_Star_explorer.py

import streamlit as st
import numpy as np
from utils import fetch_star_data, set_galaxy_background, set_sidebar_style
from astroquery.skyview import SkyView
from astropy.wcs import WCS

st.set_page_config(
    page_title="Explorator de stele",
    layout="wide",
    initial_sidebar_state="expanded"
)

PIXELS_DEFAULT = 600  # dimensiune fixă pentru imaginea SkyView

# --- Aplicare styling IMEDIAT ---
set_sidebar_style()
set_galaxy_background("cosmic")

st.header("Explorator de stele")
st.caption(
    "Pasul 4 din 4 · Explorează în detaliu proprietățile unei stele și ale planetelor ei cunoscute."
)
st.markdown(
    "Introdu numele unei stele pentru a-i obține datele din TESS Input Catalog (TIC) "
    "și din alte arhive astronomice (NASA Exoplanet Archive, ExoFOP, SkyView)."
)


# =========================
# Helper: imagine SkyView
# =========================
def fetch_skyview_image(position: str, survey: str, pixels: int = PIXELS_DEFAULT):
    """
    Întoarce (img_data, wcs) pentru imaginea SkyView sau (None, None) dacă nu reușește.
    Forțează copierea datelor în RAM pentru a evita eroarea de fișier închis.
    """
    try:
        images = SkyView.get_images(
            position=position,
            survey=[survey],
            pixels=pixels,
        )
        if images:
            # images[0] este un HDUList (o listă de unități de date astronomice)
            hdu_list = images[0]
            hdu = hdu_list[0]
            
            # --- MODIFICARE CRITICĂ: .copy() ---
            # Copiem datele și header-ul în RAM pentru a rupe legătura cu fișierul temporar
            data = hdu.data.copy()
            header = hdu.header.copy()
            
            try:
                wcs = WCS(header)
            except Exception:
                wcs = None
            
            # Închidem explicit lista de HDU pentru a elibera resursele
            hdu_list.close()
            
            return data, wcs
            
    except Exception as e:
        # Folosim st.error sau logging pentru a vedea eroarea exactă
        st.warning(f"Nu s-a putut descărca imaginea din SkyView ({survey}): {e}")
    
    return None, None


def fetch_skyview_image_vechi(position: str, survey: str, pixels: int = PIXELS_DEFAULT):
    """
    Întoarce (img_data, wcs) pentru imaginea SkyView sau (None, None) dacă nu reușește.
    """
    try:
        images = SkyView.get_images(
            position=position,
            survey=[survey],
            pixels=pixels,
        )
        if images:
            hdu_list = images[0]
            hdu = hdu_list[0]
            data = hdu.data
            header = hdu.header
            try:
                wcs = WCS(header)
            except Exception:
                wcs = None
            return data, wcs
    except Exception as e:
        st.warning(f"Nu s-a putut descărca imaginea din SkyView ({survey}): {e}")
    return None, None


# =========================
# Input: nume / TIC
# =========================
star_name_input = st.text_input(
    label="Introdu un nume de stea sau un ID TIC",
    value=st.session_state.get("star_name_input", "TRAPPIST-1"),
    key="star_name_input",
    help=(
        "Poți introduce nume precum „Kepler-10”, „TRAPPIST-1” sau un ID TIC "
        "de forma „TIC 261136679”."
    ),
)

st.markdown("---")

# =========================
# Căutare stea
# =========================
cauta = st.button("Caută steaua")

if cauta:
    star_name = st.session_state.get("star_name_input", "").strip()
    if not star_name:
        st.warning("Te rog introdu un nume de stea sau un ID TIC pentru a începe.")
    else:
        with st.status(f"Se caută „{star_name}”...", expanded=True) as status:
            try:
                # 1) Date catalog (TESS + exoplanete)
                data = fetch_star_data(star_name)
                st.session_state["star_data"] = data

                # 2) Imagine implicită DSS2 IR
                img_data, wcs = fetch_skyview_image(
                    position=star_name,
                    survey="DSS2 IR",
                    pixels=PIXELS_DEFAULT,
                )
                st.session_state["skyview_img"] = img_data
                st.session_state["skyview_wcs"] = wcs
                st.session_state["skyview_survey"] = "DSS2 IR"

                if data and "error" not in data:
                    tic_id = data.get("tic_id", "N/A")
                    status.update(
                        label=f"S-a găsit TIC {tic_id}! Căutare finalizată.",
                        state="complete",
                        expanded=False,
                    )
                else:
                    error_message = data.get(
                        "error", "A apărut o eroare necunoscută la preluarea datelor."
                    )
                    status.update(
                        label=f"Căutarea a eșuat: {error_message}",
                        state="error",
                        expanded=True,
                    )
            except Exception as e:
                st.error(f"A apărut o eroare neașteptată: {e}")


# =========================
# Afișare rezultate
# =========================
data = st.session_state.get("star_data")

if data:
    st.divider()

    if "error" in data:
        st.error(data["error"])
    else:
        confirmed_planets = data.get("confirmed_planet_count", 0)
        star_display_name = data.get("name") or st.session_state.get(
            "star_name_input", "Stea necunoscută"
        )
        tic_display = data.get("tic_id", "N/A")

        st.subheader(f"Date pentru {star_display_name} (TIC {tic_display})")

        col1, col2 = st.columns([1.4, 1.6])

        # ======================================
        # COL 1: Imagine SkyView stabilă
        # ======================================
        with col1:
            st.subheader("Imagine SkyView")

            survey_options = [
                "DSS2 Red",
                "DSS2 Blue",
                "DSS2 IR",
                "2MASS-K",
                "GALEXGR6-AIS",
            ]

            default_survey = st.session_state.get("skyview_survey", "DSS2 IR")
            if default_survey not in survey_options:
                default_survey = "DSS2 IR"

            survey = st.selectbox(
                "Survey",
                survey_options,
                index=survey_options.index(default_survey),
                help="Alege survey-ul (optical, IR, UV etc.).",
            )

            # Contrast
            vmin_pct, vmax_pct = st.slider(
                "Contrast (percentile)",
                0,
                100,
                (1, 99),
                help="Ajustează contrastul imaginii.",
            )

            # Inversare culori
            invert_colors = st.checkbox(
                "Inversează culorile (negative)",
                value=False,
                help="Util pentru evidențierea detaliilor.",
            )

            # Marcaj pe țintă
            show_crosshair = st.checkbox(
                "Arată marcaj verde pe steaua țintă (centru)",
                value=True,
            )

            # Actualizare imagine la schimbarea survey-ului
            if st.button("📷 Reîncarcă imaginea SkyView", key="update_image_btn"):
                img_data, wcs = fetch_skyview_image(
                    position=star_display_name,
                    survey=survey,
                    pixels=PIXELS_DEFAULT,
                )
                st.session_state["skyview_img"] = img_data
                st.session_state["skyview_wcs"] = wcs
                st.session_state["skyview_survey"] = survey

            img_raw = st.session_state.get("skyview_img")
            wcs = st.session_state.get("skyview_wcs")

            if img_raw is not None:
                finite = img_raw[np.isfinite(img_raw)]
                if finite.size > 0:
                    if vmin_pct >= vmax_pct:
                        vmin_pct, vmax_pct = 1, 99
                    vmin, vmax = np.percentile(finite, [vmin_pct, vmax_pct])
                    if vmax > vmin:
                        img_clipped = np.clip(img_raw, vmin, vmax)
                        img_norm = (img_clipped - vmin) / (vmax - vmin)
                    else:
                        img_norm = np.zeros_like(img_raw, dtype=float)
                else:
                    img_norm = np.zeros_like(img_raw, dtype=float)

                # Inversare culori
                if invert_colors:
                    img_norm = 1.0 - img_norm

                ny, nx = img_norm.shape

                # Construim imagine RGB pentru a marca crucea cu verde
                rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)

                if show_crosshair:
                    cy, cx = ny // 2, nx // 2
                    half_len = min(ny, nx) // 20
                    y1, y2 = max(cy - half_len, 0), min(cy + half_len, ny - 1)
                    x1, x2 = max(cx - half_len, 0), min(cx + half_len, nx - 1)

                    # verde: (R=0, G=1, B=0)
                    rgb[y1:y2, cx, 0] = 0.0
                    rgb[y1:y2, cx, 1] = 1.0
                    rgb[y1:y2, cx, 2] = 0.0

                    rgb[cy, x1:x2, 0] = 0.0
                    rgb[cy, x1:x2, 1] = 1.0
                    rgb[cy, x1:x2, 2] = 0.0

                st.image(
                    rgb,
                    caption=f"SkyView – {star_display_name} – {st.session_state.get('skyview_survey', survey)}",
                    use_container_width=True,
                    clamp=True,
                )

                # RA/Dec ale centrului imaginii (steaua țintă)
                if wcs is not None:
                    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
                    ra_deg, dec_deg = wcs.wcs_pix2world([[cx, cy]], 0)[0]
                    st.caption(
                        f"Poziția aproximativă a stelei țintă (centru imagine): "
                        f"RA = {ra_deg:.5f}°, Dec = {dec_deg:.5f}°"
                    )
                else:
                    st.caption(
                        "Coordonatele RA/Dec nu sunt disponibile (WCS lipsă în header)."
                    )

            else:
                st.info("Nu s-a putut obține imaginea din SkyView pentru această stea.")

        # ======================================
        # COL 2: Date TESS + exoplanete + info extinsă
        # ======================================
        with col2:
            st.subheader("Date din catalogul TESS")

            # câmpurile „clasice”
            Tmag = data.get("Tmag", "N/A")
            Teff = data.get("Teff", "N/A")
            radius = data.get("radius", "N/A")
            mass = data.get("mass", "N/A")
            distance_ly = data.get("distance_ly", "N/A")
            toi_pc_count = data.get("toi_pc_count", 0)
            toi_cp_count = data.get("toi_cp_count", 0)

            # câmpuri noi (de populat în fetch_star_data)
            luminosity = data.get("luminosity", "N/A")              # în L☉
            metallicity = data.get("metallicity", "N/A")            # [Fe/H]
            ra_exact = data.get("ra", "N/A")                        # RA J2000
            dec_exact = data.get("dec", "N/A")                      # Dec J2000
            tess_sectors = data.get("tess_sectors", [])            # listă de sectoare TESS
            hz_inner = data.get("hz_inner_au", None)                # margine interioară HZ (AU)
            hz_outer = data.get("hz_outer_au", None)                # margine exterioară HZ (AU)
            spectral_type = data.get("spectral_type", "N/A")        # ex. "K2V"
            evolution_class = data.get("evolution_class", "N/A")    # ex. "pitică", "subgigantă", "gigantă"
            rotation_period = data.get("rotation_period_days", "N/A")  # în zile

            # helper pentru conversie numerică
            def to_float(x):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return None

            Teff_num = to_float(Teff)
            radius_num = to_float(radius)
            mass_num = to_float(mass)
            dist_num = to_float(distance_ly)
            hz_inner_num = to_float(hz_inner)
            hz_outer_num = to_float(hz_outer)
            rot_num = to_float(rotation_period)

            # sectoare TESS
            if isinstance(tess_sectors, (list, tuple, set)):
                tess_sectors_str = ", ".join(str(s) for s in sorted(tess_sectors))
            else:
                tess_sectors_str = (
                    str(tess_sectors) if tess_sectors not in (None, "", []) else "N/A"
                )

            # zonă locuibilă
            if hz_inner_num is not None and hz_outer_num is not None:
                hz_zone_str = f"{hz_inner_num:.2f}–{hz_outer_num:.2f} AU"
            else:
                hz_zone_str = "N/A"

            # rotație
            if rot_num is not None:
                rotation_str = f"{rot_num:.2f} zile"
            else:
                rotation_str = "N/A"

            st.markdown(
                f"""
| Parametru | Valoare | Descriere |
| :--- | :--- | :--- |
| **Magnitudine TESS** | `{Tmag}` | Strălucirea stelei în banda TESS |
| **Temperatură efectivă** | `{Teff} K` | Temperatura efectivă aproximativă |
| **Rază** | `{radius} raze solare` | Dimensiune relativă față de Soare |
| **Masă** | `{mass} mase solare` | Masă relativă față de Soare |
| **Luminozitate** | `{luminosity}` | Luminozitate în unități solare (L☉) |
| **Metallicitate [Fe/H]** | `{metallicity}` | Compoziție chimică relativă față de Soare |
| **RA (J2000)** | `{ra_exact}` | Ascensie dreaptă exactă |
| **Dec (J2000)** | `{dec_exact}` | Declinație exactă |
| **Distanță** | `{distance_ly} ani-lumină` | Distanță aproximativă față de Pământ |
| **Sectoare TESS observate** | `{tess_sectors_str}` | Sectoare în care TESS a observat steaua |
| **Zonă locuibilă estimată** | `{hz_zone_str}` | Interval aproximativ al zonei locuibile (AU) |
| **Tip spectral estimat** | `{spectral_type}` | Tip spectral derivat din datele fotometrice |
| **Stare evolutivă** | `{evolution_class}` | Pitică / subgigantă / gigantă (estimativ) |
| **Perioadă de rotație** | `{rotation_str}` | Perioada de rotație a stelei, dacă este cunoscută |
| **Planete confirmate** | `{confirmed_planets}` | Din arhiva principală de exoplanete NASA |
| **Candidați TESS** | `{toi_pc_count}` | TESS Objects of Interest (PC) |
| **Confirmate TESS** | `{toi_cp_count}` | TESS Confirmed Planets (CP) |
"""
            )

            st.markdown("### Analiză pe scurt")

            bullets = []

            # Interpretare temperatură
            if Teff_num:
                if Teff_num < 3800:
                    bullets.append(
                        f"- Stea **rece**, probabil de tip târziu K sau M (Teff ≈ {Teff_num:.0f} K)."
                    )
                elif Teff_num < 5200:
                    bullets.append(
                        f"- Stea **mai rece decât Soarele**, probabil de tip K (Teff ≈ {Teff_num:.0f} K)."
                    )
                elif Teff_num < 6200:
                    bullets.append(
                        f"- Stea de tip **similar cu Soarele** (tip G, Teff ≈ {Teff_num:.0f} K)."
                    )
                elif Teff_num < 7500:
                    bullets.append(
                        f"- Stea **mai fierbinte decât Soarele**, probabil de tip F (Teff ≈ {Teff_num:.0f} K)."
                    )
                else:
                    bullets.append(
                        f"- Stea **foarte fierbinte** (tip A sau mai devreme, Teff ≈ {Teff_num:.0f} K)."
                    )

            # Dimensiune
            if radius_num:
                if radius_num < 0.8:
                    bullets.append(
                        f"- Este mai **mică decât Soarele** (≈ {radius_num:.2f} raze solare)."
                    )
                elif radius_num <= 1.2:
                    bullets.append(
                        f"- Are o rază **comparabilă cu a Soarelui** (≈ {radius_num:.2f} raze solare)."
                    )
                else:
                    bullets.append(
                        f"- Este o stea **mai mare decât Soarele** (≈ {radius_num:.2f} raze solare)."
                    )

            # Masă
            if mass_num:
                if mass_num < 0.8:
                    bullets.append(
                        f"- Masa este **sub-solară** (≈ {mass_num:.2f} mase solare) – stea mai puțin masivă."
                    )
                elif mass_num <= 1.2:
                    bullets.append(
                        f"- Masa este **similară cu a Soarelui** (≈ {mass_num:.2f} mase solare)."
                    )
                else:
                    bullets.append(
                        f"- Masa este **peste cea a Soarelui** (≈ {mass_num:.2f} mase solare) – stea mai masivă."
                    )

            # Distanță
            if dist_num:
                if dist_num < 50:
                    bullets.append(
                        f"- Se află **relativ aproape** de noi, la ≈ {dist_num:.1f} ani-lumină."
                    )
                elif dist_num < 300:
                    bullets.append(
                        f"- Se află la o **distanță moderată** (≈ {dist_num:.1f} ani-lumină)."
                    )
                else:
                    bullets.append(
                        f"- Se află **destul de departe** (≈ {dist_num:.1f} ani-lumină)."
                    )

            # Zonă locuibilă
            if hz_inner_num is not None and hz_outer_num is not None:
                bullets.append(
                    f"- Zona locuibilă estimată se află între **{hz_inner_num:.2f} și {hz_outer_num:.2f} AU**; "
                    "planete în acest interval ar putea avea condiții pentru apă lichidă la suprafață."
                )

            # Sistem planetar
            if confirmed_planets > 0:
                bullets.append(
                    f"- Steaua are **{confirmed_planets} planetă(e) confirmată(e)** în arhiva NASA de exoplanete."
                )
            elif toi_pc_count or toi_cp_count:
                bullets.append(
                    "- Steaua are **candidați de exoplanete în TESS**, dar fără planete confirmate încă."
                )
            else:
                bullets.append(
                    "- Până în prezent **nu există planete confirmate** raportate pentru această stea."
                )

            # Tip spectral / stare evolutivă / rotație
            if spectral_type not in (None, "", "N/A"):
                bullets.append(f"- Tipul spectral estimat este **{spectral_type}**.")
            if evolution_class not in (None, "", "N/A"):
                bullets.append(f"- Starea evolutivă este aproximată ca **{evolution_class}**.")
            if rot_num is not None:
                bullets.append(
                    f"- Perioada de rotație estimată este de ≈ **{rot_num:.1f} zile**."
                )

            if not bullets:
                bullets.append(
                    "- Nu există suficiente informații numerice pentru o interpretare detaliată."
                )

            for b in bullets:
                st.markdown(b)

            st.success(
                "Această secțiune îți oferă o privire rapidă, în limbaj uman, asupra stelei țintă."
            )

        # ======================================
        # Tabel planete confirmate
        # ======================================
        if confirmed_planets > 0 and data.get("planet_df") is not None:
            st.divider()
            st.subheader(f"Planete confirmate ({confirmed_planets})")

            st.dataframe(
                data["planet_df"],
                use_container_width=True,
                hide_index=True,
            )

else:
    st.info(
        "Introdu un nume de stea și apasă „🔍 Caută steaua” pentru a vedea informații și imaginea SkyView."
    )

