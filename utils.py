# utils.py
# Versiunea finală care conține toate funcțiile de ajutor pentru aplicație.

import streamlit as st
import lightkurve as lk
import matplotlib.pyplot as plt
import astropy.time
from astropy import units as u
from astropy.timeseries import BoxLeastSquares
from astropy import constants as const
import time
import pandas as pd
import numpy as np
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
from astroquery.skyview import SkyView
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
import traceback # Folosit pentru a afișa erori detaliate, deși este eliminat din funcția principală

# --- Funcții pentru pagina TESS Planet Search ---

@st.cache_data(ttl="1d")
def get_toi_catalog():
    """Descarcă și stochează în cache catalogul complet TESS Objects of Interest (TOI)."""
    toi_url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    try:
        df = pd.read_csv(toi_url)
        return df
    except Exception as e:
        st.error(f"Eroare la descărcarea catalogului TESS TOI: {e}")
        return pd.DataFrame()

def search_toi_catalog(dispositions=None, radius_range=None, period_range=None, tic_id=None):
    """
    Filtrează catalogul TOI stocat în cache pe baza criteriilor utilizatorului.
    """
    df = get_toi_catalog()
    if df.empty:
        return df

    # Se face o copie pentru a evita modificarea datelor din cache
    filtered_df = df.copy()

    # Se aplică filtrele
    if dispositions:
        filtered_df = filtered_df[filtered_df['TFOPWG Disposition'].isin(dispositions)]
    
    if radius_range:
        if 'Planet Radius (R_earth)' in filtered_df.columns:
            filtered_df['Planet Radius (R_earth)'] = pd.to_numeric(filtered_df['Planet Radius (R_earth)'], errors='coerce')
            filtered_df.dropna(subset=['Planet Radius (R_earth)'], inplace=True)
            filtered_df = filtered_df[
                (filtered_df['Planet Radius (R_earth)'] >= radius_range[0]) &
                (filtered_df['Planet Radius (R_earth)'] <= radius_range[1])
            ]

    if period_range:
        if 'Perioadă (zile)' in filtered_df.columns:
            filtered_df['Perioadă (zile)'] = pd.to_numeric(filtered_df['Perioadă (zile)'], errors='coerce')
            filtered_df.dropna(subset=['Perioadă (zile)'], inplace=True)
            filtered_df = filtered_df[
                (filtered_df['Perioadă (zile)'] >= period_range[0]) &
                (filtered_df['Perioadă (zile)'] <= period_range[1])
            ]

    if tic_id:
        if 'TIC ID' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['TIC ID'] == tic_id]

    # Se selectează și se redenumesc coloanele pentru o afișare mai curată
    column_map = {
        'TIC ID': 'TIC ID',
        'TOI': 'Nume TOI',
        'TFOPWG Disposition': 'Status TFOPWG',
        'Perioadă (zile)': 'Orbital Period (days)',
        'Planet Radius (R_earth)': 'Planet Radius (R_earth)',
        'Planet Temp (K)': 'Planet Temp (K)',
        'Stellar Teff (K)': 'Stellar Teff (K)',
        'Stellar Radius (R_sun)': 'Stellar Radius (R_sun)'
    }
    
    existing_cols = [col for col in column_map.keys() if col in filtered_df.columns]
    final_df = filtered_df[existing_cols].rename(columns=column_map)

    return final_df.reset_index(drop=True)


# --- Funcție pentru pagina Star Explorer ---

@st.cache_data(ttl="7d")
def fetch_star_data(star_name):
    """
    Preia inteligent datele despre stele. Dacă este furnizat un ID TIC, interoghează direct TIC.
    Altfel, folosește Simbad pentru a rezolva numele într-un ID TIC.
    """
    facts = {}
    tic_id = None
    resolved_name_for_planets = star_name

    # Pasul 1: Verifică dacă intrarea este un ID TIC sau un nume comun
    cleaned_input = star_name.upper().replace("TIC", "").strip()
    if cleaned_input.isdigit():
        tic_id = cleaned_input
    else:
        # Dacă nu este un ID TIC, folosește Simbad pentru a rezolva numele
        try:
            simbad = Simbad()
            simbad.add_votable_fields('ids')
            simbad_result = simbad.query_object(star_name)

            if simbad_result is None:
                return {"error": f"Nu s-a putut găsi '{star_name}' în baza de date Simbad."}
            if 'IDS' not in simbad_result.columns:
                return {"error": f"Nu s-au putut prelua identificatorii de catalog pentru '{star_name}'."}

            all_ids = simbad_result['IDS'][0].decode('utf-8')
            for identifier in all_ids.split('|'):
                if 'TIC' in identifier:
                    tic_id = identifier.replace("TIC", "").strip()
                    break
            if tic_id is None:
                return {"error": f"S-a găsit '{star_name}', dar nu s-a putut găsi un ID TESS Input Catalog (TIC) corespunzător."}
        except Exception as e:
            return {"error": f"A apărut o eroare la interogarea Simbad: {e}"}

    # Pasul 2: Interoghează Catalogul TESS folosind ID-ul TIC
    try:
        star_data = Catalogs.query_criteria(catalog="TIC", ID=int(tic_id))
        
        if len(star_data) == 0:
            return {"error": f"Nu s-au putut prelua date pentru TIC {tic_id} din catalogul MAST."}

        # Populează faptele din catalogul TESS
        facts['tic_id'] = tic_id
        facts['name'] = star_name
        facts['ra'] = star_data['ra'][0]
        facts['dec'] = star_data['dec'][0]
        
        facts['Tmag'] = f"{star_data['Tmag'][0]:.2f}" if 'Tmag' in star_data.columns and not np.ma.is_masked(star_data['Tmag'][0]) else "N/A"
        facts['Teff'] = f"{star_data['Teff'][0]:.0f}" if 'Teff' in star_data.columns and not np.ma.is_masked(star_data['Teff'][0]) else "N/A"
        facts['radius'] = f"{star_data['rad'][0]:.2f}" if 'rad' in star_data.columns and not np.ma.is_masked(star_data['rad'][0]) else "N/A"
        facts['mass'] = f"{star_data['mass'][0]:.2f}" if 'mass' in star_data.columns and not np.ma.is_masked(star_data['mass'][0]) else "N/A"
        
        if 'plx' in star_data.columns and not np.ma.is_masked(star_data['plx'][0]) and star_data['plx'][0] > 0:
            parallax = star_data['plx'][0]
            distance_pc = 1000 / parallax
            facts['distance_pc'] = f"{distance_pc:.2f}"
            facts['distance_ly'] = f"{distance_pc * 3.26156:.2f}"
        else:
            facts['distance_pc'] = "N/A"
            facts['distance_ly'] = "N/A"

    except Exception as e:
        return {"error": f"A apărut o eroare la interogarea catalogului TESS: {e}"}

    # Pasul 3: Interoghează arhive suplimentare
    
    # Arhiva Exoplanetelor NASA pentru planete CONFIRMATE
    try:
        planet_data = NasaExoplanetArchive.query_criteria(table="pscomppars", where=f"hostname = '{resolved_name_for_planets}'")
        facts['confirmed_planet_count'] = len(planet_data)
        if facts['confirmed_planet_count'] > 0:
            df = planet_data.to_pandas()
            columns_to_show = {'pl_name': 'Nume planetă', 'pl_orbper': 'Perioadă (zile)', 'pl_rade': 'Rază (raze terestre)', 'pl_masse': 'Masă (mase terestre)', 'discoverymethod': 'Metodă de descoperire', 'disc_year': 'Anul descoperirii'}
            available_cols = {k: v for k, v in columns_to_show.items() if k in df.columns}
            facts['planet_df'] = df[list(available_cols.keys())].rename(columns=available_cols)
        else:
            facts['planet_df'] = None
    except Exception:
        facts['confirmed_planet_count'] = 0
        facts['planet_df'] = None

    # Catalogul TOI pentru candidați și confirmați TESS
    try:
        toi_catalog = get_toi_catalog()
        if not toi_catalog.empty:
            star_in_toi = toi_catalog[toi_catalog['TIC ID'] == int(tic_id)]
            facts['toi_pc_count'] = len(star_in_toi[star_in_toi['TFOPWG Disposition'] == 'PC'])
            facts['toi_cp_count'] = len(star_in_toi[star_in_toi['TFOPWG Disposition'] == 'CP'])
        else:
            facts['toi_pc_count'] = 0
            facts['toi_cp_count'] = 0
    except Exception:
        facts['toi_pc_count'] = "N/A"
        facts['toi_cp_count'] = "N/A"
        
    # SkyView pentru imagini
    try:
        image_urls = SkyView.get_image_links(position=f"{facts['ra']} {facts['dec']}", survey=['DSS2 Red'])
        facts['image_url'] = image_urls[0] if image_urls else None
    except Exception:
        facts['image_url'] = None
        
    return facts


# --- Funcții pentru pagina "Caută o stea" ---

@st.cache_data(ttl="7d")
def get_star_parameters(target_id):
    """Interoghează catalogul TIC pentru raza stelei."""
    try:
        numeric_id = target_id.upper().replace("TIC", "").strip()
        if not numeric_id.isdigit():
            return None
        star_data = Catalogs.query_criteria(catalog="TIC", ID=int(numeric_id))
        if len(star_data) == 0 or 'rad' not in star_data.columns:
            st.warning(f"Nu s-a putut găsi raza pentru {target_id} în catalog.")
            return None
        star_radius_solar = star_data['rad'][0]
        if not np.isfinite(star_radius_solar) or star_radius_solar <= 0:
            st.warning(f"Catalogul conține o rază invalidă pentru {target_id}.")
            return None
        return star_radius_solar
    except Exception as e:
        st.warning(f"Eroare la preluarea parametrilor stelari: {e}")
        return None

def process_selected_data(selected_items, bin_minutes, outlier_sigma, period_min, period_max):
    """
    Descarcă, procesează, randează fișiere FITS și analizează datele curbei de lumină.
    """
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    try:
        search_result_df = selected_items.table.to_pandas()
        target_id = search_result_df['target_name'][0]
        star_radius_solar = get_star_parameters(target_id)
        
        total_files = len(selected_items)
        st.header("Step 3: Analysis Results")
        status_placeholder.info(f"✅ Începe analiza pe {total_files} produse de date selectate...")
        progress_bar = st.progress(0)
        
        processed_light_curves = []
        downloaded_tpfs = []  # Listă pentru a stoca fișierele TPF descărcate

        for i in range(total_files):
            status_text = f"⬇️ Se descarcă și se pregătește fișierul {i + 1} din {total_files}..."
            status_placeholder.info(status_text)
            
            data_product = selected_items[i].download()

            if data_product is None:
                st.write(f"Se omite fișierul {i + 1} (gol sau invalid).")
                progress_bar.progress((i + 1) / total_files)
                continue

            # Verifică dacă este un Target Pixel File (TPF)
            if isinstance(data_product, lk.targetpixelfile.TargetPixelFile):
                downloaded_tpfs.append(data_product)
                lc = data_product.to_lightcurve(aperture_mask='pipeline').remove_nans()
            # Verifică dacă este un Light Curve File
            elif isinstance(data_product, lk.lightcurve.LightCurve):
                lc = data_product.remove_nans()
            else:
                st.write(f"Se omite fișierul {i + 1} (tip de date neacceptat).")
                progress_bar.progress((i + 1) / total_files)
                continue

            if len(lc.flux) > 0:
                normalized_lc = lc.normalize()
                processed_light_curves.append(normalized_lc)
            
            progress_bar.progress((i + 1) / total_files)

        if not processed_light_curves:
            st.error("Nu s-au putut procesa sau extrage date valide ale curbei de lumină din selecție.")
            status_placeholder.empty(); progress_placeholder.empty()
            return

        # Randează fișierele TPF descărcate
        if downloaded_tpfs:
            st.subheader("Rendered Target Pixel Files (FITS)")
            st.info("Aceste imagini arată pixelii observați de telescop. Conturul roșu este 'masca de apertură' — pixelii folosiți pentru a măsura luminozitatea stelei.")
            
            num_cols = min(len(downloaded_tpfs), 3)
            cols = st.columns(num_cols)
            for idx, tpf in enumerate(downloaded_tpfs):
                with cols[idx % num_cols]:
                    st.markdown(f"**Observația {idx + 1}** (`{tpf.mission}`)")
                    fig, ax = plt.subplots(figsize=(5,5))
                    tpf.plot(ax=ax, aperture_mask='pipeline', mask_color='red')
                    st.pyplot(fig)
                    plt.close(fig)
            st.divider()

        # Restul analizei continuă
        status_placeholder.info("⚙️ Se unesc segmentele curbei de lumină...")
        final_collection = lk.LightCurveCollection(processed_light_curves)
        lc = final_collection.stitch().remove_nans()
        time.sleep(1)

        status_placeholder.info("🧹 Se curăță și se aplatizează curba de lumină...")
        progress_placeholder.empty()

        bin_size_in_days = bin_minutes / 1440.0
        binned_lc = lc.bin(time_bin_size=bin_size_in_days * u.day)
        
        clean_lc = binned_lc.flatten().remove_outliers(sigma=outlier_sigma)

        status_placeholder.info("🔍 Se caută semnale periodice folosind Astropy...")
        
        time_vals = clean_lc.time.value
        flux_vals = clean_lc.flux.value
        
        model = BoxLeastSquares(t=time_vals, y=flux_vals)
        period_grid = np.arange(period_min, period_max, 0.01)
        results = model.power(period_grid, duration=0.1)
        
        index = np.argmax(results.power)
        
        planet_period_val = results.period[index]
        planet_t0_val = results.transit_time[index]
        planet_duration_val = results.duration[index]
        planet_depth_val = results.depth[index]
        
        planet_period = planet_period_val * u.day
        planet_duration = planet_duration_val * u.day
        planet_t0 = astropy.time.Time(planet_t0_val, format=clean_lc.time.format, scale=clean_lc.time.scale)

        st.success(f"Cel mai puternic semnal găsit la o perioadă de: **{planet_period.value:.4f} zile**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Durata Tranzitului", value=f"{planet_duration.value:.4f} zile",
                      help="Timpul estimat pe care planeta îl petrece trecând prin fața stelei.")
        with col2:
            st.metric(label="Adâncimea Tranzitului", value=f"{planet_depth_val:.4f}",
                      help="Cu cât scade luminozitatea stelei în timpul tranzitului (flux normalizat).")
        if star_radius_solar:
            planet_radius_solar = star_radius_solar * np.sqrt(planet_depth_val)
            planet_radius_earth = (planet_radius_solar * u.R_sun).to(u.R_earth).value
            with col3:
                st.metric(label="Raza Est. a Planetei", value=f"{planet_radius_earth:.2f} Pământuri",
                          help=f"Raza estimată a planetei, presupunând o rază stelară de {star_radius_solar:.2f} ori cea a Soarelui.")
        else:
            with col3:
                st.info("Raza planetei nu a putut fi estimată (raza stelei nu este disponibilă).")

        st.subheader("Detectarea Semnalului Periodic (Periodograma)")
        st.markdown(
            "O **periodogramă** este un grafic care ajută la găsirea semnalelor repetitive. Algoritmul caută scăderi de luminozitate la mii de perioade orbitale posibile. Vârful cel mai înalt indică cea mai probabilă perioadă orbitală.")
        fig2, ax2 = plt.subplots()
        ax2.plot(results.period, results.power)
        ax2.set_xlabel(f"Perioada ({u.day.to_string('latex')})")
        ax2.set_ylabel("Putere (Power)")
        ax2.axvline(planet_period.value, color='red', linestyle='--', alpha=0.7)
        st.pyplot(fig2)
        
        status_placeholder.info("🌟 Se pliază curba de lumină pentru a dezvălui tranzitul...")
        folded_lc = clean_lc.fold(period=planet_period, epoch_time=planet_t0)

        st.subheader("Curba de Lumină Curățată vs. Pliată")
        st.markdown(
            "**Curba de Lumină Curățată** (stânga) arată datele complete de luminozitate după eliminarea zgomotului. **Curba de Lumină Pliată** (dreapta) este dovada cea mai importantă; toate datele sunt suprapuse în funcție de perioada găsită. O scădere clară în formă de 'U' este un indiciu puternic al unei potențiale planete.")
        fig1, ax1 = plt.subplots()
        clean_lc.plot(ax=ax1, ylabel="Flux Normalizat", label="Curățată și Aplatizată")
        fig3, ax3 = plt.subplots()
        folded_lc.plot(ax=ax3, label=f"Pliată la {planet_period.value:.4f} zile")
        plot_col1, plot_col2 = st.columns(2)
        with plot_col1:
            st.pyplot(fig1)
        with plot_col2:
            st.pyplot(fig3)
        status_placeholder.success("🎉 Analiză completă!")
        
    except Exception as e:
        st.error(f"A apărut o eroare neașteptată în timpul analizei: {e}")
        tb_str = traceback.format_exc()
        st.subheader("Urmărire Tehnică (Traceback)")
        st.code(tb_str, language='text')
        status_placeholder.empty(); progress_placeholder.empty()


# --- Funcții pentru paginile "Explore" ---

@st.cache_data(ttl="1d")
def fetch_catalog_targets(mission_name, disposition_type, num_targets=25):
    """Preia un eșantion de ținte din cataloagele TESS sau Kepler/K2."""
    try:
        url, id_col, prefix, disposition_col, dispositions_to_find = "", "", "", "", []
        if mission_name == "TESS":
            url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
            id_col, prefix, disposition_col = 'TIC ID', 'TIC ', 'TFOPWG Disposition'
            if disposition_type == "PLANETS": dispositions_to_find = ["CP", "PC"]
            else: dispositions_to_find = ["FP"]
        elif mission_name in ["Kepler", "K2"]:
            url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&select=kepid,koi_disposition,koi_period,koi_prad&format=csv"
            id_col, prefix, disposition_col = 'kepid', 'KIC ', 'koi_disposition'
            if disposition_type == "PLANETS": dispositions_to_find = ["CONFIRMED", "CANDIDATE"]
            else: dispositions_to_find = ["FALSE POSITIVE"]
        else:
            return pd.DataFrame()
        
        comment_char = '#' if "kepler" in url.lower() else None
        catalog_df = pd.read_csv(url, comment=comment_char)
        filtered_df = catalog_df[catalog_df[disposition_col].isin(dispositions_to_find)]
        if filtered_df.empty: return pd.DataFrame()
        
        final_sample = filtered_df.sample(n=min(num_targets, len(filtered_df)))
        column_map = {
            id_col: "Searchable ID", disposition_col: "Status",
            'Perioadă (zile)': 'Orbital Period (days)', 'koi_period': 'Orbital Period (days)',
            'Planet Radius (R_earth)': 'Planet Radius (Earths)', 'koi_prad': 'Planet Radius (Earths)'
        }
        available_cols = [col for col in column_map.keys() if col in final_sample.columns]
        result_df = final_sample[available_cols].rename(columns=column_map)
        result_df["Searchable ID"] = prefix + result_df["Searchable ID"].astype(str)
        return result_df
    except Exception as e:
        st.warning(f"Nu s-au putut prelua țintele dinamic: {e}")
        return pd.DataFrame()


# --- Funcție pentru pagina "Find Untested Targets" ---

@st.cache_data(ttl="1d")
def fetch_untested_targets(num_to_sample=100):
    """Preia un eșantion de stele care nu sunt încă TOI-uri cunoscute."""
    try:
        toi_url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
        toi_df = pd.read_csv(toi_url)
        known_toi_tics = set(toi_df['TIC ID'])
        
        tic_sample = Catalogs.query_criteria(
            catalog="TIC", Vmag=(9, 13), plx=(2, 100), pagesize=num_to_sample)
        tic_sample_df = tic_sample.to_pandas()
        
        untested_mask = ~tic_sample_df['ID'].isin(known_toi_tics)
        untested_df = tic_sample_df[untested_mask]
        
        untested_df = untested_df.rename(columns={'ID': 'Searchable ID', 'Tmag': 'TESS Magnitude', 'dst': 'Distance (pc)'})
        untested_df['Searchable ID'] = "TIC " + untested_df['Searchable ID'].astype(str)
        
        final_columns = ['Searchable ID', 'TESS Magnitude', 'Distance (pc)', 'ra', 'dec']
        available_final_cols = [col for col in final_columns if col in untested_df.columns]
        return untested_df[available_final_cols]
    except Exception as e:
        st.warning(f"Nu s-au putut prelua țintele netestate: {e}")
        return pd.DataFrame()
