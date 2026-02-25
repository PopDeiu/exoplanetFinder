import pandas as pd
import streamlit as st
import numpy as np
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
from astroquery.skyview import SkyView
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

# --- FUNCȚII CATALOG TOI (TESS Objects of Interest) ---

@st.cache_data(ttl="1d")
def get_toi_catalog():
    """Descarcă catalogul TESS Objects of Interest (TOI). Cache 24h."""
    toi_url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    try:
        return pd.read_csv(toi_url)
    except Exception as e:
        st.error(f"Eroare la descărcarea catalogului ExoFOP: {e}")
        return pd.DataFrame()

def search_toi_catalog(dispositions=None, radius_range=None, period_range=None, tic_id=None):
    """Filtrează catalogul TOI cu detecție automată a coloanelor."""
    df = get_toi_catalog()
    if df.empty:
        return df

    filtered_df = df.copy()

    # Identificăm numele coloanelor (NASA le modifică periodic)
    disp_col = next((c for c in ['TFOPWG Disposition', 'Disposition'] if c in filtered_df.columns), None)
    rad_col = next((c for c in ['Planet Radius (R_earth)', 'Radius (R_earth)'] if c in filtered_df.columns), None)
    per_col = next((c for c in ['Orbital Period (days)', 'Period (days)'] if c in filtered_df.columns), None)

    if dispositions and disp_col:
        filtered_df = filtered_df[filtered_df[disp_col].isin(dispositions)]
    
    if radius_range and rad_col:
        filtered_df[rad_col] = pd.to_numeric(filtered_df[rad_col], errors='coerce')
        filtered_df = filtered_df.dropna(subset=[rad_col])
        filtered_df = filtered_df[(filtered_df[rad_col] >= radius_range[0]) & (filtered_df[rad_col] <= radius_range[1])]
    
    if period_range and per_col:
        filtered_df[per_col] = pd.to_numeric(filtered_df[per_col], errors='coerce')
        filtered_df = filtered_df.dropna(subset=[per_col])
        filtered_df = filtered_df[(filtered_df[per_col] >= period_range[0]) & (filtered_df[per_col] <= period_range[1])]
    
    if tic_id:
        filtered_df = filtered_df[filtered_df['TIC ID'].astype(str) == str(tic_id)]
    
    return filtered_df.reset_index(drop=True)

# --- FUNCȚII DATE STELARE ---

@st.cache_data(ttl="7d")
def fetch_star_data(star_name):
    """Preia date fizice despre stea și imagini SkyView."""
    facts = {}
    try:
        cleaned_input = star_name.upper().replace("TIC", "").strip()
        if cleaned_input.isdigit():
            tic_id = cleaned_input
        else:
            simbad = Simbad()
            simbad.add_votable_fields('ids')
            result = simbad.query_object(star_name)
            ids = result['IDS'][0].decode('utf-8')
            tic_id = [i.replace("TIC", "").strip() for i in ids.split('|') if 'TIC' in i][0]

        star_data = Catalogs.query_criteria(catalog="TIC", ID=int(tic_id))
        facts = {
            'tic_id': tic_id, 'name': star_name, 'ra': star_data['ra'][0], 'dec': star_data['dec'][0],
            'Tmag': star_data['Tmag'][0] if 'Tmag' in star_data.columns else "N/A",
            'Teff': star_data['Teff'][0] if 'Teff' in star_data.columns else "N/A",
            'radius': star_data['rad'][0] if 'rad' in star_data.columns else None,
            'mass': star_data['mass'][0] if 'mass' in star_data.columns else "N/A"
        }
        
        planet_data = NasaExoplanetArchive.query_criteria(table="pscomppars", where=f"hostname = '{star_name}'")
        facts['confirmed_planet_count'] = len(planet_data)
        
        image_urls = SkyView.get_image_links(position=f"{facts['ra']} {facts['dec']}", survey=['DSS2 Red'])
        facts['image_url'] = image_urls[0] if image_urls else None
    except:
        pass
    return facts

@st.cache_data(ttl="7d")
def get_star_parameters(target_id):
    """Returnează doar raza stelei din TIC."""
    try:
        tid = target_id.upper().replace("TIC", "").strip()
        data = Catalogs.query_criteria(catalog="TIC", ID=int(tid))
        return data['rad'][0] if len(data) > 0 else None
    except:
        return None

# --- FUNCȚII EXPLORARE ȘI VÂNĂTOARE DE PLANETE ---

def fetch_catalog_targets(mission_name, disposition_type, num_targets=25):
    """Funcție pentru paginile Explore (TESS/Kepler)."""
    try:
        if mission_name == "TESS":
            url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
            id_col, prefix, disp_col = 'TIC ID', 'TIC ', 'TFOPWG Disposition'
            targets = ["CP", "PC"] if disposition_type == "PLANETS" else ["FP", "EB"]
        else:
            url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&select=kepid,koi_disposition,koi_period,koi_prad&format=csv"
            id_col, prefix, disp_col = 'kepid', 'KIC ', 'koi_disposition'
            targets = ["CONFIRMED", "CANDIDATE"] if disposition_type == "PLANETS" else ["FALSE POSITIVE"]
        
        df = pd.read_csv(url, comment='#')
        filtered = df[df[disp_col].isin(targets)].sample(n=min(num_targets, len(df)))
        filtered["Searchable ID"] = prefix + filtered[id_col].astype(str)
        return filtered
    except:
        return pd.DataFrame()

@st.cache_data(ttl="1h")
def fetch_untested_targets(num_to_sample=20):
    """Găsește stele luminoase care nu sunt în catalogul TOI. Optimizat pentru viteză."""
    try:
        toi_df = get_toi_catalog()
        known_toi_tics = set(toi_df['TIC ID'].astype(str)) if not toi_df.empty else set()

        # Căutare rapidă pe o zonă restrânsă pentru a evita latența serverului MAST
        tic_sample = Catalogs.query_criteria(
            catalog="TIC", Vmag=(10.0, 11.5), 
            dec=(0, 25), ra=(0, 25), pagesize=100
        ).to_pandas()

        untested_df = tic_sample[~tic_sample['ID'].astype(str).isin(known_toi_tics)]
        result = untested_df.head(num_to_sample).copy()
        result['Searchable ID'] = "TIC " + result['ID'].astype(str)
        
        result = result.rename(columns={'Tmag': 'Tmag', 'rad': 'rad', 'mass': 'mass'})
        return result[['Searchable ID', 'Tmag', 'ra', 'dec', 'rad', 'mass']]
    except:
        return pd.DataFrame()
