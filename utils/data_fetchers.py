import pandas as pd
import streamlit as st
import numpy as np
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
from astroquery.skyview import SkyView
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
import astropy.units as u
from astropy.coordinates import SkyCoord

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
    facts = {}
    try:
        # 1. Configurare SIMBAD
        custom_simbad = Simbad()
        custom_simbad.add_votable_fields('ids', 'ra', 'dec', 'sp', 'dist')
        
        # Curățare input
        search_query = star_name.strip()
        result_table = custom_simbad.query_object(search_query)
        
        search_name_for_nasa = search_query
        tic_id = None
        spectral_type = "N/A"

        # Verificăm dacă SIMBAD a găsit ceva
        if result_table is not None and len(result_table) > 0:
            # Extragere sigură IDS
            if 'IDS' in result_table.colnames:
                raw_ids = result_table['IDS'][0]
                # Decodare din bytes în string dacă e necesar
                ids_str = raw_ids.decode('utf-8') if isinstance(raw_ids, bytes) else str(raw_ids)
                ids_list = ids_str.split('|')
                
                for identifier in ids_list:
                    if 'TIC' in identifier:
                        tic_id = identifier.replace("TIC", "").strip()
                    # Căutăm nume recunoscute de NASA
                    if any(x in identifier for x in ["WASP-", "Kepler-", "K2-", "HD ", "HIP ", "GJ ", "TOI"]):
                        search_name_for_nasa = identifier.strip()

            if 'SP_TYPE' in result_table.colnames:
                sp = result_table['SP_TYPE'][0]
                spectral_type = sp.decode('utf-8') if isinstance(sp, bytes) else str(sp)

        # 2. Date din Catalogul TIC (MAST)
        if not tic_id and "TIC" in search_query.upper():
            tic_id = "".join(filter(str.isdigit, search_query))

        if tic_id:
            star_data = Catalogs.query_criteria(catalog="TIC", ID=int(tic_id))
            if len(star_data) > 0:
                row = star_data[0]
                teff = row.get('Teff')
                rad = row.get('rad')
                
                # Calcule Astrofizice
                lum_val = None
                hz_i, hz_o = None, None
                if rad and teff:
                    # L/Lsun = (R/Rsun)^2 * (T/Tsun)^4
                    lum_val = (float(rad)**2) * ((float(teff) / 5778)**4)
                    hz_i = round(np.sqrt(lum_val / 1.1), 3)
                    hz_o = round(np.sqrt(lum_val / 0.53), 3)

                facts.update({
                    'tic_id': tic_id,
                    'ra': round(float(row['ra']), 5),
                    'dec': round(float(row['dec']), 5),
                    'Tmag': row.get('Tmag', "N/A"),
                    'Teff': int(teff) if teff else "N/A",
                    'radius': round(float(rad), 3) if rad else "N/A",
                    'mass': round(float(row['mass']), 3) if row.get('mass') else "N/A",
                    'luminosity': f"{lum_val:.2f}" if lum_val else "N/A",
                    'hz_inner_au': hz_i,
                    'hz_outer_au': hz_o,
                    'spectral_type': spectral_type,
                    'distance_ly': round(float(row['dist']) * 3.26, 2) if row.get('dist') else "N/A"
                })

        # 3. Interogare NASA (PSCompPars)
        try:
            planets = NasaExoplanetArchive.query_object(search_name_for_nasa, table="pscomppars")
            # Convertim tabelul Astropy în Pandas DataFrame
            df = planets.to_pandas()
            facts['confirmed_planet_count'] = len(df)
            facts['planet_df'] = df
        except:
            facts['confirmed_planet_count'] = 0
            facts['planet_df'] = None

        facts['name'] = search_name_for_nasa
        
    except Exception as e:
        # Returnăm eroarea în dicționar pentru a o vedea în UI
        return {"error": f"Eroare tehnică: {str(e)}"}
        
    return facts

@st.cache_data(ttl="7d")
def fetch_star_data(star_name):
    facts = {}
    try:
        # 1. Configurare SIMBAD (Am eliminat 'dist' care cauza eroarea)
        custom_simbad = Simbad()
        custom_simbad.add_votable_fields('ids', 'ra', 'dec', 'sp')
        
        search_query = star_name.strip()
        result_table = custom_simbad.query_object(search_query)
        
        search_name_for_nasa = search_query
        tic_id = None
        spectral_type = "N/A"

        if result_table is not None and len(result_table) > 0:
            if 'IDS' in result_table.colnames:
                raw_ids = result_table['IDS'][0]
                ids_str = raw_ids.decode('utf-8') if isinstance(raw_ids, bytes) else str(raw_ids)
                ids_list = ids_str.split('|')
                
                for identifier in ids_list:
                    if 'TIC' in identifier:
                        # Extragem doar cifrele pentru TIC ID
                        tic_id = "".join(filter(str.isdigit, identifier))
                    if any(x in identifier for x in ["WASP-", "Kepler-", "K2-", "HD ", "HIP ", "GJ ", "TOI"]):
                        search_name_for_nasa = identifier.strip()

            if 'SP_TYPE' in result_table.colnames:
                sp = result_table['SP_TYPE'][0]
                spectral_type = sp.decode('utf-8') if isinstance(sp, bytes) else str(sp)

        # 2. Date din Catalogul TIC (MAST)
        # Dacă nu am găsit TIC-ul în SIMBAD, îl extragem din input dacă e de forma "TIC 123"
        if not tic_id and "TIC" in search_query.upper():
            tic_id = "".join(filter(str.isdigit, search_query))

        if tic_id:
            star_data = Catalogs.query_criteria(catalog="TIC", ID=int(tic_id))
            if len(star_data) > 0:
                row = star_data[0]
                # Extragere valori cu fallback la None pentru calcule
                teff = row.get('Teff')
                rad = row.get('rad')
                mass = row.get('mass')
                dist_pc = row.get('dist') # Distanța în parseci din TIC
                
                # --- CALCULE ASTROFIZICE ---
                lum_val = None
                hz_i, hz_o = None, None
                
                if rad and teff:
                    # L/Lsun = (R/Rsun)^2 * (T/Tsun)^4
                    lum_val = (float(rad)**2) * ((float(teff) / 5778)**4)
                    # Estimare Zona Locuibilă (HZ)
                    hz_i = round(np.sqrt(lum_val / 1.1), 3)
                    hz_o = round(np.sqrt(lum_val / 0.53), 3)

                facts.update({
                    'tic_id': tic_id,
                    'ra': round(float(row['ra']), 5) if row.get('ra') else "N/A",
                    'dec': round(float(row['dec']), 5) if row.get('dec') else "N/A",
                    'Tmag': round(float(row['Tmag']), 3) if row.get('Tmag') else "N/A",
                    'Teff': int(teff) if teff else "N/A",
                    'radius': round(float(rad), 3) if rad else "N/A",
                    'mass': round(float(mass), 3) if mass else "N/A",
                    'luminosity': f"{lum_val:.2f}" if lum_val else "N/A",
                    'hz_inner_au': hz_i,
                    'hz_outer_au': hz_o,
                    'spectral_type': spectral_type,
                    'distance_ly': round(float(dist_pc) * 3.26156, 2) if dist_pc else "N/A"
                })

        # 3. Interogare NASA Arhivă (Folosim numele găsit sau cel introdus)
        try:
            planets = NasaExoplanetArchive.query_object(search_name_for_nasa, table="pscomppars")
            df = planets.to_pandas()
            facts['confirmed_planet_count'] = len(df)
            facts['planet_df'] = df
        except:
            facts['confirmed_planet_count'] = 0
            facts['planet_df'] = None

        facts['name'] = search_name_for_nasa
        
    except Exception as e:
        return {"error": f"Eroare tehnică: {str(e)}"}
        
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
def fetch_untested_targets(num_to_sample=100):
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

def get_common_name(ra_deg, dec_deg):
    """
    Interoghează SIMBAD pentru a găsi un nume comun bazat pe coordonate.
    """
    try:
        # Creăm un obiect de coordonate
        coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.degree, u.degree), frame='icrs')
        
        # Căutăm obiecte pe o rază foarte mică (2 secunde de arc)
        result_table = Simbad.query_region(coord, radius=2 * u.arcsec)
        
        if result_table is not None and len(result_table) > 0:
            # Luăm identificatorul principal (coloana MAIN_ID)
            main_id = result_table[0]['MAIN_ID']
            # Curățăm puțin string-ul (uneori vin cu spații extra)
            return main_id.decode('utf-8') if isinstance(main_id, bytes) else main_id
            
    except Exception:
        pass
    
    return None

from astroquery.simbad import Simbad
import numpy as np

def get_stars_from_simbad(bortle_level, limit=100, lat=None, lon=None):
    from astroquery.simbad import Simbad
    import numpy as np
    
    # Mapăm nivelul Bortle la magnitudinea vizuală (Vmag) aproximativă
    bortle_vmag_limits = {
        1: 7.5, 2: 7.0, 3: 6.5, 4: 6.0, 
        5: 5.5, 6: 5.0, 7: 4.5, 8: 4.0, 9: 3.0
    }
    
    max_vmag = bortle_vmag_limits.get(bortle_level, 4.0)
    
    # --- LOGICA DE VIZIBILITATE PE BAZA LOCAȚIEI ---
    min_dec = -90.0 # Default: vedem tot cerul sudic
    max_dec = 90.0  # Default: vedem tot cerul nordic
    
    if lat is not None:
        # O stea e vizibilă deasupra orizontului dacă DEC > (Latitudine - 90)
        min_dec = float(lat) - 90.0
        
    try:
        # Configurăm SIMBAD să aducă exact ce avem nevoie
        custom_simbad = Simbad()
        custom_simbad.ROW_LIMIT = limit
        
        # Adăugăm filtre suplimentare: Vmag, Clasa Spectrală, Paralaxa
        custom_simbad.add_votable_fields('flux(V)', 'sp', 'plx')
        
        # Interogăm SIMBAD, limitând după Vmag și declinație
        query_string = f"flux(V) <= {max_vmag} & dec >= {min_dec}"
        query_data = custom_simbad.query_criteria(query_string)
        
        if query_data is None or len(query_data) == 0:
            return []

        stars_list = []
        
        for row in query_data:
            # 1. Numele stelei (SIMBAD returnează bytes uneori)
            raw_name = row['MAIN_ID']
            base_name = raw_name.decode('utf-8').strip() if isinstance(raw_name, bytes) else str(raw_name).strip()
            # Curățăm numele (ex: "NAME Sirius" -> "Sirius")
            clean_name = base_name.replace("NAME ", "").replace("* ", "")
            
            # 2. Magnitudinea V
            vmag = float(row['FLUX_V']) if not np.ma.is_masked(row['FLUX_V']) else max_vmag
            
            # 3. Clasa spectrală (temperatura/culoarea)
            raw_sp = row['SP_TYPE']
            sp_type = raw_sp.decode('utf-8').strip() if isinstance(raw_sp, bytes) else str(raw_sp).strip()
            sp_text = sp_type if sp_type and not np.ma.is_masked(row['SP_TYPE']) else "Necunoscută"

            # 4. Calculul distanței (dacă avem paralaxă)
            distanta_al = "Necunoscută"
            if not np.ma.is_masked(row['PLX_VALUE']) and float(row['PLX_VALUE']) > 0:
                plx_mas = float(row['PLX_VALUE'])
                # Formula distanței: D (parseci) = 1000 / paralaxă(mas)
                # 1 parsec = 3.26156 ani-lumină
                distanta_pc = 1000.0 / plx_mas
                distanta_al = f"{round(distanta_pc * 3.26156, 1)} ani-lumină"

            # 5. Coordonate
            # SIMBAD le returnează în format "HH MM SS" (RA) și "+DD MM SS" (DEC).
            # Le trimitem ca string-uri, așa cum făcea și funcția ta originală.
            ra_str = str(row['RA'])
            dec_str = str(row['DEC'])
            
            # Formăm variabilele pentru a respecta tuplul tău original
            star_id = f"SIMBAD_{clean_name.replace(' ', '_')}"
            display_name = f"{clean_name} (Vmag: {round(vmag, 2)})"
            
            loc_text = f"Lat: {lat}°" if lat else "Global"
            desc = (f"Bortle: {bortle_level}. Locație: {loc_text}. "
                    f"Clasă spectrală: {sp_text}. Distanță estimată: {distanta_al}.")

            # Păstrăm exact formatul tuplului așteptat de restul aplicației tale
            stars_list.append((
                star_id,       # Înlocuiește tic_id
                display_name,  # Înlocuiește name
                ra_str,        # Coord RA
                dec_str,       # Coord DEC
                desc           # Descriere extinsă
            ))
            
        return stars_list

    except Exception as e:
        # Folosim importul local doar pentru logging ca să nu pice în Docker
        import logging
        logging.error(f"Eroare la descărcarea stelelor din SIMBAD: {e}")
        return []