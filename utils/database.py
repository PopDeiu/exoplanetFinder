import mysql.connector
import os
import logging
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
from .data_fetchers import get_common_name 
from .data_fetchers import get_common_name_from_simbad

# Încărcăm variabilele de mediu
load_dotenv()

def update_app_setting(nume_setare, valoare_noua):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Folosim UPDATE pentru a modifica valoarea unde coloana 'setare' se potrivește
        sql = "UPDATE setariVR SET valoare = %s WHERE setare = %s"
        cursor.execute(sql, (str(valoare_noua), nume_setare))
        conn.commit()
    except Exception as e:
        print(f"Eroare la update DB: {e}")
    finally:
        cursor.close()
        conn.close()

# În utils.py
def get_all_settings():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT setare, valoare FROM setariVR") # Ajustează numele tabelului dacă diferă
        rows = cursor.fetchall()
        # Transformăm lista de rânduri într-un singur dicționar { "latitudine": "46.18", ... }
        settings_dict = {row['setare']: row['valoare'] for row in rows}
        return settings_dict
    except Exception as e:
        print(f"Eroare la citirea setărilor: {e}")
        return {}
    finally:
        cursor.close()
        conn.close()

def get_connection():
    """Creează și returnează o conexiune cache-uită la MySQL."""
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            port=os.getenv("DB_PORT", 3306)
        )
        return conn
    except mysql.connector.Error as err:
        st.error(f"Eroare de conectare la DB: {err}")
        return None
    
def register_user(username, password):
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            # Verificăm dacă user-ul există deja
            cursor.execute("SELECT * FROM users WHERE user = %s", (username,))
            if cursor.fetchone():
                return False, "Utilizatorul există deja!"
            
            # Insert noul user
            sql = "INSERT INTO users (user, password) VALUES (%s, %s)"
            cursor.execute(sql, (username, password))
            conn.commit()
            cursor.close()
            return True, "Cont creat cu succes!"
        except Exception as e:
            return False, f"Eroare: {e}"
        finally:
            conn.close()
    return False, "Nu s-a putut conecta la DB"

def save_star_observation(user_id, star_id, period, depth, radius, obs_text):
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            # Avem 6 coloane listate aici (fără ID și created_at care sunt automate)
            sql = """INSERT INTO saved_observations 
                     (user_id, star_id, period, depth, radius, observations) 
                     VALUES (%s, %s, %s, %s, %s, %s)"""
            
            # Trebuie să avem exact 6 elemente în acest tuplu:
            values = (user_id, star_id, period, depth, radius, obs_text)
            
            cursor.execute(sql, values)
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            st.error(f"Eroare SQL: {e}") # Debugging pentru a vedea eroarea exactă
            return False
        finally:
            conn.close()
    return False

def get_user_observations(user_id):
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            # Ordonăm după data creării (cele mai noi primele)
            query = "SELECT ID, star_id, period, depth, radius, observations, created_at FROM saved_observations WHERE user_id = %s ORDER BY created_at DESC"
            cursor.execute(query, (user_id,))
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            print(f"Eroare la citire: {e}")
            return []
        finally:
            conn.close()
    return []

def update_observation_notes(obs_id, user_id, new_notes):
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            query = "UPDATE saved_observations SET observations = %s WHERE ID = %s AND user_id = %s"
            cursor.execute(query, (new_notes, obs_id, user_id))
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Eroare la update: {e}")
            return False
        finally:
            conn.close()
    return False

def delete_observation(obs_id, user_id):
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            # Verificăm user_id pentru siguranță (să nu poată șterge cineva observația altcuiva prin ID)
            query = "DELETE FROM saved_observations WHERE ID = %s AND user_id = %s"
            cursor.execute(query, (obs_id, user_id))
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Eroare la ștergere: {e}")
            return False
        finally:
            conn.close()
    return False


def get_user_by_id(user_id):
    # 1. Dacă nu avem un user_id (ex: la deschiderea paginii înainte de login), ne oprim din start.
    if not user_id:
        return None
        
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT ID, user FROM users WHERE ID = %s", (user_id,))
            user = cursor.fetchone()
            cursor.close()
            return user
        except Exception as e:
            # 2. Folosim logging în loc de print pentru a evita eroarea "closed file" în Docker
            logging.error(f"Eroare la preluarea userului din DB: {e}")
            return None
        finally:
            conn.close()
    return None

def verify_credentials(username, password):
    """Verifică user-ul și parola în baza de date."""
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            # Folosim interogare parametrizată pentru a preveni SQL Injection
            sql = "SELECT * FROM users WHERE user = %s AND password = %s"
            cursor.execute(sql, (username, password))
            result = cursor.fetchone()
            cursor.close()
            return result
        except mysql.connector.Error as err:
            st.error(f"Eroare la query: {err}")
        finally:
            conn.close()
    return None



def save_naked_eye_star(tic_id, name, ra, dec, description):
    """Salvează o stea vizibilă cu ochiul liber în DB."""
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            sql = """INSERT INTO stars (TIC_ID, name, ra, declination, description) 
                     VALUES (%s, %s, %s, %s, %s)"""
            values = (tic_id, name, ra, dec, description)
            cursor.execute(sql, values)
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Eroare SQL la salvarea stelei: {e}")
            return False
        finally:
            conn.close()
    return False


def get_all_naked_eye_stars():
    """Returnează toate stelele salvate din DB."""
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT ID, TIC_ID, name, ra, declination, description FROM stars ORDER BY ID DESC")
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            print(f"Eroare la citirea stelelor: {e}")
            return []
        finally:
            conn.close()
    return []



def clear_all_naked_eye_stars():
    """Șterge toate înregistrările vechi din tabelul stele."""
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
           
            cursor.execute("DELETE FROM stars") 
            
           
            cursor.execute("ALTER TABLE stars AUTO_INCREMENT = 1")
            
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            # Afișăm eroarea exactă direct în interfața Streamlit ca să știm ce se întâmplă
            import streamlit as st
            st.error(f"Eroare SQL detaliată la ștergere: {e}")
            return False
        finally:
            conn.close()
    return False

def bulk_save_stars(stars_list):
    """Salvează o listă de stele rapid în baza de date."""
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            sql = """INSERT INTO stars (TIC_ID, name, ra, declination, description) 
                     VALUES (%s, %s, %s, %s, %s)"""
            # Folosim executemany pentru a insera mii de rânduri instantaneu
            cursor.executemany(sql, stars_list)
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Eroare la bulk insert: {e}")
            return False
        finally:
            conn.close()
    return False




def get_real_stars_by_bortle(bortle_level, limit=100, lat=None, lon=None):
    import pandas as pd
    from astroquery.mast import Catalogs
    
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
        # Interogăm MAST, adăugând limitarea pe axa Declinației (dec)
        logging.info("Încep interogarea MAST...")
        query_data = Catalogs.query_criteria(
            catalog="TIC", 
            Vmag=(-2.0, max_vmag),
            #dec=(min_dec, max_dec), # Filtrul nostru magic de locație!
            pagesize=limit
        ).to_pandas()
        
        if query_data.empty:
            return []

        hip_ids = []
        for val in query_data['HIP']:
            if pd.notna(val):
                hip_ids.append(f"HIP {int(val)}")

        # 3. INTEROGARE UNICĂ SIMBAD (Bulk Name Lookup)
        names_map = {}
        if hip_ids:
            logging.info(f"Interogare SIMBAD Bulk pentru {len(hip_ids)} obiecte...")
            
            # Configurăm SIMBAD să aducă și câmpul 'ids'
            Simbad.reset_votable_fields() # Curățăm câmpurile anterioare
            Simbad.add_votable_fields('ids', 'main_id')
            
            # Interogarea Bulk (o singură cerere pentru toate ID-urile)
            result_table = Simbad.query_objects(hip_ids)

            if result_table is not None:
                # Iterăm prin tabelul de rezultate SIMBAD
                # Notă: result_table păstrează ordinea listei hip_ids
                for i in range(len(result_table)):
                    current_hip = hip_ids[i]
                    
                    # Extragem câmpul 'ids' (care conține toate denumirile separate prin |)
                    all_ids_raw = result_table['ids'][i]
                    all_ids = all_ids_raw.decode('utf-8') if isinstance(all_ids_raw, bytes) else str(all_ids_raw)
                    
                    found_name = None
                    
                    # LOGICA TA: Căutăm denumirea care începe cu "NAME"
                    for identifier in all_ids.split('|'):
                        identifier = identifier.strip()
                        if identifier.startswith('NAME'):
                            found_name = identifier.replace('NAME', '').strip()
                            break
                    
                    # Dacă nu găsim "NAME", folosim main_id ca fallback
                    if not found_name:
                        m_id = result_table['main_id'][i]
                        found_name = m_id.decode('utf-8') if isinstance(m_id, bytes) else str(m_id)
                        found_name = found_name.strip()

                    # Salvăm în dicționar: { "HIP 123": "Sirius" }
                    names_map[current_hip] = found_name
        stars_list = []

        for index, row in query_data.iterrows():
            ra_deg = float(row['ra'])
            dec_deg = float(row['dec'])
            hip_name = f"TIC {row['ID']}"
        
        # --- LOGICA NOUĂ PENTRU NUME ---
            hip_key = f"HIP {int(row['HIP'])}" if pd.notna(row['HIP']) else None
            
            # Căutăm numele în dicționarul nostru (fără alt API call)
            common_name = names_map.get(hip_key) if hip_key else None
            
            if common_name:
                name = f"{common_name}"
            else:
                name = f"Stea TIC: {row['ID']}" # Fallback dacă nu are nume sau HIP
        
        # -------------------------------

            ra_str = f"{ra_deg:.2f}°"
            tic_id = f"TIC {row['ID']}"
            #name = f"Stea Vmag: {round(row['Vmag'], 2)}" 
            
            ra_deg = float(row['ra'])
            dec_deg = float(row['dec'])
            
            ra_str = f"{ra_deg:.2f}°"
            dec_str = f"{dec_deg:.2f}°"
            
            
            # --- Extragere Date Fizice ---
            vmag = row['Vmag']
            teff = row['Teff'] if pd.notna(row['Teff']) else "Necunoscută"


            masa = f"{row['mass']:.2f} Mase Solare" if pd.notna(row['mass']) else "Necunoscută"
            raza = f"{row['rad']:.2f} Raze Solare" if pd.notna(row['rad']) else "Necunoscută"

            # --- Calcul Distanță (din paralaxă plx în mas) ---
            dist_ly = "Necunoscută"
            if pd.notna(row['plx']) and row['plx'] > 0:
                dist_pc = 1000.0 / row['plx']
                dist_ly = f"{round(dist_pc * 3.26156, 1)} ani-lumină"

            hip_key = f"HIP {int(row['HIP'])}" if pd.notna(row['HIP']) else None
            name_display = names_map.get(hip_key) if hip_key else f"TIC {row['ID']}"

            # --- Construire Descriere (formatată pentru UI) ---
            desc = (
                f"Sursă: {name_display}. \n"
                f"• Magnitudine: {vmag:.2f} \n"
                f"• Distanță: {dist_ly} \n"
                f"• Masă: {masa} \n"
                f"• Temperatură: {teff} K \n"
                f"• Rază: {raza}"
            )

            stars_list.append((
                tic_id,
                name,
                ra_str,
                dec_str,
                desc
            ))
            
        return stars_list

    except Exception as e:
        # Folosim importul local doar pentru logging ca să nu pice în Docker
        logging.info(f"Coloane disponibile: {query_data.columns.tolist()}")
        logging.error(f"Eroare la descărcarea stelelor din MAST: {e}")
        return []
    
def get_saved_location():
    """Extrage latitudinea și longitudinea salvate în baza de date."""
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            # ATENȚIE: Schimbă 'setari_observator' cu numele real al tabelului tău
            # și asigură-te că numele coloanelor corespund!
            cursor.execute("SELECT setare, valoare FROM setariVR WHERE setare IN ('latitudine', 'longitudine')")
            locatie = cursor.fetchall()
            cursor.close()
            
            if locatie:
                lat = None
                lon = None
                for row in locatie:
                    if row['setare'] == 'latitudine':
                        lat = float(row['valoare'])
                    elif row['setare'] == 'longitudine':
                        lon = float(row['valoare'])
                return lat, lon
        except Exception as e:
            logging.error(f"Eroare la citirea locației din DB: {e}")
        finally:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if conn:
                conn.close()
            
    # Returnăm None dacă nu găsim locația în DB
    return None, None
