import mysql.connector
import os
import logging
import streamlit as st
import bcrypt
from dotenv import load_dotenv
import pandas as pd
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
from .data_fetchers import get_common_name
from .data_fetchers import get_common_name_from_simbad
from .tts_utils import generate_scenario_wav

# Încărcăm variabilele de mediu
load_dotenv()

def update_app_setting(nume_setare, valoare_noua):
    conn = get_connection()
    if not conn:
        st.error("Nu s-a putut conecta la DB pentru a salva setarea.")
        return
    cursor = conn.cursor()
    valoare_str = str(valoare_noua)
    try:
        cursor.execute(
            "UPDATE setariVR SET valoare = %s WHERE setare = %s",
            (valoare_str, nume_setare)
        )
        conn.commit()
    except Exception as e:
        st.error(f"Eroare la update DB: {e}")
    finally:
        cursor.close()
        conn.close()

# În utils.py
def get_all_settings():
    conn = get_connection()
    if not conn:
        st.error("Nu s-a putut conecta la DB pentru a citi setările.")
        return {}
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT setare, valoare FROM setariVR")
        rows = cursor.fetchall()
        settings_dict = {row['setare']: row['valoare'] for row in rows}
        return settings_dict
    except Exception as e:
        st.error(f"Eroare la citirea setărilor: {e}")
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
            
            # Hash parola cu bcrypt înainte de stocare
            hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            sql = "INSERT INTO users (user, password) VALUES (%s, %s)"
            cursor.execute(sql, (username, hashed.decode('utf-8')))
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
            # Luăm user-ul după username, apoi verificăm parola cu bcrypt
            cursor.execute("SELECT * FROM users WHERE user = %s", (username,))
            result = cursor.fetchone()
            cursor.close()
            if result and bcrypt.checkpw(password.encode('utf-8'), result['password'].encode('utf-8')):
                return result
            return None
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


def init_stars_bortle_table():
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stars_bortle (
                    ID INT AUTO_INCREMENT PRIMARY KEY,
                    TIC_ID VARCHAR(255),
                    name VARCHAR(255),
                    ra VARCHAR(50),
                    declination VARCHAR(50),
                    description TEXT,
                    bortle INT NOT NULL
                )
            """)
            conn.commit()
            cursor.close()
        except Exception as e:
            st.error(f"Eroare la crearea tabelului stars_bortle: {e}")
        finally:
            conn.close()


def clear_stars_by_bortle(bortle_level):
    """Șterge doar stelele dintr-un nivel Bortle specific."""
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM stars_bortle WHERE bortle = %s", (bortle_level,))
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            st.error(f"Eroare la ștergerea stelelor după bortle: {e}")
            return False
        finally:
            conn.close()
    return False


def bulk_save_stars_bortle(stars_list, bortle_level):
    """Salvează stele într-un nivel Bortle specific."""
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            sql = """INSERT INTO stars_bortle (TIC_ID, name, ra, declination, description, bortle) 
                     VALUES (%s, %s, %s, %s, %s, %s)"""
            stars_with_bortle = [(*star, bortle_level) for star in stars_list]
            cursor.executemany(sql, stars_with_bortle)
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Eroare la bulk insert stars_bortle: {e}")
            return False
        finally:
            conn.close()
    return False


def get_stars_bortle_by_level(bortle_level):
    """Returnează stelele dintr-un nivel Bortle specific."""
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT ID, TIC_ID, name, ra, declination, description, bortle FROM stars_bortle WHERE bortle = %s ORDER BY ID DESC", (bortle_level,))
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            print(f"Eroare la citirea stelelor după bortle: {e}")
            return []
        finally:
            conn.close()
    return []



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


def ensure_tables_have_user_id():
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            for table in ['scenarii', 'lectii']:
                cursor.execute(
                    "SELECT COUNT(*) FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = 'user_id'",
                    (table,)
                )
                exists = cursor.fetchone()[0] > 0
                if not exists:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN user_id INT DEFAULT NULL")
            conn.commit()
            cursor.close()
        except Exception as e:
            logging.error(f"Eroare la migrarea tabelelor: {e}")
        finally:
            conn.close()


def save_scenariu(nume, viteza, bortle, longitudine, latitudine, data_si_ora_obs, foloseste_data_curenta, afisare_constelatii, text="", durata=0, user_id=None, scenariu_id=None):
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            existing_id = None
            if scenariu_id is not None:
                existing_id = scenariu_id
                cursor.execute("""
                    UPDATE scenarii SET nume=%s, viteza=%s, bortle=%s, longitudine=%s, latitudine=%s,
                    data_si_ora_obs=%s, foloseste_data_curenta=%s, afisare_constelatii=%s,
                    text=%s, durata=%s
                    WHERE ID=%s AND user_id=%s
                """, (nume, str(viteza), str(bortle), str(longitudine), str(latitudine),
                      str(data_si_ora_obs), str(foloseste_data_curenta), str(afisare_constelatii),
                      str(text), str(durata), scenariu_id, user_id))
            else:
                cursor.execute(
                    "SELECT ID FROM scenarii WHERE nume = %s AND user_id = %s", (nume, user_id)
                )
                row = cursor.fetchone()
                if row:
                    existing_id = row[0]
                    cursor.execute("""
                        UPDATE scenarii SET viteza=%s, bortle=%s, longitudine=%s, latitudine=%s,
                        data_si_ora_obs=%s, foloseste_data_curenta=%s, afisare_constelatii=%s,
                        text=%s, durata=%s
                        WHERE ID=%s AND user_id=%s
                    """, (str(viteza), str(bortle), str(longitudine), str(latitudine),
                          str(data_si_ora_obs), str(foloseste_data_curenta), str(afisare_constelatii),
                          str(text), str(durata), existing_id, user_id))
                else:
                    cursor.execute("""
                        INSERT INTO scenarii (nume, viteza, bortle, longitudine, latitudine, data_si_ora_obs, foloseste_data_curenta, afisare_constelatii, text, durata, user_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (nume, str(viteza), str(bortle), str(longitudine), str(latitudine),
                          str(data_si_ora_obs), str(foloseste_data_curenta), str(afisare_constelatii),
                          str(text), str(durata), user_id))
                    existing_id = cursor.lastrowid
            conn.commit()
            cursor.close()

            if text and text.strip():
                wav_result = generate_scenario_wav(existing_id, text)
                if wav_result:
                    st.session_state['_tts_msg'] = ("success", f"🔊 Audio regenerat cu succes pentru scenariul ID {existing_id}")
                else:
                    st.session_state['_tts_msg'] = ("warning", f"TTS a eșuat pentru scenariul ID {existing_id}. Verifică textul introdus.")
            else:
                wav_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", f"{existing_id}.wav")
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                    st.session_state['_tts_msg'] = ("info", f"Fișierul audio pentru scenariul ID {existing_id} a fost șters (text gol).")

            return existing_id
        except Exception as e:
            st.error(f"Eroare la salvarea scenariului: {e}")
            return False
        finally:
            conn.close()
    return False


def get_all_scenarii(user_id=None):
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            if user_id is not None:
                cursor.execute("SELECT * FROM scenarii WHERE user_id = %s ORDER BY nume ASC", (user_id,))
            else:
                cursor.execute("SELECT * FROM scenarii ORDER BY nume ASC")
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            st.error(f"Eroare la citirea scenariilor: {e}")
            return []
        finally:
            conn.close()
    return []


def get_scenariu_by_id(scenariu_id, user_id=None):
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            if user_id is not None:
                cursor.execute("SELECT * FROM scenarii WHERE ID = %s AND user_id = %s", (scenariu_id, user_id))
            else:
                cursor.execute("SELECT * FROM scenarii WHERE ID = %s", (scenariu_id,))
            result = cursor.fetchone()
            cursor.close()
            return result
        except Exception as e:
            st.error(f"Eroare la citirea scenariului: {e}")
            return None
        finally:
            conn.close()
    return None


def delete_scenariu(scenariu_id, user_id=None):
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM scenarii WHERE ID = %s AND user_id = %s", (scenariu_id, user_id))
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            st.error(f"Eroare la ștergerea scenariului: {e}")
            return False
        finally:
            conn.close()
    return False


def rename_scenariu(scenariu_id, nume_nou, user_id=None):
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("UPDATE scenarii SET nume = %s WHERE ID = %s AND user_id = %s", (nume_nou, scenariu_id, user_id))
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            st.error(f"Eroare la redenumirea scenariului: {e}")
            return False
        finally:
            conn.close()
    return False


# ========== LECȚII (LESSONS) ==========

def init_lectii_table():
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lectii (
                    ID INT AUTO_INCREMENT PRIMARY KEY,
                    nume VARCHAR(255) NOT NULL,
                    descriere TEXT,
                    scenarii_ids TEXT NOT NULL,
                    user_id INT DEFAULT NULL
                )
            """)
            conn.commit()
            cursor.close()
        except Exception as e:
            st.error(f"Eroare la crearea tabelului lectii: {e}")
        finally:
            conn.close()


def save_lectie(nume, descriere, scenarii_ids, user_id=None):
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT ID FROM lectii WHERE nume = %s AND user_id = %s", (nume, user_id))
            existing = cursor.fetchone()
            if existing:
                cursor.execute(
                    "UPDATE lectii SET descriere=%s, scenarii_ids=%s WHERE ID=%s AND user_id=%s",
                    (descriere, scenarii_ids, existing[0], user_id)
                )
            else:
                cursor.execute(
                    "INSERT INTO lectii (nume, descriere, scenarii_ids, user_id) VALUES (%s, %s, %s, %s)",
                    (nume, descriere, scenarii_ids, user_id)
                )
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            st.error(f"Eroare la salvarea lecției: {e}")
            return False
        finally:
            conn.close()
    return False


def get_all_lectii(user_id=None):
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            if user_id is not None:
                cursor.execute("SELECT * FROM lectii WHERE user_id = %s ORDER BY nume ASC", (user_id,))
            else:
                cursor.execute("SELECT * FROM lectii ORDER BY nume ASC")
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            st.error(f"Eroare la citirea lecțiilor: {e}")
            return []
        finally:
            conn.close()
    return []


def get_lectie_by_id(lectie_id, user_id=None):
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            if user_id is not None:
                cursor.execute("SELECT * FROM lectii WHERE ID = %s AND user_id = %s", (lectie_id, user_id))
            else:
                cursor.execute("SELECT * FROM lectii WHERE ID = %s", (lectie_id,))
            result = cursor.fetchone()
            cursor.close()
            return result
        except Exception as e:
            st.error(f"Eroare la citirea lecției: {e}")
            return None
        finally:
            conn.close()
    return None


def delete_lectie(lectie_id, user_id=None):
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM lectii WHERE ID = %s AND user_id = %s", (lectie_id, user_id))
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            st.error(f"Eroare la ștergerea lecției: {e}")
            return False
        finally:
            conn.close()
    return False


def rename_lectie(lectie_id, nume_nou, user_id=None):
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("UPDATE lectii SET nume = %s WHERE ID = %s AND user_id = %s", (nume_nou, lectie_id, user_id))
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            st.error(f"Eroare la redenumirea lecției: {e}")
            return False
        finally:
            conn.close()
    return False
