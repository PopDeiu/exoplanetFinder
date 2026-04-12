import mysql.connector
import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from astroquery.mast import Catalogs
import logging 


# Încărcăm variabilele de mediu
load_dotenv()

@st.cache_resource
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
        query_data = Catalogs.query_criteria(
            catalog="TIC", 
            Vmag=(-2.0, max_vmag),
            dec=(min_dec, max_dec), # Filtrul nostru magic de locație!
            pagesize=limit
        ).to_pandas()
        
        if query_data.empty:
            return []

        stars_list = []
        
        for index, row in query_data.iterrows():
            tic_id = f"TIC {row['ID']}"
            name = f"Stea Vmag: {round(row['Vmag'], 2)}" 
            
            ra_deg = float(row['ra'])
            dec_deg = float(row['dec'])
            
            ra_str = f"{ra_deg:.2f}°"
            dec_str = f"{dec_deg:.2f}°"
            
            # Formăm o descriere care să includă și locația pentru context
            loc_text = f"Lat: {lat}°" if lat else "Global"
            desc = f"Descărcată automat via MAST. Bortle: {bortle_level}. Vizibilă din locația: {loc_text}."

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
        import logging
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
            cursor.execute("SELECT setare, valoare FROM setariVR LIMIT 1")
            locatie = cursor.fetchone()
            cursor.close()
            
            if locatie:
                return float(locatie['setare']), float(locatie['valoare'])
        except Exception as e:
            import logging
            logging.error(f"Eroare la citirea locației din DB: {e}")
            
    # Returnăm None dacă nu găsim locația în DB
    return None, None