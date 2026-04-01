import mysql.connector
import os
import streamlit as st
from dotenv import load_dotenv

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
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT ID, user FROM users WHERE ID = %s", (user_id,))
            user = cursor.fetchone()
            cursor.close()
            return user
        except Exception as e:
            print(f"Eroare la preluarea userului: {e}")
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
            sql = """INSERT INTO stele (TIC_ID, name, ra, declination, description) 
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
            cursor.execute("SELECT ID, TIC_ID, name, ra, declination, description FROM stele ORDER BY ID DESC")
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
            cursor.execute("TRUNCATE TABLE stele") # Truncate e mult mai rapid decât DELETE pentru tot tabelul
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Eroare la ștergerea stelelor: {e}")
            return False
    return False

def bulk_save_stars(stars_list):
    """Salvează o listă de stele rapid în baza de date."""
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            sql = """INSERT INTO stele (TIC_ID, name, ra, declination, description) 
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

def get_stars_by_bortle_mock(bortle_level):
    """
    SIMULARE: NASA / Cataloage Stelare.
    Aici ar trebui integrat un API real (ex: astroquery cu catalogul Hipparcos).
    Momentan generăm date de test bazate pe magnitudine pentru a nu bloca serverul.
    """
    # Mapăm Bortle la numărul aproximativ de stele vizibile (pentru simulare)
    # În realitate, aici faci un query: "SELECT * FROM catalog WHERE magnitude < limit"
    limits = {1: 3000, 2: 2000, 3: 1500, 4: 800, 5: 400, 6: 200, 7: 100, 8: 50, 9: 20}
    
    num_stars = limits.get(bortle_level, 50)
    stars_data = []
    
    for i in range(1, num_stars + 1):
        stars_data.append((
            f"TIC {100000 + i}", 
            f"Stea Bortle {bortle_level} #{i}", 
            f"{10 + (i%14)}h {i%60}m", 
            f"+{i%90}°", 
            f"Generată automat pentru Bortle {bortle_level}"
        ))
        
    return stars_data