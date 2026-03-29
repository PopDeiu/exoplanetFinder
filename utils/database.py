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