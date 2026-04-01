import streamlit as st
# Asigură-te că imporți funcțiile din fișierul tău utils.py

from utils.database import save_naked_eye_star, get_all_naked_eye_stars

st.set_page_config(page_title="Observații Ochiul Liber", page_icon="✨")

st.title("Planetariu VR")
st.markdown("Documentează stelele pe care le poți observa direct, în funcție de nivelul de poluare luminoasă.")

# --- SETĂRI CONTEXTUALE BORTLE ---
with st.sidebar:
    st.header("Condiții de Observare")
    bortle_scale = st.slider("Scara Bortle (1 = Cer perfect, 9 = Centru oraș)", 1, 9, 4)
    st.info(f"Setat la: **Bortle {bortle_scale}**")
    

# --- FORMULAR ADĂUGARE ---
with st.form("star_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    
    with col1:
        tic_id = st.text_input("TIC_ID", help="Ex: TIC 123456 (dacă nu-l știi, pune N/A)")
        name = st.text_input("Nume Stea *", help="Ex: Vega, Arcturus")
    
    with col2:
        ra = st.text_input("Ascensie Dreaptă (RA) *", placeholder="ex: 18h 36m")
        dec = st.text_input("Declinație (DEC) *", placeholder="ex: +38° 47'")
        
    # Pre-completăm text_area cu nivelul Bortle ca să fie salvat în DB în coloana description

    description = st.text_area("Descriere / Note observație", height=100, placeholder=f"Observat la Bortle {bortle_scale}")
    
    submit_button = st.form_submit_button("Salvează Observația")

# --- LOGICA DE SALVARE ---
if submit_button:
    # Validare minimă a datelor
    if not name or not ra or not dec:
        st.warning("Te rog să completezi Numele, RA și Declinația!")
    else:
        # Dacă userul lasă TIC_ID gol, punem un default ca să nu pice constrângerea NOT NULL din DB
        final_tic = tic_id if tic_id.strip() != "" else "N/A"
        
        success = save_naked_eye_star(final_tic, name, ra, dec, description)
        
        if success:
            st.success(f"Steaua '{name}' a fost salvată cu succes în baza de date!")
            st.balloons()
        else:
            st.error("A apărut o eroare la salvare. Verifică terminalul pentru detalii.")

st.divider()

# --- VIZUALIZARE BAZĂ DE DATE ---
st.subheader("Catalogul tău de observații")
if st.button("Încarcă înregistrările"):
    stars_data = get_all_naked_eye_stars()
    
    if stars_data:
        # Afișăm datele într-un tabel interactiv din Streamlit
        st.dataframe(
            stars_data,
            use_container_width=True,
            column_config={
                "ID": st.column_config.NumberColumn("ID", format="%d"),
                "TIC_ID": "TIC_ID",
                "name": "Nume",
                "ra": "RA",
                "declination": "DEC",
                "description": "Note"
            },
            hide_index=True
        )
    else:
        st.info("Nu există nicio stea înregistrată momentan.")