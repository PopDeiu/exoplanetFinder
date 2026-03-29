# pages/1_Search_for_a_Star.py

import streamlit as st
import lightkurve as lk
from utils import process_selected_data, set_galaxy_background, set_sidebar_style, init_session_state, generate_pdf_report
from utils.database import save_star_observation

st.set_page_config(
    page_title="Caută o stea",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Aplicare styling IMEDIAT ---
set_sidebar_style()
set_galaxy_background("nebula")

# --- Inițializare session state din setări persistente ---
init_session_state()

st.header("Caută o stea după nume sau ID")

st.caption("Pasul 1 din 4 · Alege ținta pe care vrei să o studiezi.")

with st.expander("Cum funcționează această pagină", expanded=False):
    st.markdown(
        """
        1. Introdu numele sau ID-ul unei stele (de exemplu `Kepler-10`, `TIC 261136679`).
        2. Apasă **„Caută date”** pentru a vedea ce curbe de lumină sunt disponibile.
        3. Bifează produsele de date care te interesează (sectoare, misiuni, pipeline-uri).
        4. Apasă **„Procesează fișierele selectate”** pentru a rula analiza de tranzite.

        Poți ajusta sensibilitatea și intervalul de perioade din pagina **Setări**.
        """
    )
st.markdown("Introdu numele unei stele sau ID‑ul ei pentru a căuta tranzite de exoplanete. Poți ajusta toți parametrii de analiză din pagina **Setări**.")
star_id_input = st.text_input(label="Introdu numele sau ID‑ul unei stele", value="TIC 261136679", help="Încearcă să copiezi un „ID căutabil” din paginile de explorare!")
st.info(f"Setări active: Binning: {st.session_state.bin_size} min | "
        f"Sigma: {st.session_state.sigma_val} | "
        f"Periodă: {st.session_state.period_range[0]}-{st.session_state.period_range[1]} zile")
if st.button("Caută date", type="primary"):
    st.session_state.search_result = None
    if not star_id_input:
        st.warning("Te rog introdu un nume de stea.")
    else:
        with st.spinner(f"Se interoghează pentru „{star_id_input}”..."):
            search_term = star_id_input.upper().replace("TIC", "").replace("KIC", "").replace("EPIC", "").strip()
            is_id_search = search_term.isdigit()
            if is_id_search:
                st.info("ID numeric detectat. Se caută în toate misiunile și pentru toți autorii disponibili...")
                result = lk.search_lightcurve(star_id_input)
            else:
                if not st.session_state.selected_missions or not st.session_state.selected_authors:
                    st.warning("Te rog selectează cel puțin o misiune și un autor pe pagina Setări.")
                    result = None
                else:
                    result = lk.search_lightcurve(star_id_input, mission=st.session_state.selected_missions, author=st.session_state.selected_authors)
            
            if result is not None and len(result) > 0:
                st.session_state.search_result = result
            else:
                st.warning("Nu s‑au găsit date pentru criteriile specificate.")


    st.divider()
    st.subheader("Pasul 2: Selectează datele pentru procesare")
    results_df = st.session_state.search_result.table.to_pandas()
    link_col_name = "Archive Link"
    if 'TIC ID' in results_df.columns:
        results_df[link_col_name] = "https://exofop.ipac.caltech.edu/tess/target.php?id=" + results_df['TIC ID'].astype(str)
    elif 'KIC ID' in results_df.columns:
        results_df[link_col_name] = "https://exoplanetarchive.ipac.caltech.edu/overview/" + results_df['KIC ID'].astype(str)
    st.dataframe(results_df, column_config={link_col_name: st.column_config.LinkColumn("Details", display_text="View on Archive ↗️")})
    
    options = [f"Data Product #{i}" for i in range(len(st.session_state.search_result))]
    selected_options = st.multiselect("Alege ce produse de date vrei să descarci:", options=options, default=options)
    
    if st.button("Procesează fișierele selectate", type="primary"):
        if not selected_options:
            st.warning("Te rog selectează cel puțin un produs de date.")
        else:
            selected_indices = [int(opt.split('#')[-1]) for opt in selected_options]
            selected_data_products = st.session_state.search_result[selected_indices]
            min_p, max_p = st.session_state.period_range
            process_selected_data(
                selected_data_products,
                bin_minutes=st.session_state.bin_size,
                outlier_sigma=st.session_state.sigma_val,
                period_min=min_p,
                period_max=max_p
            )

# Afișează butonul de descărcare PDF dacă analiza a fost completată
if hasattr(st.session_state, 'pdf_export_data') and st.session_state.pdf_export_data:
    st.divider()
    st.subheader("Descarcă Raportul")
    
    pdf_data = st.session_state.pdf_export_data
    pdf_bytes = generate_pdf_report(
        star_name=pdf_data['star_name'],
        period=pdf_data['period'],
        depth=pdf_data['depth'],
        radius=pdf_data['radius'],
        figure=pdf_data['figure']
    )

    if hasattr(st.session_state, 'pdf_export_data') and st.session_state.pdf_export_data:
        st.divider()
        st.subheader("Finalizare Analiză")
    
        col_pdf, col_save = st.columns(2)
        
        with col_pdf:
            st.write("📄 Raport PDF")
            pdf_data = st.session_state.pdf_export_data
            pdf_bytes = generate_pdf_report(
                star_name=pdf_data['star_name'],
                period=pdf_data['period'],
                depth=pdf_data['depth'],
                radius=pdf_data['radius'],
                figure=pdf_data['figure']
            )
        
            st.download_button(
                label="Descarcă raportul PDF",
                data=pdf_bytes,
                file_name=f"Exoplanet_Report_{pdf_data['star_name']}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )

        st.markdown("---")
        st.write("📢 **Înregistrează-ti observatiile in bazele de date NASA**")
        st.link_button("Accesează NASA ExoFOP (TESS)", "https://exofop.ipac.caltech.edu/tess/", use_container_width=True)

        # --- SECȚIUNE SALVARE ÎN CONT ---
        with col_save:
            st.write("💾 Salvează în profil")
            if st.session_state.get('logged_in', False):
                with st.container(border=True):
                    user_notes = st.text_area("Notele tale / Observații", placeholder="Ex: Tranzit foarte clar, merită re-analizat.")
                    
                    if st.button("Salvează în baza de date", use_container_width=True):
                        # Preluăm datele din session_state-ul creat de analiza anterioară
                        success = save_star_observation(
                            user_id=st.session_state.user_info['ID'], # Presupunem că ID-ul e în user_info
                            star_id=pdf_data['star_name'],
                            period=pdf_data['period'],
                            depth=pdf_data['depth'],
                            radius=pdf_data['radius'],
                            obs_text=user_notes
                        )
                        
                        if success:
                            st.success("Analiza a fost salvată în contul tău!")
                        else:
                            st.error("Eroare la salvarea datelor.")
            else:
                st.warning("Autentifică-te din sidebar pentru a salva rezultatele în contul tău.")