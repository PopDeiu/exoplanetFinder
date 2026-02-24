# pages/1_Search_for_a_Star.py

import streamlit as st
import lightkurve as lk
from utils import process_selected_data, set_galaxy_background, set_sidebar_style

st.set_page_config(
    page_title="Caută o stea",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Aplicare styling IMEDIAT ---
set_sidebar_style()
set_galaxy_background("nebula")

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

if st.session_state.search_result is not None:
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
