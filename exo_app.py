# =============================================================================
# EXOPLANET HUNTER - WEB INTERFACE (V36, Corrected Query Filter)
#
# This version fixes an error in the "Find Untested Targets" tab by using
# the correct 'plx' (parallax) filter instead of 'distance'.
#
# To run this app:
# 1. Save the code as 'app.py'
# 2. Run the command in your terminal: streamlit run app.py
# =============================================================================

import streamlit as st
import lightkurve as lk
import matplotlib.pyplot as plt
from astropy import units as u
import time
import pandas as pd
import numpy as np
from astroquery.mast import Catalogs

# --- Helper Functions (process_selected_data and fetch_catalog_targets are unchanged) ---
def process_selected_data(selected_items):
    # This function remains the same as the previous version.
    # (Code is omitted for brevity but should be included in your file)
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    try:
        total_files = len(selected_items)
        status_placeholder.info(f"✅ Starting analysis on {total_files} selected data products...")
        progress_bar = progress_placeholder.progress(0)
        processed_light_curves = []
        for i in range(total_files):
            status_text = f"⬇️ Downloading and preparing file {i + 1} of {total_files}..."
            status_placeholder.info(status_text)
            lc = selected_items[i].download()
            if lc is not None and len(lc.remove_nans().flux) > 0:
                normalized_lc = lc.normalize()
                processed_light_curves.append(normalized_lc)
            else:
                st.write(f"Skipping empty or invalid data for file {i + 1}.")
            progress_bar.progress((i + 1) / total_files)
        if not processed_light_curves:
            st.error("Could not process any of the selected light curve data.")
            status_placeholder.empty(); progress_placeholder.empty()
            return
        status_placeholder.info("⚙️ All files processed. Stitching data segments together...")
        final_collection = lk.LightCurveCollection(processed_light_curves)
        lc = final_collection.stitch().remove_nans()
        time.sleep(1)
        status_placeholder.info("🧹 Binning and flattening the light curve to remove noise...")
        progress_placeholder.empty()
        binned_lc = lc.bin(time_bin_size=10 * u.minute)
        clean_lc = binned_lc.flatten().remove_outliers()
        st.subheader("Cleaned & Flattened Light Curve")
        fig1, ax1 = plt.subplots(); clean_lc.plot(ax=ax1, ylabel="Normalized Flux"); st.pyplot(fig1)
        status_placeholder.info("🔍 Searching for periodic transit signals...")
        periodogram = clean_lc.to_periodogram(method='bls')
        planet_period = periodogram.period_at_max_power
        planet_transit_time = periodogram.transit_time_at_max_power
        st.success(f"Strongest signal found at a period of: **{planet_period.value:.4f} days**")
        st.subheader("Finding a Repeating Signal (Periodogram)")
        fig2, ax2 = plt.subplots(); periodogram.plot(ax=ax2); st.pyplot(fig2)
        status_placeholder.info("🌟 Folding the light curve to reveal the transit...")
        folded_lc = clean_lc.fold(period=planet_period, epoch_time=planet_transit_time)
        st.subheader("Folded Light Curve: The Final Proof")
        fig3, ax3 = plt.subplots(); folded_lc.plot(ax=ax3); plt.title(f"Light Curve Folded at {planet_period.value:.4f} days"); st.pyplot(fig3)
        status_placeholder.success("🎉 Analysis Complete!")
    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        status_placeholder.empty(); progress_placeholder.empty()


@st.cache_data(ttl="1d")
def fetch_catalog_targets(mission_name, disposition_type, num_targets=25):
    # This function remains the same as the previous version.
    # (Code is omitted for brevity but should be included in your file)
    try:
        url, id_col, prefix, disposition_col, dispositions_to_find = "", "", "", "", []
        
        if mission_name == "TESS":
            url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
            id_col, prefix, disposition_col = 'TIC ID', 'TIC ', 'TFOPWG Disposition'
            if disposition_type == "PLANETS": dispositions_to_find = ["CP", "PC"]
            else: dispositions_to_find = ["FP"]
        elif mission_name in ["Kepler", "K2"]:
            url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&select=kepid,koi_disposition,koi_period,koi_prad&format=csv"
            id_col, prefix, disposition_col = 'kepid', 'KIC ', 'koi_disposition'
            if disposition_type == "PLANETS": dispositions_to_find = ["CONFIRMED", "CANDIDATE"]
            else: dispositions_to_find = ["FALSE POSITIVE"]
        else:
            return pd.DataFrame()

        comment_char = '#' if "kepler" in url.lower() else None
        catalog_df = pd.read_csv(url, comment=comment_char)
        filtered_df = catalog_df[catalog_df[disposition_col].isin(dispositions_to_find)]
        if filtered_df.empty: return pd.DataFrame()

        final_sample = filtered_df.sample(n=min(num_targets, len(filtered_df)))
        
        column_map = {
            id_col: "Searchable ID", disposition_col: "Status",
            'Period (days)': 'Orbital Period (days)', 'koi_period': 'Orbital Period (days)',
            'Planet Radius (R_earth)': 'Planet Radius (Earths)', 'koi_prad': 'Planet Radius (Earths)'
        }
        available_cols = [col for col in column_map.keys() if col in final_sample.columns]
        result_df = final_sample[available_cols].rename(columns=column_map)
        result_df["Searchable ID"] = prefix + result_df["Searchable ID"].astype(str)
        return result_df
    except Exception as e:
        st.warning(f"Could not dynamically fetch targets: {e}")
        return pd.DataFrame()

# ✅ UPDATED FUNCTION: Uses the correct filter 'plx' instead of 'distance'.
@st.cache_data(ttl="1d")
def fetch_untested_targets(num_to_sample=100):
    """
    Fetches a random sample of stars from the TESS Input Catalog (TIC) and
    filters out any that are already known TESS Objects of Interest (TOIs).
    """
    try:
        # Step 1: Get the full list of known planet candidates (TOIs)
        st.toast("Downloading the list of known planet candidates (TOIs)...")
        toi_url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
        toi_df = pd.read_csv(toi_url)
        known_toi_tics = set(toi_df['TIC ID'])

        # Step 2: Query a random sample of bright, nearby stars from the main TESS catalog
        st.toast("Fetching a random sample of stars from the TESS Input Catalog...")
        
        # ✅ CORRECTION: Filter by parallax `plx` (in milliarcseconds) instead of distance.
        # A larger parallax means a closer star. This range (2 to 100) is for nearby stars.
        tic_sample = Catalogs.query_criteria(
            catalog="TIC", Vmag=(9, 13), plx=(2, 100), pagesize=num_to_sample
        )
        tic_sample_df = tic_sample.to_pandas()
        
        # Step 3: Find the stars from our sample that are NOT on the TOI list
        untested_mask = ~tic_sample_df['ID'].isin(known_toi_tics)
        untested_df = tic_sample_df[untested_mask]

        # Step 4: Prepare the results for display
        # ✅ CORRECTION: Use the correct distance column name 'dst' returned by the query.
        untested_df = untested_df.rename(columns={'ID': 'Searchable ID', 'Tmag': 'TESS Magnitude', 'dst': 'Distance (pc)'})
        untested_df['Searchable ID'] = "TIC " + untested_df['Searchable ID'].astype(str)
        
        # Ensure all required columns exist before returning
        final_columns = ['Searchable ID', 'TESS Magnitude', 'Distance (pc)', 'ra', 'dec']
        available_final_cols = [col for col in final_columns if col in untested_df.columns]
        return untested_df[available_final_cols]

    except Exception as e:
        st.warning(f"Could not fetch untested targets: {e}")
        return pd.DataFrame()


# --- Main Application Interface ---
st.set_page_config(page_title="AI Exoplanet Hunter", layout="wide")
st.title("🔭 AI Exoplanet Hunter")
for key in ['search_result', 'explore_planets_results', 'explore_fps_results', 'untested_results']:
    if key not in st.session_state: st.session_state[key] = None

# --- Sidebar ---
st.sidebar.header("Search Parameters")
st.sidebar.info("These filters only apply when searching by name. They are ignored for numerical ID searches.")
available_missions = ["TESS", "Kepler", "K2"]
selected_missions = st.sidebar.multiselect("Select Mission(s)", options=available_missions, default=["TESS", "Kepler", "K2"])
available_authors = ["SPOC", "Kepler", "K2", "QLP", "TESS-SPOC"]
selected_authors = st.sidebar.multiselect("Select Data Author/Pipeline(s)", options=available_authors, default=["SPOC", "Kepler", "K2"])
st.sidebar.header("About")
st.sidebar.info("This web app helps analyze astronomical data to find exoplanets.")

# --- CREATE FOUR TABS ---
search_tab, planets_tab, fps_tab, untested_tab = st.tabs([
    "Search for a Star", "Explore Planet Candidates", "Explore False Positives", "Find Untested Targets"
])

with search_tab:
    # (This tab's code is unchanged and omitted for brevity, but should be included in your file)
    st.markdown("Enter the name of a star or its ID to search for exoplanet transits.")
    star_id_input = st.text_input(label="Enter a Star Name or ID", value="TIC 261136679", help="Try copying a full Searchable ID from the explore tabs!")
    if st.button("Search for Data", type="primary"):
        st.session_state.search_result = None
        if not star_id_input: st.warning("Please enter a star name.")
        else:
            with st.spinner(f"Querying for '{star_id_input}'..."):
                search_term = star_id_input.upper().replace("TIC", "").replace("KIC", "").replace("EPIC", "").strip()
                is_id_search = search_term.isdigit()
                if is_id_search:
                    st.info("Numerical ID detected. Searching all available missions and authors...")
                    result = lk.search_lightcurve(star_id_input)
                else:
                    if not selected_missions or not selected_authors:
                        st.sidebar.warning("Please select at least one mission and author for name searches.")
                        result = None
                    else:
                        result = lk.search_lightcurve(star_id_input, mission=selected_missions, author=selected_authors)
                if result is not None and len(result) > 0:
                    st.session_state.search_result = result
                else:
                    st.warning("No data found for the given criteria.")

    if st.session_state.search_result is not None:
        st.divider()
        st.subheader("Step 2: Select Data to Process")
        results_df = st.session_state.search_result.table.to_pandas()
        link_col_name = "Archive Link"
        if 'TIC ID' in results_df.columns:
            results_df[link_col_name] = "https://exofop.ipac.caltech.edu/tess/target.php?id=" + results_df['TIC ID'].astype(str)
        elif 'KIC ID' in results_df.columns:
            results_df[link_col_name] = "https://exoplanetarchive.ipac.caltech.edu/overview/" + results_df['KIC ID'].astype(str)
        st.dataframe(results_df, column_config={link_col_name: st.column_config.LinkColumn("Details", display_text="View on Archive ↗️")})
        options = [f"Data Product #{i}" for i in range(len(st.session_state.search_result))]
        selected_options = st.multiselect("Choose which data products to download:", options=options, default=options)
        if st.button("Process Selected Files", type="primary"):
            if not selected_options: st.warning("Please select at least one data product.")
            else:
                selected_indices = [int(opt.split('#')[-1]) for opt in selected_options]
                selected_data_products = st.session_state.search_result[selected_indices]
                process_selected_data(selected_data_products)

with planets_tab:
    st.header("Discover High-Interest Planet Candidates")
    st.markdown("Select a mission below to fetch a sample of stars with confirmed planets or planet candidates.")
    mission_choice_planets = st.selectbox("Select a mission:", options=["TESS", "Kepler"], index=0, key="planet_mission_select")
    if st.button("Fetch Planet Candidates", type="primary"):
        with st.spinner(f"Downloading {mission_choice_planets} catalog... (Cached daily)"):
            st.session_state.explore_planets_results = fetch_catalog_targets(mission_choice_planets, disposition_type="PLANETS")
    if 'explore_planets_results' in st.session_state and st.session_state.explore_planets_results is not None:
        if st.session_state.explore_planets_results.empty:
            st.error("No targets were found. This could be a temporary issue with the data archive.")
        else:
            st.markdown(f"#### Sample of Planet Candidates from **{mission_choice_planets}**:")
            st.dataframe(st.session_state.explore_planets_results)
            st.info("You can copy a 'Searchable ID' and paste it into the 'Search for a Star' tab.")

with fps_tab:
    st.header("Discover Non-Planet Signals (False Positives)")
    st.markdown("Find stars with signals that look like planets but are not. This is crucial for training the AI.")
    mission_choice_fps = st.selectbox("Select a mission:", options=["TESS", "Kepler"], index=0, key="fp_mission_select")
    if st.button("Fetch False Positives", type="primary"):
        with st.spinner(f"Downloading {mission_choice_fps} catalog... (Cached daily)"):
            st.session_state.explore_fps_results = fetch_catalog_targets(mission_choice_fps, disposition_type="FALSE_POSITIVES")
    if 'explore_fps_results' in st.session_state and st.session_state.explore_fps_results is not None:
        if st.session_state.explore_fps_results.empty:
            st.error("No targets were found for this category.")
        else:
            st.markdown(f"#### Sample of False Positives from **{mission_choice_fps}**:")
            st.dataframe(st.session_state.explore_fps_results)
            st.info("You can copy a 'Searchable ID' and paste it into the 'Search for a Star' tab.")

with untested_tab:
    st.header("Find Potentially Un-tested Targets")
    st.markdown("This tool fetches a random sample of bright, nearby stars and checks if they are already on the official TESS Object of Interest (TOI) list. Any star shown below has TESS data but is not a known planet candidate, making it a great place to search for new signals.")

    if st.button("Find Untested Stars", type="primary", key="fetch_untested"):
        with st.spinner("Cross-referencing TESS catalogs... (Cached daily, may take a moment the first time)"):
            st.session_state.untested_results = fetch_untested_targets()

    if 'untested_results' in st.session_state and st.session_state.untested_results is not None:
        if st.session_state.untested_results.empty:
            st.error("No untested targets were found in the random sample, or an error occurred.")
        else:
            st.markdown("#### Potentially Un-tested TESS Targets:")
            st.dataframe(st.session_state.untested_results)
            st.info("Copy a 'Searchable ID' and paste it into the 'Search for a Star' tab to analyze it.")