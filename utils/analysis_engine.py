import streamlit as st
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import astropy.time
from astropy import units as u
from astropy.timeseries import BoxLeastSquares
from .ui_styles import apply_pro_plotting_style
from .data_fetchers import get_star_parameters
from io import BytesIO
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors
from PIL import Image as PILImage

def generate_pdf_report(star_name, period, depth, radius, figure, semi_major_axis=None, hz_inner=None, hz_outer=None, star_mass=None, star_teff=None, star_luminosity=None):
    """
    Generează un raport PDF cu rezultatele analizei exoplanetei.
    
    Parameters:
    - star_name: Numele stelei
    - period: Perioada găsită
    - depth: Adâncime (depth)
    - radius: Raza estimată
    - figure: Figure matplotlib cu graficul
    - semi_major_axis: Semi-axa mare (UA)
    - hz_inner: Zona locuibilă interioară (UA)
    - hz_outer: Zona locuibilă exterioară (UA)
    - star_mass: Masa stelei (M☉)
    - star_teff: Temperatura efectivă (K)
    - star_luminosity: Luminozitatea (L☉)
    
    Returns:
    - bytes_pdf: Fișierul PDF în format bytes
    """
    
    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer, pagesize=landscape(letter),
        topMargin=0.4*inch, bottomMargin=0.3*inch,
        leftMargin=0.5*inch, rightMargin=0.5*inch
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'],
        fontSize=20, textColor="#A132DD", spaceAfter=8,
        alignment=TA_CENTER, fontName='Helvetica-Bold'
    )
    cell_style = ParagraphStyle(
        'CellStyle', parent=styles['Normal'],
        fontSize=9, leading=13, spaceBefore=2, spaceAfter=2
    )
    cell_bold = ParagraphStyle(
        'CellBold', parent=cell_style,
        fontName='Helvetica-Bold'
    )

    elements = []

    # Titlu
    elements.append(Paragraph(
        f"Exoplaneta candidata in jurul stelei <b><font color=\"#A132DD\">{star_name}</font></b>",
        title_style
    ))
    elements.append(Spacer(1, 0.08*inch))

    # Graficul
    img_buffer = BytesIO()
    figure.savefig(img_buffer, format='png', dpi=130, bbox_inches='tight')
    img_buffer.seek(0)
    img = Image(img_buffer, width=7.8*inch, height=3.3*inch)
    elements.append(img)
    elements.append(Spacer(1, 0.08*inch))

    # Tabel cu 2 coloane
    data_rows = [
        [Paragraph("<b>Parametru</b>", cell_bold),
         Paragraph("<b>Valoare</b>", cell_bold),
         Paragraph("<b>Parametru</b>", cell_bold),
         Paragraph("<b>Valoare</b>", cell_bold)],
    ]

    def add_pair(left_label, left_val, right_label, right_val):
        data_rows.append([
            Paragraph(left_label, cell_bold),
            Paragraph(left_val, cell_style),
            Paragraph(right_label, cell_bold),
            Paragraph(right_val, cell_style),
        ])

    # Construim perechi stânga-dreapta
    pairs = [
        ("Perioada", f"{period:.4f} zile",
         "Raza planetei", f"{radius:.2f} R⊕" if radius else "N/A"),
        ("Adancime", f"{depth:.4f}",
         "Masa stelei", f"{star_mass:.3f} M☉" if star_mass else "N/A"),
        ("Temp. efectiva", f"{star_teff} K" if star_teff else "N/A",
         "Luminozitate", f"{star_luminosity:.4f} L☉" if star_luminosity else "N/A"),
        ("Semi-axa mare (a)", f"{semi_major_axis:.4f} UA" if semi_major_axis else "N/A",
         "Zona loc. interioara", f"{hz_inner:.4f} UA" if hz_inner else "N/A"),
        ("Formula a", "∛(M/M☉·(P/365.25)²)" if semi_major_axis else "N/A",
         "Zona loc. exterioara", f"{hz_outer:.4f} UA" if hz_outer else "N/A"),
    ]

    for l_label, l_val, r_label, r_val in pairs:
        add_pair(l_label, l_val, r_label, r_val)

    col_widths = [1.6*inch, 2.3*inch, 1.6*inch, 2.3*inch]
    table = Table(data_rows, colWidths=col_widths, repeatRows=1)
    table.setStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#A132DD")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('ALIGN', (3, 0), (3, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F0F7")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ])

    elements.append(table)

    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

def process_selected_data(selected_items, bin_minutes, outlier_sigma, period_min, period_max):
    apply_pro_plotting_style()
    status_placeholder = st.empty()
    
    try:
        target_id = selected_items.table.to_pandas()['target_name'][0]
        star_params = get_star_parameters(target_id)
        star_radius = star_params['rad'] if star_params else None
        star_mass = star_params['mass'] if star_params else None
        star_teff = star_params['Teff'] if star_params else None
        
        st.header("Rezultatele Analizei Pas cu Pas")
        
        # DESCĂRCARE
        lcs = []
        for i in range(len(selected_items)):
            status_placeholder.info(f"Se descarcă segmentul {i+1}...")
            data = selected_items[i].download()
            if data:
                lc = data.to_lightcurve(aperture_mask='pipeline') if hasattr(data, 'to_lightcurve') else data
                lcs.append(lc.normalize())
        
        full_lc = lk.LightCurveCollection(lcs).stitch().remove_nans()

        # PASUL 1: PREPROCESARE
        bin_days = bin_minutes / 1440.0
        binned_lc = full_lc.bin(time_bin_size=bin_days * u.day)
        clean_lc = binned_lc.flatten(window_length=101).remove_outliers(sigma=outlier_sigma)

        st.subheader("Pasul 1: Curățarea și Aplatizarea")
        st.caption("Curba de lumină este curățată de valori extreme (outlieri) și aplatizată prin eliminarea variațiilor de lungă durată (pete stelare, rotație). Acest pas este esențial pentru a evidenția tranzitele.")
        fig1, ax1 = plt.subplots(figsize=(10, 3))
        ax1.plot(clean_lc.time.value, clean_lc.flux.value, color='#1E90FF', lw=0.5, alpha=0.8)
        ax1.set_ylabel("Flux (luminozitate relativă)"); ax1.set_xlabel("Timp (zile)")
        st.caption("**Axa X:** Timpul în zile de la începutul observațiilor. **Axa Y:** Fluxul relativ al stelei (1.0 = luminozitatea medie). Scăderile sub 1.0 indică posibile tranzite.")
        st.pyplot(fig1)

        # PASUL 2: PERIODOGRAMA
        status_placeholder.info("Calculăm periodograma BLS...")
        model = BoxLeastSquares(t=clean_lc.time.value, y=clean_lc.flux.value)
        p_grid = np.linspace(period_min, period_max, 10000)
        results = model.power(p_grid, np.linspace(0.05, 0.2, 10))
        
        best_p = results.period[np.argmax(results.power)]
        best_t0 = results.transit_time[np.argmax(results.power)]
        best_dur = results.duration[np.argmax(results.power)]
        best_depth = results.depth[np.argmax(results.power)]

        st.subheader("Pasul 2: Identificarea Perioadei")
        st.caption("Algoritmul BLS (Box Least Squares) încearcă mii de perioade posibile și măsoară cât de probabil este un tranzit la fiecare. Vârful cel mai înalt indică perioada candidată.")
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(results.period, results.power, color='#1E90FF', lw=0.8)
        ax2.axvline(best_p, color="#B307AD", ls='--', alpha=0.7)
        ax2.set_xlabel("Perioada (zile)"); ax2.set_ylabel("Putere (puterea semnalului)")
        st.caption("**Axa X:** Perioada orbitală testată (zile). **Axa Y:** Puterea semnalului — cu cât vârful este mai înalt, cu atât perioada este mai probabilă. Linia mov marchează perioada cea mai bună.")
        st.pyplot(fig2)

        # PASUL 3: PLIERE ȘI MODEL
        st.subheader("Pasul 3: Pliere și Modelare")
        st.caption("Toate tranzitele sunt „împăturite” după perioada găsită, suprapunându-se pentru a evidenția forma tranzitului. Linia mov arată modelul BLS ajustat.")
        folded = clean_lc.fold(period=best_p, epoch_time=astropy.time.Time(best_t0, format=clean_lc.time.format))
        
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.scatter(folded.time.value, folded.flux.value, s=2, color='#1E90FF', alpha=0.5, label="Date")
        
        x_m = np.linspace(-0.5, 0.5, 1000)
        y_m = model.model(x_m + best_t0, best_p, best_dur, best_t0)
        ax3.plot(x_m, y_m, color="#B307AD", lw=2.5, label="Model BLS")
        
        ax3.set_xlim(-0.4, 0.4); ax3.legend(); ax3.grid(alpha=0.1)
        ax3.set_xlabel("Fază (fracție din perioadă)"); ax3.set_ylabel("Flux relativ")
        st.caption("**Axa X:** Faza orbitală (0 = mijlocul tranzitului, -0.5/0.5 = o jumătate de orbită). **Axa Y:** Fluxul relativ. Punctele albastre sunt măsurătorile individuale, linia mov este modelul BLS.")
        st.pyplot(fig3)

        # Metrici
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Perioada Găsită", f"{best_p:.4f} d")
        c2.metric("Adâncime", f"{best_depth:.4f}")
        r_e = None
        if star_radius:
            r_e = (star_radius * np.sqrt(best_depth) * u.R_sun).to(u.R_earth).value
            c3.metric("Raza Est.", f"{r_e:.2f} R⊕")

        # Calcule: distanță planetă-stea (semi-axa mare) și zona locuibilă
        semi_major_axis = None
        hz_inner = None
        hz_outer = None
        L_val = None

        if star_mass and best_p:
            # Legea a III-a a lui Kepler: a³ = G·M·P²/(4π²)
            # În unități solare: a (UA) = (M/M☉ · (P/365.25)²)^(1/3)
            semi_major_axis = round((star_mass * (best_p / 365.25) ** 2) ** (1/3), 4)

        if star_radius and star_teff:
            # L/L☉ = (R/R☉)² · (T/T☉)⁴,  T☉ = 5778 K
            L_val = (star_radius ** 2) * ((star_teff / 5778) ** 4)
            hz_inner = round(np.sqrt(L_val / 1.1), 4)
            hz_outer = round(np.sqrt(L_val / 0.53), 4)

        # Store data for PDF export
        st.session_state.pdf_export_data = {
            'star_name': target_id,
            'period': best_p,
            'depth': best_depth,
            'radius': r_e,
            'figure': fig3,
            'semi_major_axis': semi_major_axis,
            'hz_inner': hz_inner,
            'hz_outer': hz_outer,
            'star_mass': round(star_mass, 3) if star_mass else None,
            'star_teff': int(star_teff) if star_teff else None,
            'star_luminosity': round(L_val, 4) if L_val else None,
        }


        status_placeholder.success("Analiza finalizata, poti descarca pdf-ul!")
        
    except Exception as e:
        st.error(f"Analiza a eșuat: {e}")

