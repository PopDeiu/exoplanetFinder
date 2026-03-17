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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from PIL import Image as PILImage

def generate_pdf_report(star_name, period, depth, radius, figure):
    """
    Generează un raport PDF cu rezultatele analizei exoplanetei.
    
    Parameters:
    - star_name: Numele stelei
    - period: Perioada găsită
    - depth: Adâncime (depth)
    - radius: Raza estimată
    - figure: Figure matplotlib cu graficul
    
    Returns:
    - bytes_pdf: Fișierul PDF în format bytes
    """
    
    # Creează un buffer pentru PDF
    pdf_buffer = BytesIO()
    
    # Creează documentul PDF în format landscape
    doc = SimpleDocTemplate(pdf_buffer, pagesize=landscape(letter))
    styles = getSampleStyleSheet()
    
    # Creează stiluri personalizate
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor="#000000",
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    normal_style = styles['Normal']
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor="#A132DD",
        spaceAfter=12,
        fontName='Helvetica-Bold'
    )
    
    # Lista de elemente pentru PDF
    elements = []
    
    # Titlu și mesaj de felicitări
    elements.append(Paragraph(f"Felicitari! Ai gasit o potentiala exoplaneta in jurul stelei <b><font color=\"#A132DD\">{star_name}</font></b>!", title_style))
    elements.append(Spacer(1, 0.15*inch))
    
    
    # Convertește figura matplotlib în imagine PNG
    img_buffer = BytesIO()
    figure.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    
    
    
    # Redimensionează imaginea să încapă pe pagină
    img = Image(img_buffer, width=7.5*inch, height=3.3*inch)
    elements.append(img)
    elements.append(Spacer(1, 0.15*inch))
    
    # Adaugă o nouă pagină pentru metrici
    
    
    # Metrici
    elements.append(Paragraph("Rezultatele Analizei", heading_style))
    elements.append(Spacer(1, 0.2*inch))
    
    metrics_text = f"""
    <b>Perioada Gasita:</b> {period:.4f} zile<br/>
    <b>Adancime:</b> {depth:.4f}<br/>
    """
    
    if radius:
        metrics_text += f"<b>Raza Estimata:</b> {radius:.2f} R⊕ (raze terestre)"
    
    elements.append(Paragraph(metrics_text, styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    
    
    # Construiește PDF-ul
    doc.build(elements)
    
    # Obține bytes-ul PDF-ului
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

def process_selected_data(selected_items, bin_minutes, outlier_sigma, period_min, period_max):
    apply_pro_plotting_style()
    status_placeholder = st.empty()
    
    try:
        target_id = selected_items.table.to_pandas()['target_name'][0]
        star_radius = get_star_parameters(target_id)
        
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
        fig1, ax1 = plt.subplots(figsize=(10, 3))
        ax1.plot(clean_lc.time.value, clean_lc.flux.value, color='#1E90FF', lw=0.5, alpha=0.8)
        ax1.set_ylabel("Flux"); ax1.set_xlabel("Timp (Zile)")
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
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(results.period, results.power, color='#1E90FF', lw=0.8)
        ax2.axvline(best_p, color="#B307AD", ls='--', alpha=0.7)
        ax2.set_xlabel("Perioada (zile)"); ax2.set_ylabel("Putere")
        st.pyplot(fig2)

        # PASUL 3: PLIERE ȘI MODEL
        st.subheader("Pasul 3: Pliere și Modelare")
        folded = clean_lc.fold(period=best_p, epoch_time=astropy.time.Time(best_t0, format=clean_lc.time.format))
        
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.scatter(folded.time.value, folded.flux.value, s=2, color='#1E90FF', alpha=0.5, label="Date")
        
        x_m = np.linspace(-0.5, 0.5, 1000)
        y_m = model.model(x_m + best_t0, best_p, best_dur, best_t0)
        ax3.plot(x_m, y_m, color="#B307AD", lw=2.5, label="Model BLS")
        
        ax3.set_xlim(-0.4, 0.4); ax3.legend(); ax3.grid(alpha=0.1)
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

        # Store data for PDF export
        st.session_state.pdf_export_data = {
            'star_name': target_id,
            'period': best_p,
            'depth': best_depth,
            'radius': r_e,
            'figure': fig3
        }


        status_placeholder.success("Analiza finalizata, poti descarca pdf-ul!")
        
    except Exception as e:
        st.error(f"Analiza a eșuat: {e}")

