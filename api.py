from fastapi import FastAPI, HTTPException, Query
from typing import Optional
from utils import get_all_naked_eye_stars, get_real_stars_by_bortle, get_connection
from utils import get_saved_location

# Inițializăm aplicația API
app = FastAPI(
    title="API Stele Vizibile",
    description="API pentru extragerea stelelor din baza de date și cataloage NASA",
    version="1.0.0"
)

@app.get("/")
def read_root():
    """Mesaj de bun venit pe ruta principală."""
    return {"mesaj": "Bun venit la API-ul Observatorului Stelar! Accesează /docs pentru documentație."}

@app.get("/api/stele/obseravtorVR")
def get_saved_stars():
    """
    Returnează stelele din baza de date locală, filtrate automat 
    pe baza latitudinii salvate tot în baza de date.
    """
    # 1. Luăm locația direct din baza de date
    lat, lon = get_saved_location()
    
    # 2. Cerem toate stelele din baza de date
    stele_toate = get_all_naked_eye_stars()
    
    # 3. Verificăm dacă avem stele
    if not stele_toate:
        return {
            "latitudine": lat,
            "longitudine": lon,
            "total": 0, 
            "date": [], 
            "mesaj": "Conexiunea la DB e OK, dar query-ul nu a găsit date salvate."
        }
        
    # 4. Filtrăm stelele dacă avem latitudinea salvată în DB
    stele_vizibile = []
    
    if lat is not None:
        min_dec = lat - 90.0
        
        for stea in stele_toate:
            try:
                dec_str = str(stea['declination']).replace('°', '').replace('"', '').replace("'", '').strip()
                dec_val = float(dec_str)
                
                if dec_val > min_dec:
                    stele_vizibile.append(stea)
            except ValueError:
                stele_vizibile.append(stea)
                
        stele_finale = stele_vizibile
    else:
        # Dacă nu s-a găsit locația în DB, dăm tot cerul
        stele_finale = stele_toate

    # 5. Răspunsul JSON final
    return {
        "latitudine_salvata": lat,
        "longitudine_salvata": lon,
        "total_gasite": len(stele_finale),
        "date": stele_finale
    }

@app.get("/api/stele/nasa/{bortle_level}")
def get_nasa_stars_by_bortle(
    bortle_level: int, 
    limit: int = Query(50, description="Numărul maxim de stele returnate"),
    
):
    """
    Ruta LIVE: Interoghează direct catalogul NASA (TIC) în funcție de nivelul Bortle.
    - bortle_level: între 1 (cer perfect) și 9 (centru oraș)
    - limit: câte stele să aducă maxim (default 50)
    - lat, lon: coordonatele observatorului
    """
    if bortle_level < 1 or bortle_level > 9:
        raise HTTPException(status_code=400, detail="Nivelul Bortle trebuie să fie între 1 și 9.")
        
    # Trimitem coordonatele mai departe către funcția din utils
    stele = get_real_stars_by_bortle(bortle_level, limit=limit)
    
    # Răspunsul JSON - elementele puse primele aici vor apărea primele în JSON
    return {
        "bortle_level": bortle_level,
        "total_gasite": len(stele),
        "date": stele
    }