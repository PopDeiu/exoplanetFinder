from fastapi import FastAPI, HTTPException, Query
from typing import Optional
from utils import get_all_naked_eye_stars, get_real_stars_by_bortle, get_connection

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
def get_saved_stars(
    lat: float = Query(None, description="Latitudinea locației (ex: 46.18 pentru Arad)"),
    lon: float = Query(None, description="Longitudinea locației (ex: 21.31 pentru Arad)")
):

    stele = get_all_naked_eye_stars()
    
    
    if not stele:
        return {"total": 0, "date": [], "mesaj": "Conexiunea la DB e OK, dar query-ul nu a găsit date."}
        
    return {
        "total": len(stele),
        "date": stele
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