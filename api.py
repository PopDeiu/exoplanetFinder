from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from typing import Optional
from pydantic import BaseModel
from utils import get_all_naked_eye_stars, get_stars_bortle_by_level, get_connection, update_app_setting
from utils import get_saved_location
from utils.database import get_all_settings, get_all_lectii, get_all_scenarii, get_scenariu_by_id
import os


class SteaCurentaRequest(BaseModel):
    TIC_ID: str = "TIC"

# Inițializăm aplicația API
app = FastAPI(
    title="API Stele Vizibile",
    description="API pentru extragerea stelelor din baza de date și cataloage NASA",
    version="1.0.0"
)

@app.get("/api/setari")
def get_app_settings():
    """
    Returnează setările curente ale aplicației: 
    locația salvată, modul de timp și data observației.
    """
    settings = get_all_settings()
    
    if not settings:
        raise HTTPException(status_code=404, detail="Setările nu au putut fi găsite în baza de date.")
    
    # Opțional: Putem structura răspunsul mai frumos
    return {
        "status": "succes",
        "date_configurare": {
            "oras": settings.get("oras", "Nespecificat"), 
            "latitudine": settings.get("latitudine"),
            "longitudine": settings.get("longitudine"),
            "viteza": int(settings.get("viteza", 0)),
            "foloseste_data_curenta": settings.get("foloseste_data_curenta"),
            "data_si_ora_obs": settings.get("data_si_ora_obs"),
            "afisare_constelatii": settings.get("afisare_constelatii", "da"),
            "lectie_activa": int(settings.get("lectie_activa", 0))
        }
    }

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
#    stele_vizibile = []
    
#    if lat is not None:
#        min_dec = lat - 90.0
        
#        for stea in stele_toate:
#            try:
#                dec_str = str(stea['declination']).replace('°', '').replace('"', '').replace("'", '').strip()
#                dec_val = float(dec_str)
                
#                if dec_val > min_dec:
#                    stele_vizibile.append(stea)
#            except ValueError:
#                stele_vizibile.append(stea)
                
#        stele_finale = stele_vizibile
#    else:
        # Dacă nu s-a găsit locația în DB, dăm tot cerul
#        stele_finale = stele_toate

    # 5. Răspunsul JSON final
    return {
        "latitudine": lat,
        "longitudine": lon,
        "total_gasite": len(stele_toate),
        "date": stele_toate
    }

@app.post("/api/stea_curenta")
def set_current_star(request: SteaCurentaRequest):
    tic_id = request.TIC_ID.strip()
    if tic_id.upper().startswith("TIC ") or tic_id.upper() == "TIC":
        pass
    else:
        tic_id = f"TIC {tic_id}"
    update_app_setting("stea_curenta", tic_id)
    return {"status": "succes", "TIC_ID": tic_id}

@app.get("/api/assets/{filename:path}")
def serve_asset(filename: str):
    """Servește fișiere din directorul assets (inclusiv videoclipuri)."""
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    file_path = os.path.abspath(os.path.join(assets_dir, filename))
    # Security: ensure we don't escape the assets directory
    if not file_path.startswith(assets_dir) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Fișierul nu a fost găsit")
    return FileResponse(file_path)

@app.get("/api/lectii")
def get_lectii(user_id: Optional[int] = Query(None, description="ID-ul utilizatorului pentru a filtra lecțiile")):
    """
    Returnează toate lecțiile, fiecare cu scenariile aferente (expandate din IDs).
    Opțional, se poate filtra după user_id.
    """
    lectii = get_all_lectii(user_id)
    scenarii = get_all_scenarii(user_id)
    scenarii_map = {s["ID"]: s for s in scenarii}

    rezultat = []
    for lectie in lectii:
        ids_str = lectie.get("scenarii_ids", "")
        ids_list = [int(x) for x in ids_str.split(",") if x.strip().isdigit()]
        scenarii_lectie = []
        for sid in ids_list:
            if sid in scenarii_map:
                s = scenarii_map[sid]
                scenarii_lectie.append({
                    "id": s["ID"],
                    "nume": s["nume"],
                    "viteza": int(s["viteza"]),
                    "bortle": int(s["bortle"]),
                    "latitudine": float(s["latitudine"]),
                    "longitudine": float(s["longitudine"]),
                    "data_si_ora_obs": s["data_si_ora_obs"],
                    "foloseste_data_curenta": s["foloseste_data_curenta"],
                    "afisare_constelatii": s["afisare_constelatii"],
                    "text": s["text"],
                    "durata": int(s["durata"]) if s.get("durata") else 0
                })

        rezultat.append({
            "id": lectie["ID"],
            "nume": lectie["nume"],
            "descriere": lectie.get("descriere", ""),
            "user_id": lectie.get("user_id"),
            "scenarii": scenarii_lectie
        })

    return {
        "status": "succes",
        "total": len(rezultat),
        "date": rezultat
    }


@app.get("/api/lectii/{lectie_id}")
def get_lectie_by_id(lectie_id: int):
    """
    Returnează o lecție specifică, cu scenariile aferente expandate.
    """
    from utils.database import get_lectie_by_id as db_get_lectie

    lectie = db_get_lectie(lectie_id, user_id=None)
    if not lectie:
        raise HTTPException(status_code=404, detail="Lecția nu a fost găsită")

    ids_str = lectie.get("scenarii_ids", "")
    ids_list = [int(x) for x in ids_str.split(",") if x.strip().isdigit()]

    scenarii = []
    for sid in ids_list:
        s = get_scenariu_by_id(sid, user_id=None)
        if s:
            scenarii.append({
                "id": s["ID"],
                "nume": s["nume"],
                "viteza": int(s["viteza"]),
                "bortle": int(s["bortle"]),
                "latitudine": float(s["latitudine"]),
                "longitudine": float(s["longitudine"]),
                "data_si_ora_obs": s["data_si_ora_obs"],
                "foloseste_data_curenta": s["foloseste_data_curenta"],
                "afisare_constelatii": s["afisare_constelatii"],
                "text": s["text"],
                "durata": int(s["durata"]) if s.get("durata") else 0
            })

    return {
        "status": "succes",
        "date": {
            "id": lectie["ID"],
            "nume": lectie["nume"],
            "descriere": lectie.get("descriere", ""),
            "user_id": lectie.get("user_id"),
            "scenarii": scenarii
        }
    }


@app.get("/api/stele/{bortle_level}")
def get_stars_by_bortle(
    bortle_level: int,
):
    """
    Ruta LIVE: Returnează stelele din baza de date pentru un nivel Bortle specific.
    - bortle_level: între 1 (cer perfect) și 9 (centru oraș)
    """
    if bortle_level < 1 or bortle_level > 9:
        raise HTTPException(status_code=400, detail="Nivelul Bortle trebuie să fie între 1 și 9.")
        
    stele = get_stars_bortle_by_level(bortle_level)
    
    return {
        "bortle_level": bortle_level,
        "total_gasite": len(stele),
        "date": stele
    }
