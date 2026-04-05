#!/bin/bash

# Pornim FastAPI în fundal (simbolul & de la final face magia asta)
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Pornim Streamlit în prim-plan (fără &)
# Acest proces va ține containerul activ
streamlit run Acasa.py --server.port=8501 --server.address=0.0.0.0