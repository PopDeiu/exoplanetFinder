# exoplanetFinder (versiune în română)

Aceasta este o aplicație web construită cu **Streamlit** pentru a explora datele de la misiunile spațiale **TESS** și **Kepler** și pentru a căuta posibile exoplanete folosind metoda tranzitului.

Aplicația îți permite să:
- cauți o stea după nume sau ID (TIC / TOI / KIC) și să analizezi curba ei de lumină;
- explorezi sisteme care au deja candidați sau planete confirmate;
- vezi exemple de „false pozitive” (semnale care arată ca o planetă, dar nu sunt);
- găsești ținte potențial netestate, bune pentru noi descoperiri;
- ajustezi parametrii analizei (binning, sigma pentru outlieri, interval de perioade etc.).

## Structura de bază

- `exo_app.py` – pagina principală Streamlit (landing page, inițializare stare).
- `pages/` – paginile aplicației (căutare stea, candidați, false pozitive, ținte netestate, setări, explorator de stele, căutare TOI).
- `utils.py` – funcții de utilitate pentru interogarea arhivelor, prelucrarea curbelor de lumină și rularea Box Least Squares (BLS).
- `requirements.txt` – dependențele Python.
- `Dockerfile` – imaginea Docker pentru aplicație.
- `docker-compose.yml` – fișier de orchestrare pentru a porni aplicația rapid cu Docker Compose.

---

## Rulare locală (fără Docker)

1. Creează și activează un mediu virtual (recomandat):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # pe Linux / macOS
   # sau
   .venv\Scripts\activate    # pe Windows
   ```

2. Instalează dependențele:
   ```bash
   pip install -r requirements.txt
   ```

3. Rulează aplicația Streamlit:
   ```bash
   streamlit run exo_app.py
   ```

4. Deschide browserul la adresa afișată (în mod normal `http://localhost:8501`).

---

## Rulare cu Docker Compose 🐳

Pentru a porni aplicația folosind Docker & Docker Compose:

1. Asigură-te că ai instalat:
   - **Docker** (Engine / Desktop)
   - **Docker Compose** (integrat în versiunile moderne de Docker Desktop)

2. Din directorul proiectului (acolo unde se află `docker-compose.yml`), rulează:
   ```bash
   docker compose up --build
   ```

   (Pe unele sisteme comanda poate fi `docker-compose up --build`.)

3. După ce imaginile sunt construite și containerul pornește, aplicația va fi disponibilă de obicei la:
   ```
   http://localhost:8501
   ```

4. Pentru a opri aplicația:
   ```bash
   docker compose down
   ```

---

## Notă

- Aplicația folosește servicii externe (ExoFOP, MAST, etc.), deci are nevoie de acces la internet.
- Versiunea aceasta are interfața tradusă în **limba română**, dar logica internă și denumirile din cod rămân în mare parte în engleză, pentru compatibilitate cu bibliotecile științifice.
