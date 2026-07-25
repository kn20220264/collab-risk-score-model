# Collab Risk Score

Alata za procjenu rizika saradnje brenda sa YouTube
kreatorom. Na osnovu YouTube handle-a i opisa/naziva brenda, sistem
prikuplja podatke o kanalu i računa skor rizika (0–100) kroz četiri
modula: kvantitativne metrike, autentičnost, sentiment i brand-fit —
ponderisane ROC metodom, uz risk cap mehanizam i AI obrazloženje.

## Pokretanje

### Backend

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

`.env` u root-u projekta:

```
YOUTUBE_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

```bash
uvicorn backend.main:app --reload
```

API: `http://127.0.0.1:8000` (dokumentacija na `/docs`).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard: `http://localhost:5173`.
