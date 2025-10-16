# ML + Next.js Stack

This repository packages a FastAPI machine-learning microservice together with a Next.js frontend so you can serve predictions straight from the browser.

## Run locally (Docker)

```bash
docker compose build
docker compose up
```

- App: http://localhost:3000
- ML API health: http://localhost:8000/health

### Dev tips

- Put your trained model at `ml_service/models/model.joblib`.
- Call the API from the browser via `/api/predict` (Next.js proxies to Python).
- Update the UI in `web/app/page.tsx` to match your feature set.

---

## FastAPI service (`ml_service/`)

`main.py` exposes `/health` and `/predict`. The predict endpoint expects JSON of the shape:

```json
{
  "features": [0.1, 1.4, 2.3]
}
```

Place your serialized model in `ml_service/models/model.joblib`. If the file is missing the API serves a simple average-based baseline so you can verify the integration end-to-end; replace it with the production artifact generated from `ULTRA_OPTIMIZED_PRODUCTION_MODEL 4.py` when you are ready.

Run the Python service by itself:

```bash
cd ml_service
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Next.js app (`web/`)

The frontend provides a simple comma-separated input box that posts to `/api/predict`.

Standalone run without Docker:

```bash
cd web
cp .env.local.example .env.local   # set ML_URL=http://localhost:8000
npm install
npm run dev
```

Docker images are multi-stage to keep runtime small and production-friendly.
