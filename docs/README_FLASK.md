# BiasX PBL - Flask Frontend/Backend

This repository previously used Streamlit. This branch provides a Flask backend and a plain static frontend (HTML/JS) that you can host on GitHub Pages.

Quickstart (local)

1. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the Flask backend:

```bash
python flask_app.py
```

The backend runs on http://0.0.0.0:5000 by default.

3. Serve the static frontend (the simplest is to open `frontend/index.html` in a browser). For cross-origin convenience, run a simple static server locally (python 3):

```bash
cd frontend
python -m http.server 8000
# then open http://localhost:8000 in your browser
```

Notes
- The frontend is intentionally minimal (no React) to make deployment on GitHub Pages trivial. If you prefer React, I can scaffold a create-react-app frontend that calls the Flask API.
- The Flask backend keeps session state in-memory (a dict). For production, replace with database or other state store.

Deployment
- Frontend: host the `frontend/` folder on GitHub Pages.
- Backend: host `flask_app.py` on AWS EC2, Render, or another host with Python support. Ensure `requirements.txt` is installed on the server.

Endpoints
- GET /metrics
- POST /upload_dataset (multipart form 'dataset')
- POST /analyze (JSON)
- POST /mitigate (JSON)
- GET /download_dataset?session_id=...&which=mitigated

If you'd like, I can now:
- Replace the simple frontend with a React app (no SSR) and wire it to the backend.
- Convert in-memory session storage to a small SQLite-backed store.
- Add CI for deployment.

Tell me which next step you'd like me to take.
