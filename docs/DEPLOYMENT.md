# Deployment Guide – WordleAI API & Web UI

This document outlines one simple way to host the **WordleAI API** and the
**web UI** so users can play and watch the solver online.

## 1. Backend (FastAPI) – API Server

### 1.1. Local run

```bash
pip install -r requirements.txt
python -m src.api.server
```

This starts a FastAPI server at `http://localhost:8000` with:

- API docs: `http://localhost:8000/docs`
- Solver endpoints:
  - `POST /api/solve/step`
  - `POST /api/solve/reset`
- Web UI (static): `http://localhost:8000/web/`

### 1.2. Production command

Use `uvicorn` (or `gunicorn` + `uvicorn.workers.UvicornWorker`) in production:

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

## 2. Frontend – Web UI

The web UI is a static site under `web/public`.

- `web/public/index.html` – main HTML shell and styles
- `web/public/src/app.js` – UI logic for:
  - 6×5 board that resembles the NYT Wordle grid
  - On-screen keyboard (and physical keyboard support)
  - Clickable feedback tiles cycling `X → Y → G`
  - Suggestions panel with a **strategy selector**:
    - Information theory
    - ML (classic)
    - ML (min guesses)
  - Calls `POST /api/solve/step` on the backend

When served via `src.api.server`, the UI is available at `/web/`.

## 3. Hosting on a PaaS (example: Render / Railway / Fly.io)

### 3.1. Repository structure

Push this repository to GitHub/GitLab. The important paths:

- `requirements.txt`
- `src/api/server.py`
- `web/public/index.html`
- `web/public/src/app.js`

### 3.2. Service configuration

Create a new **Web Service** with:

- **Runtime**: Python 3.10+  
- **Build command**:

  ```bash
  pip install -r requirements.txt
  ```

- **Start command**:

  ```bash
  uvicorn src.api.server:app --host 0.0.0.0 --port $PORT
  ```

  (Many platforms inject `PORT` as an environment variable.)

### 3.3. Environment variables

For this project, the defaults are minimal:

- `MODEL_DIR` (optional) – where ML models are stored, defaults to `./models`
- `DATA_DIR` (optional) – where word lists live, defaults to `./data`

Most platforms allow you to configure these in a dashboard.

### 3.4. Static files

The FastAPI app already mounts:

- `/web` → `web/public`

So users visiting `https://your-app-domain/web/` will see the Wordle-like UI,
which calls the API at the same origin (`/api/solve/step`).

## 4. Verification checklist

After deployment:

1. Visit `https://your-app-domain/docs` – FastAPI docs load.
2. Visit `https://your-app-domain/web/` – board and side panel render.
3. Click “Get suggestion”:
   - A starting word appears in the side input.
4. Play that word in the real NYT Wordle, then:
   - Click each feedback tile until it matches the colors:
     - `X` = gray
     - `Y` = yellow
     - `G` = green
   - Click “Get suggestion” again to receive the next word.

If all steps work, the app is successfully hosted and ready for users.


