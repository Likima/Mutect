## STR Predictions Web UI

This folder contains:
- Flask backend (`web/src/backend.py`) exposing the ML pipeline outputs
- React + TypeScript frontend (`web/frontend/`) to visualize predictions and loci

### Prerequisites
- Python 3.13+ (project uses `pyproject.toml`)
- Node.js 18+ (for the frontend)

### Install Python deps
From the repository root:
```bash
uv pip install -e .
```
This installs Flask and flask-cors (added in `pyproject.toml`) along with model deps.

### Run the backend
```bash
python web/src/backend.py
```
Environment variables:
- `OUTPUT_DIR` (default: `<repo>/output`) — where `predictions.json`, `predictions_strs_only.json`, and `predictions_summary.json` are read
- `FIGURES_DIR` (default: `<repo>/figures`) — dashboard images to serve and list

Endpoints:
- `GET /health`
- `GET /api/summary`
- `GET /api/predictions?only_strs=true&min_prob=0.5&motif=TCCAT&page=1&page_size=100`
- `GET /api/loci`
- `GET /api/figures`
- `GET /figures/...` (static image pass-through)

### Run the frontend (Vite)
```bash
cd web/frontend
npm install
npm run dev
```
The dev server runs on `http://localhost:5173` and proxies `/api` and `/figures` to `http://localhost:5001`.

### Build frontend
```bash
cd web/frontend
npm run build
```
The static bundle is output to `dist/`. You can serve it with any static file server. In dev we recommend running the Flask backend and the Vite dev server separately.


