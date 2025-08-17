# FPL Free Agent — Full Project (Backend + Vite Frontend)

This bundle contains:
- **backend/** FastAPI API (+ optional server LLM via Ollama) and static serving
- **frontend/** Your Vite + React UI (build with Tailwind)

## Run locally

### 1) Frontend — build the UI
```bash
cd frontend
npm install
npm run build
```
This creates `frontend/dist/`.

### 2) Backend — start API + static serving
```bash
cd ../backend
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
Open http://127.0.0.1:8000 to see the UI (served by the backend).

## Dev notes
- If you want to run the Vite dev server separately (`npm run dev`), keep the backend running on port 8000 and set:
  ```
  VITE_API_BASE=http://127.0.0.1:8000
  ```
  in a `.env` (or `.env.development`) inside the `frontend/`. CORS is already allowed for :5173 and :3000.

## Optional: Server-LLM (no WebGPU required)
- Install Ollama and pull a tiny instruct model:
  ```powershell
  winget install Ollama.Ollama
  ollama pull qwen2.5:1.5b-instruct
  ```
- Copy `backend/.env.example` → `backend/.env` and adjust if needed.
- Restart the backend; POST to `/chat` from the UI (if wired) or a client.

## Troubleshooting
- **405 on OPTIONS** → backend CORS not applied or wrong origin. Ensure you restarted Uvicorn after edits.
- **CBC solver** → if PuLP complains, `pip install coin-or-cbc`.
- **Frontend 404s** → ensure you built the frontend (`dist/` exists) before hitting `/` on the backend.

Happy managing! ⚽
