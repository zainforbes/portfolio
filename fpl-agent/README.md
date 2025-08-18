# FPL Free Agent

A free, local-first Fantasy Premier League squad builder.  
- **Optimizer**: integer program (PuLP/CBC) respects budget, positions, team caps, and optional formation.  
- **Expected Points**: blends **recent form vs season PPG** and **next-fixture difficulty (FDR)**.  
- **UI**: Vite + React + Tailwind (high-contrast).  
- **LLM (optional)**: chat to a tiny local model via **Ollama**; otherwise go fully headless.

---

## ✨ Features

- Live FPL data from official endpoints (no API key):
  - `bootstrap-static` (players, prices, form, PPG)
  - `fixtures?event=GW` (home/away + difficulty)
- Knobs that matter:
  - **Formation** (e.g. `3-4-3`, or leave flexible)
  - **Budget** (£m)
  - **Form weight** (how much to favor *recent form* vs *season PPG*)
  - **FDR weight** (how strongly to penalize/boost by fixture difficulty)
  - **Bench weight** (how much bench xPts count in the objective)
- Shows **captain/vice**, opponents, prices, xPts, total value & bank.
- Single-origin build: backend serves the frontend → **no CORS headaches**.
- Optional **/chat** endpoint to parse natural language when Ollama is enabled.

---

## 🧱 Tech Stack

- **Frontend**: React + Vite + TypeScript + TailwindCSS  
- **Backend**: FastAPI (Uvicorn)  
- **Solver**: PuLP (CBC)  
- **LLM (optional)**: Ollama (OpenAI-compatible)

---

## 📦 Project Structure

```
.
├─ backend/
│  ├─ app.py              # API (+ serves frontend/dist)
│  ├─ fpl_api.py          # FPL fetch & shaping
│  ├─ features.py         # xPts = mix(form, PPG) * FDR factor * play_prob
│  ├─ optimizer.py        # ILP constraints & objective
│  ├─ requirements.txt
│  └─ .env.example        # LLM config (optional)
└─ frontend/
   ├─ index.html
   ├─ src/
   │  ├─ main.tsx
   │  ├─ App.tsx
   │  └─ index.css
   ├─ tailwind.config.js
   └─ vite.config.ts
```

---

## 🚀 Quickstart

### 1) Build the UI
```bash
cd frontend
npm install
npm run build
```
This produces `frontend/dist/`.

> **Windows PowerShell note:** if `npm.ps1` isn’t signed, either use **Command Prompt** (`cmd`) or run `npm.cmd install && npm.cmd run build`.

### 2) Run the backend (serves API + UI)
```bash
cd ../backend
python -m venv .venv
# PowerShell:
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Open **http://127.0.0.1:8000**.

---

## ⚙️ Configuration

### Frontend
- **Production** (served by FastAPI): set **empty** API base so requests are same-origin:
  ```
  frontend/.env
  VITE_API_BASE=
  ```
- **Dev (Vite on 5173)**:
  ```
  VITE_API_BASE=http://127.0.0.1:8000
  ```

Ensure Tailwind scans your files:
```js
// tailwind.config.js
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx,js,jsx}"],
  theme: { extend: {} },
  plugins: [],
};
```

### Backend (.env) — optional LLM
```
LLM_API_BASE=http://127.0.0.1:11434/v1
LLM_MODEL=qwen2.5:1.5b-instruct
LLM_API_KEY=ollama
```
> Install Ollama: `winget install Ollama.Ollama` then `ollama pull qwen2.5:1.5b-instruct`.

---

## 🔌 API

### `POST /optimize`
**Body**
```json
{
  "gw": 3,
  "formation": "4-4-2",
  "budget": 100.0,
  "bench_weight": 0.1,
  "form_weight": 0.6,
  "fdr_weight": 0.3
}
```

**Response**
```json
{
  "captain": "Salah",
  "vice": "Haaland",
  "starting_XI": [
    { "name": "A. Becker", "pos": "GKP", "team": "LIV", "opponent": "ARS (H)", "price": 5.5, "xPts": 5.1 }
  ],
  "bench": [],
  "total_value": 99.8,
  "bank": 0.2,
  "objective_xpts": 47.85
}
```

**How xPts are computed**
```
base = form_weight * FORM + (1 - form_weight) * PPG
play_prob = chance_of_playing_next_round / 100
fdr_factor = avg(FDR_FACTOR for team’s GW fixtures)
mix = (1 - fdr_weight) + fdr_weight * fdr_factor
exp_pts = play_prob * base * mix + small_positional_bias
```
`FDR_FACTOR` (default): 1→1.10, 2→1.05, 3→1.00, 4→0.92, 5→0.85

### `POST /chat` (optional)
- Accepts `{ "message": "Best squad for GW3 with 4-4-2 and £100m" }`.
- When LLM is enabled, the server parses the request and internally calls `/optimize`.
- Returns a plain‑text summary.

### `GET /healthz`
- Readiness probe: `{ "ok": true }`.

---

## 🧮 Optimizer (constraints)

- 15 total players; starters = 11  
- Positions: 2 GKP, 5 DEF, 5 MID, 3 FWD  
- ≤ 3 per real team  
- Budget in **£m** (`now_cost/10`)  
- Formation:
  - If provided (e.g. `4-4-2`), enforce exactly D‑M‑F
  - Else: `DEF ≥ 3`, `MID ≥ 2`, `FWD ≥ 1`
- Objective:
  ```
  maximize  Σ(start_i * xPts_i) + bench_weight * Σ((pick_i - start_i) * xPts_i)
  ```
  Captain and vice = top two xPts starters (UI display only).

---

## 🛠️ Development

- Run Vite and backend separately:
  ```bash
  # terminal 1
  cd frontend && npm run dev  # http://127.0.0.1:5173
  # terminal 2
  cd backend && uvicorn app:app --reload --port 8000
  ```
- In dev, set `VITE_API_BASE=http://127.0.0.1:8000`.

**CORS** is pre-allowed for `:5173`, `:3000`, and backend `:8000`.  
Avoid mixing `localhost` and `127.0.0.1` in the browser vs API URL.

---

## 🐞 Troubleshooting

- **PowerShell blocks `npm`**: use **Command Prompt** or run the `.cmd` shim  
  `& "C:\Program Files\nodejs\npm.cmd" install`
- **`OPTIONS /optimize 400`**: use relative URLs (`/optimize`) or add your origin to CORS.
- **`vite.svg` 404**: remove the icon from `index.html` or add your own to `frontend/public/`.
- **Top-level `await`**: keep awaits inside functions; don’t call the API at module top level.
- **CBC missing**: `pip install coin-or-cbc` (optional; PuLP ships CBC).
- **Port 11434 in use**: stop existing Ollama or change `LLM_API_BASE`.

---

## 📤 Deploy

Minimal:
1) `npm run build` in `frontend/`
2) `uvicorn app:app` in `backend/` (serves `frontend/dist`)

For Docker: build the frontend, COPY `frontend/dist` into the image with the backend, run `uvicorn` behind nginx/Caddy.

---

## ⚠️ Disclaimer

This project is **unofficial** and not affiliated with the Premier League, FPL, or any club.  
Respect FPL’s terms; don’t hammer their endpoints.

---

## 📄 License

MIT 