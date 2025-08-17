import os, re, json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd

from fpl_api import fetch_bootstrap, fetch_fixtures, build_players_df, opponent_map
from features import compute_expected_points
from optimizer import optimize

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

app = FastAPI(title="FPL Free Agent API")

# Allow dev origins; when serving the built UI from the same origin, CORS is unnecessary.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptIn(BaseModel):
    gw: int
    formation: str | None = None
    budget: float = 100.0
    bench_weight: float = 0.1
    # NEW:
    form_weight: float = 0.6
    fdr_weight: float = 0.3

class ChatIn(BaseModel):
    message: str

def _format_output(xi, bench, captain, vice, budget, objective, teams, opp):
    def lines(df):
        out = []
        for _, r in df.iterrows():
            tid_series = teams.loc[teams["team_short"]==r["team_short"], "team_id"]
            tid = int(tid_series.values[0]) if not tid_series.empty else None
            opps = " / ".join(opp.get(tid, [])) if tid is not None else "—"
            out.append({
                "name": r["name"], "pos": r["pos"], "team": r["team_short"],
                "opponent": opps, "price": round(r["now_cost"]/10.0,1),
                "xPts": round(float(r["exp_pts"]),2)
            })
        return out
    gk = bench[bench["pos"]=="GKP"].head(1)
    of = bench[bench["pos"]!="GKP"].sort_values("exp_pts", ascending=False).head(3)
    bench_view = pd.concat([gk, of])
    total_value = float(((xi["now_cost"].sum() + bench_view["now_cost"].sum()) / 10.0))
    bank = float(budget - total_value)
    return {
        "captain": captain, "vice": vice,
        "starting_XI": lines(xi),
        "bench": lines(bench_view),
        "total_value": round(total_value, 1),
        "bank": round(bank, 1),
        "objective_xpts": round(float(objective), 2),
    }

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/optimize")
def optimize_endpoint(inp: OptIn):
    bootstrap = fetch_bootstrap()
    teams = pd.DataFrame(bootstrap["teams"])[["id","name","short_name"]].rename(
        columns={"id":"team_id","name":"team_name","short_name":"team_short"}
    )
    fixtures = fetch_fixtures(inp.gw)
    players = build_players_df(bootstrap)
    # ---- NEW: pass weights + fixtures into expected points
    players = compute_expected_points(
        players, teams, fixtures,
        form_weight=inp.form_weight,
        fdr_weight=inp.fdr_weight,
    )
    opp = opponent_map(fixtures, teams)

    xi, bench, c, v, squad, obj = optimize(players, inp.budget, inp.formation, inp.bench_weight)
    return _format_output(xi, bench, c, v, inp.budget, obj, teams, opp)

@app.post("/chat")
def chat_endpoint(inp: ChatIn):
    if OpenAI is None:
        return JSONResponse({"text": "Server LLM not enabled. Install openai and set LLM_API_BASE/LLM_MODEL to use /chat."})
    import os
    LLM_API_BASE = os.getenv("LLM_API_BASE", "http://127.0.0.1:11434/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b-instruct")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
    client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)

    SYSTEM_PROMPT = (
        'You are an FPL assistant. You can call ONE tool named optimize. '
        'When you decide to call it, output EXACTLY one JSON line and nothing else: '
        '{"tool":"optimize","args":{"gw":<int>,"formation":"3-4-3","budget":100.0}} '
        'If the user omits values, infer sensible defaults.'
    )
    rsp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":inp.message}],
        temperature=0.2,
    )
    text = rsp.choices[0].message.content or ""
    m = re.search(r'\{"tool"\s*:\s*"optimize"[\s\S]*?\}\s*\}', text)
    if not m:
        return {"text": text.strip() or "I couldn't infer args. Try: 'gw=1, 3-4-3, £100m'."}
    call = json.loads(m.group(0))
    args = call.get("args", {})
    gw = int(args.get("gw", 1)); formation = args.get("formation") or "3-4-3"
    budget = float(args.get("budget", 100.0)); bench_weight = float(args.get("bench_weight", 0.1))

    bootstrap = fetch_bootstrap()
    teams = pd.DataFrame(bootstrap["teams"])[["id","name","short_name"]].rename(
        columns={"id":"team_id","name":"team_name","short_name":"team_short"}
    )
    fixtures = fetch_fixtures(gw)
    players = build_players_df(bootstrap)
    players = compute_expected_points(players)
    opp = opponent_map(fixtures, teams)
    xi, bench, c, v, squad, obj = optimize(players, budget, formation, bench_weight)
    res = _format_output(xi, bench, c, v, budget, obj, teams, opp)

    lines = [
        f"Captain: {res['captain']} | Vice: {res['vice']}",
        "",
        "Starting XI:",
        *[f" - {p['name']} ({p['pos']} – {p['team']}) vs {p['opponent']} | £{p['price']}m | xPts {p['xPts']}" for p in res["starting_XI"]],
        "",
        "Bench (GK + 3):",
        *[f" - {p['name']} ({p['pos']} – {p['team']}) vs {p['opponent']} | £{p['price']}m | xPts {p['xPts']}" for p in res["bench"]],
        "",
        f"Total Value: £{res['total_value']}m | Bank: £{res['bank']}m | Objective xPts: {res['objective_xpts']}"
    ]
    return {"text": "\n".join(lines)}

# ---- Serve Vite build (frontend/dist) from same origin ----
FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
else:
    @app.get("/")
    def root():
        return {"message": "Frontend build not found. Run `npm run build` in the frontend and place `dist/` at ../frontend/dist."}
