# FPL Headless Squad Optimizer (Portfolio Edition)

A fast, **no-LLM** Fantasy Premier League squad builder that:
- Pulls **official FPL JSON** data (no scraping, no API keys)
- Computes lightweight **expected points**
- Solves a deterministic **integer program** (ILP) to pick:
  - **15-player squad** under budget and rules
  - **Starting XI** (valid formation)
  - **Bench** (1 GK + best 3 outfield)
  - **Captain / Vice-Captain** (highest xPts starters)
- Renders a clean **ASCII “pitch”** in your terminal showing:
  - Formation rows (FWD/MID/DEF/GKP)
  - Opponent(s) for the chosen Gameweek with **H/A**
  - Player price, expected points, **Total Squad Value** and **Bank**

> Built for laptops with **8 GB RAM**. No LLM, no credits required.

---

## Demo

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Starting XI   Formation: 3-4-3                      │
│                          Captain: Haaland   Vice: Saka                       │
│                                                                              │
│                 Haaland (C)              Watkins                 Pedro       │
│                 MCI vs FUL (H)           AVL vs WOL (A)          BHA vs NEW (H)
│                 £14.0 | 7.2              £8.0 | 5.9              £6.6 | 5.1  │
│                                                                              │
│      Saka (VC)          Foden            Palmer              Gordon          │
│      ARS vs NFO (H)     MCI vs FUL (H)   CHE vs EVE (A)       NEW vs BHA (A) │
│      £8.5 | 6.9         £7.6 | 6.2       £6.0 | 5.4           £6.5 | 5.2     │
│                                                                              │
│        Estupiñán           Gabriel             Saliba                         │
│        BHA vs NEW (H)      ARS vs NFO (H)      ARS vs NFO (H)                 │
│        £5.0 | 5.1          £5.0 | 4.9          £5.5 | 4.8                     │
│                                                                              │
│                                   Raya                                      │
│                              ARS vs NFO (H)                                 │
│                               £5.5 | 4.1                                    │
│                                                                              │
│                  Total Squad Value: £99.5m   |   Bank: £0.5m                 │
│                      Objective xPts (XI + bench*0.1): 64.30                  │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                                Bench (GK + 3)                                │
│                                                                              │
│                 Areola                 Archer                 Kabore          │
│                 WHU vs BOU (A)         SHU vs CRY (H)         LUT vs BHA (A) │
│                 £4.0 | 3.5             £4.5 | 2.1             £4.0 | 2.0     │
└──────────────────────────────────────────────────────────────────────────────┘
```

*(Illustrative only; live output depends on the current season data.)*

---

## Features

- **Rules-aware** selection:
  - £100.0m default budget (configurable)
  - 15-player squad with position quotas: 2 GKP / 5 DEF / 5 MID / 3 FWD
  - Max **3 per real club**
  - Valid XI: GK=1 and standard FPL formations (3–5 at the back; 1–3 up front)
- **Explainable heuristic** for expected points:
  - Uses `ep_next` when FPL provides it, otherwise combines **form**, **PPG**, and **playing chance**
- **Pretty terminal output**:
  - Formation rows, opponent(s) with home/away marker
  - Bank and total squad value

---

## Install

### 1) Python env

```bash
python -m venv .venv
# Windows PowerShell:
. .venv/Scripts/Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

> If PuLP complains about a solver:  
> - Try `pip install coin-or-cbc` (bundled on many platforms), or  
> - On conda: `conda install -c conda-forge coincbc`

### 2) Run

```bash
python fpl_headless.py --gw 1 --budget 100.0 --formation 3-4-3
```

**Common flags**
- `--gw <int>`: Gameweek (required)
- `--budget <float>`: budget in millions (default `100.0`)
- `--formation "<D-M-F>"`: e.g., `3-4-3` (optional; inferred if omitted)
- `--bench-weight <float>`: value for bench in the objective (default `0.1`)
- `--no-color`: disable ANSI colors in terminals that don’t support them

---

## How it works (architecture)

1. **Fetch**  
   - `bootstrap-static` & `fixtures` from official FPL endpoints
2. **Assemble**  
   - Build a player dataframe (costs in tenths of £m, positions, team)
3. **Score**  
   - Compute **expected points** per player (use FPL `ep_next` when available)
4. **Optimize**  
   - ILP picks **15** under budget & rules  
   - Choose **XI** + **captain/vice** (top expected points starters)
5. **Render**  
   - ASCII pitch for XI, separate pitch for bench, bank & value

---

## Tech stack

- **Python 3.10+**
- `requests` — official FPL JSON
- `pandas` — tabular transforms
- `pulp` — integer programming (CBC solver)
- *(No LLMs, no API keys.)*

---

## Notes & extensions

- Add **fixture difficulty (FDR)** per opponent row
- Add **chip logic** (TC, FH, BB)
- Respect **existing team** and **transfer constraints** (hits, bank)
- Persist a small **cache** to avoid repeated HTTP calls during tinkering

---

## License

MIT. Do whatever you like—credit appreciated.
