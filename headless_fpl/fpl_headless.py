# fpl_headless.py
# Headless FPL squad optimizer with pretty ASCII pitch, opponent display, and bank/value.
# Usage:
#   python fpl_headless.py --gw 1 --budget 100.0 --formation 3-4-3
# Optional:
#   --no-color  (disable ANSI colors)

import argparse
import requests
import pandas as pd
import pulp

# -------------------- Config & Endpoints --------------------
BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES = "https://fantasy.premierleague.com/api/fixtures/"
POSITION_COUNTS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_TEAM = 3
STARTERS = 11
FORMATIONS = {(3,4,3),(3,5,2),(4,4,2),(4,3,3),(4,5,1),(5,4,1),(5,3,2),(5,2,3)}

# -------------------- Pretty terminal helpers --------------------
ANSI_GREEN = "\033[92m"
ANSI_CYAN = "\033[96m"
ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"
ANSI_DIM = "\033[2m"
ANSI_BOLD = "\033[1m"

def colorize(text: str, color: str, use_color: bool) -> str:
    return f"{color}{text}{ANSI_RESET}" if use_color else text

def truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:max(0, n-1)] + "…"

def make_card(name, team_short, opp_str, price_m, xpts, is_c=False, is_vc=False, width=18):
    tag = "(C)" if is_c else "(VC)" if is_vc else ""
    line1 = truncate(f"{name} {tag}".strip(), width)
    line2 = truncate(f"{team_short} vs {opp_str}", width)
    line3 = truncate(f"£{price_m:.1f} | {xpts:.1f}", width)
    return [line1, line2, line3]

def center_cells(cells: list[list[str]], pitch_width_chars=80):
    if not cells:
        return []
    cell_w = max(len(l) for c in cells for l in c)
    pad = 2
    row_w = len(cells) * cell_w + (len(cells)-1) * pad
    left = max(0, (pitch_width_chars - row_w) // 2)
    lines = []
    height = len(cells[0])
    for r in range(height):
        row = " " * left
        for idx, c in enumerate(cells):
            row += c[r].ljust(cell_w)
            if idx < len(cells)-1:
                row += " " * pad
        lines.append(row)
    return lines

def draw_pitch(rows: list[list[list[str]]], title_lines: list[str] = None, footer_lines: list[str] = None):
    W = 80
    top = "┌" + "─" * (W-2) + "┐"
    mid = "│" + " " * (W-2) + "│"
    bot = "└" + "─" * (W-2) + "┘"
    out = [top]
    if title_lines:
        for t in title_lines:
            t = truncate(t, W-4)
            pad_left = (W-2-len(t))//2
            out.append("│" + " " * pad_left + t + " " * (W-2-len(t)-pad_left) + "│")
        out.append(mid)
    for ri, row in enumerate(rows):
        centered = center_cells(row, W-2)
        for ln in centered:
            out.append("│" + ln + " " * (W-2-len(ln)) + "│")
        if ri < len(rows)-1:
            out.append(mid)
    if footer_lines:
        out.append(mid)
        for f in footer_lines:
            f = truncate(f, W-4)
            pad_left = (W-2-len(f))//2
            out.append("│" + " " * pad_left + f + " " * (W-2-len(f)-pad_left) + "│")
    out.append(bot)
    return "\n".join(out)

# -------------------- Data fetching & features --------------------
def fetch_bootstrap():
    r = requests.get(BOOTSTRAP, timeout=30); r.raise_for_status(); return r.json()

def fetch_fixtures(gw:int) -> pd.DataFrame:
    r = requests.get(FIXTURES, params={"event": gw}, timeout=30); r.raise_for_status()
    return pd.DataFrame(r.json())

def build_players_df(bootstrap:dict) -> pd.DataFrame:
    elements = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])[["id","name","short_name"]].rename(
        columns={"id":"team_id","name":"team_name","short_name":"team_short"}
    )
    et = pd.DataFrame(bootstrap["element_types"])[["id","singular_name_short"]].rename(
        columns={"id":"element_type","singular_name_short":"pos"}
    )
    df = elements.merge(et, on="element_type", how="left").merge(
        teams, left_on="team", right_on="team_id", how="left"
    )
    keep = ["id","web_name","now_cost","pos","team_name","team_short",
            "points_per_game","form","chance_of_playing_next_round","ep_next"]
    for k in keep:
        if k not in df: df[k] = None
    df = df[keep].rename(columns={"web_name":"name"})
    for c in ["points_per_game","form","ep_next","chance_of_playing_next_round"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_expected_points(players: pd.DataFrame) -> pd.DataFrame:
    df = players.copy()
    if df["ep_next"].notna().any():
        df["exp_pts"] = df["ep_next"].fillna(0.0)
    else:
        play_prob = df["chance_of_playing_next_round"].fillna(100.0)/100.0
        base = 0.7*df["form"].fillna(0) + 0.3*df["points_per_game"].fillna(0)
        df["exp_pts"] = (base.clip(lower=0) * play_prob * 1.0)
    bias = df["pos"].map({"GKP":0.6,"DEF":0.4,"MID":0.2,"FWD":0.3}).fillna(0)
    df["exp_pts"] = (df["exp_pts"] + bias).clip(lower=0)
    return df

def opponent_map(fixtures: pd.DataFrame, teams: pd.DataFrame):
    """Return dict: team_id -> list of opponent short_names (with H/A)."""
    opp = {}
    # pre-lookup for speed
    short_by_id = {int(r.team_id): r.team_short for _, r in teams.iterrows()}
    for _, row in fixtures.iterrows():
        if pd.isna(row.get("team_h")) or pd.isna(row.get("team_a")):
            continue
        home = int(row["team_h"]); away = int(row["team_a"])
        home_team = short_by_id.get(home, "?")
        away_team = short_by_id.get(away, "?")
        opp.setdefault(home, []).append(f"{away_team} (H)")
        opp.setdefault(away, []).append(f"{home_team} (A)")
    return opp

# -------------------- Optimizer --------------------
def optimize(players: pd.DataFrame, budget_million: float, formation_hint: str|None, bench_weight: float=0.1):
    df = players.copy().reset_index(drop=True)

    form_tuple = None
    if formation_hint:
        try:
            form_tuple = tuple(int(x) for x in formation_hint.split("-"))
        except:
            form_tuple = None

    prob = pulp.LpProblem("FPL", pulp.LpMaximize)
    pick = pulp.LpVariable.dicts("pick", df.index, 0, 1, cat="Binary")
    start = pulp.LpVariable.dicts("start", df.index, 0, 1, cat="Binary")

    exp = df["exp_pts"].fillna(0.0)
    prob += pulp.lpSum(start[i]*exp[i] + (pick[i]-start[i])*bench_weight*exp[i] for i in df.index)
    prob += pulp.lpSum(pick[i] for i in df.index) == 15
    prob += pulp.lpSum(pick[i]*df.loc[i,"now_cost"] for i in df.index) <= int(round(budget_million*10))

    for pos, cnt in POSITION_COUNTS.items():
        prob += pulp.lpSum(pick[i] for i in df.index if df.loc[i,"pos"]==pos) == cnt

    for team in df["team_name"].dropna().unique():
        prob += pulp.lpSum(pick[i] for i in df.index if df.loc[i,"team_name"]==team) <= MAX_PER_TEAM

    prob += pulp.lpSum(start[i] for i in df.index) == STARTERS
    for i in df.index:
        prob += start[i] <= pick[i]
    prob += pulp.lpSum(start[i] for i in df.index if df.loc[i,"pos"]=="GKP") == 1

    if form_tuple and form_tuple in FORMATIONS:
        d,m,f = form_tuple
        prob += pulp.lpSum(start[i] for i in df.index if df.loc[i,"pos"]=="DEF") == d
        prob += pulp.lpSum(start[i] for i in df.index if df.loc[i,"pos"]=="MID") == m
        prob += pulp.lpSum(start[i] for i in df.index if df.loc[i,"pos"]=="FWD") == f
    else:
        prob += pulp.lpSum(start[i] for i in df.index if df.loc[i,"pos"]=="DEF") >= 3
        prob += pulp.lpSum(start[i] for i in df.index if df.loc[i,"pos"]=="MID") >= 2
        prob += pulp.lpSum(start[i] for i in df.index if df.loc[i,"pos"]=="FWD") >= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    df["selected"] = [int(pick[i].value()) for i in df.index]
    df["starter"]  = [int(start[i].value()) for i in df.index]

    squad = df[df["selected"]==1].copy()
    xi = squad[squad["starter"]==1].copy().sort_values("exp_pts", ascending=False)
    bench = squad[squad["starter"]==0].copy().sort_values("exp_pts", ascending=False)

    captain = xi.iloc[0]["name"] if len(xi)>0 else None
    vice    = xi.iloc[1]["name"] if len(xi)>1 else None
    objective = float(pulp.value(prob.objective)) if pulp.value(prob.objective) is not None else 0.0
    return xi, bench, captain, vice, squad, objective

# -------------------- Printing --------------------
def infer_formation(xi: pd.DataFrame) -> tuple[int,int,int]:
    d = int((xi["pos"] == "DEF").sum())
    m = int((xi["pos"] == "MID").sum())
    f = int((xi["pos"] == "FWD").sum())
    return (d, m, f)

def player_opp_str(team_short: str, teams_df: pd.DataFrame, opp_dict: dict):
    tid_series = teams_df.loc[teams_df["team_short"]==team_short, "team_id"]
    if tid_series.empty:
        return "—"
    tid = int(tid_series.values[0])
    opps = opp_dict.get(tid, [])
    return " / ".join(opps) if opps else "—"

def print_output_ascii_pitch(
    xi: pd.DataFrame,
    bench: pd.DataFrame,
    squad: pd.DataFrame,
    captain: str|None,
    vice: str|None,
    opp_dict: dict,
    teams: pd.DataFrame,
    formation_hint: str|None,
    budget_million: float,
    objective_xpts: float,
    use_color: bool = True,
):
    # Determine formation
    if formation_hint:
        try:
            d, m, f = tuple(int(x) for x in formation_hint.split("-"))
        except:
            d, m, f = infer_formation(xi)
    else:
        d, m, f = infer_formation(xi)

    # Sort within positions for consistent look
    xi_sorted = xi.sort_values(["pos","exp_pts"], ascending=[True, False]).copy()

    def is_c(name): return captain is not None and name == captain
    def is_vc(name): return vice is not None and name == vice

    # Build rows (top to bottom: FWD, MID, DEF, GKP)
    def build_row(group_df):
        cards=[]
        for _, r in group_df.iterrows():
            opps = player_opp_str(r["team_short"], teams, opp_dict)
            card = make_card(
                name=r["name"],
                team_short=r["team_short"],
                opp_str=opps,
                price_m=r["now_cost"]/10.0,
                xpts=r["exp_pts"],
                is_c=is_c(r["name"]),
                is_vc=is_vc(r["name"]),
                width=18
            )
            if is_c(r["name"]):
                card[0] = colorize(card[0], ANSI_GREEN, use_color)
            elif is_vc(r["name"]):
                card[0] = colorize(card[0], ANSI_CYAN, use_color)
            cards.append(card)
        return cards

    gk_list  = xi_sorted[xi_sorted["pos"]=="GKP"].head(1)
    def_list = xi_sorted[xi_sorted["pos"]=="DEF"].head(d)
    mid_list = xi_sorted[xi_sorted["pos"]=="MID"].head(m)
    fwd_list = xi_sorted[xi_sorted["pos"]=="FWD"].head(f)

    rows_cards = [
        build_row(fwd_list),
        build_row(mid_list),
        build_row(def_list),
        build_row(gk_list),
    ]

    # Compute finances
    total_value = float(((xi["now_cost"].sum() + bench["now_cost"].sum()) / 10.0))
    bank = float(budget_million - total_value)
    bank_str = f"Bank: £{bank:.1f}m"
    value_str = f"Total Squad Value: £{total_value:.1f}m"
    xpts_str = f"Objective xPts (XI + bench*0.1): {objective_xpts:.2f}"

    title = [
        colorize(f"Starting XI   Formation: {d}-{m}-{f}", ANSI_BOLD, use_color),
        colorize(f"Captain: {captain or '-'}   Vice: {vice or '-'}", ANSI_BOLD, use_color),
    ]
    footer = [
        colorize(value_str, ANSI_BOLD, use_color),
        colorize(bank_str, ANSI_BOLD, use_color),
        colorize(xpts_str, ANSI_DIM, use_color),
    ]
    print(draw_pitch(rows_cards, title_lines=title, footer_lines=footer))

    # Bench (GK + best 3 outfield)
    gk_bench = bench[bench["pos"]=="GKP"].head(1)
    of_bench = bench[bench["pos"]!="GKP"].head(3)
    bench_view = pd.concat([gk_bench, of_bench])

    bench_cards = []
    for _, r in bench_view.iterrows():
        opps = player_opp_str(r["team_short"], teams, opp_dict)
        card = make_card(
            name=r["name"],
            team_short=r["team_short"],
            opp_str=opps,
            price_m=r["now_cost"]/10.0,
            xpts=r["exp_pts"],
            width=18
        )
        bench_cards.append(card)

    print(draw_pitch([bench_cards], title_lines=[colorize("Bench (GK + 3)", ANSI_BOLD, use_color)]))

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, required=True, help="Gameweek number, e.g., 1")
    ap.add_argument("--budget", type=float, default=100.0, help="Budget in £m (default 100.0)")
    ap.add_argument("--formation", type=str, default=None, help='Formation, e.g. "3-4-3" (optional)')
    ap.add_argument("--bench-weight", type=float, default=0.1, help="Relative value of bench in objective (0..1)")
    ap.add_argument("--no-color", action="store_true", help="Disable ANSI colors in output")
    args = ap.parse_args()

    bootstrap = fetch_bootstrap()
    teams = pd.DataFrame(bootstrap["teams"])[["id","name","short_name"]].rename(
        columns={"id":"team_id","name":"team_name","short_name":"team_short"}
    )
    fixtures = fetch_fixtures(args.gw)
    players = build_players_df(bootstrap)
    players = compute_expected_points(players)
    opp_dict = opponent_map(fixtures, teams)

    xi, bench, c, v, squad, obj = optimize(players, args.budget, args.formation, args.bench_weight)

    print_output_ascii_pitch(
        xi=xi,
        bench=bench,
        squad=squad,
        captain=c,
        vice=v,
        opp_dict=opp_dict,
        teams=teams,
        formation_hint=args.formation,
        budget_million=args.budget,
        objective_xpts=obj,
        use_color=not args.no_color
    )

if __name__ == "__main__":
    main()
