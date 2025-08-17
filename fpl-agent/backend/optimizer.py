import pandas as pd, pulp

POSITION_COUNTS = {"GKP":2,"DEF":5,"MID":5,"FWD":3}
MAX_PER_TEAM = 3
STARTERS = 11
FORMATIONS = {(3,4,3),(3,5,2),(4,4,2),(4,3,3),(4,5,1),(5,4,1),(5,3,2),(5,2,3)}

def optimize(players: pd.DataFrame, budget_million: float, formation_hint: str|None, bench_weight: float=0.1):
    df = players.copy().reset_index(drop=True)

    form_tuple = None
    if formation_hint:
        try: form_tuple = tuple(int(x) for x in formation_hint.split("-"))
        except: form_tuple = None

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
    for i in df.index: prob += start[i] <= pick[i]
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
    objective = float(pulp.value(prob.objective) or 0.0)
    return xi, bench, captain, vice, squad, objective
