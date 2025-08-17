import requests, pandas as pd

BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES = "https://fantasy.premierleague.com/api/fixtures/"

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
    keep = ["id","web_name","now_cost","pos","team_id","team_name","team_short",
            "points_per_game","form","chance_of_playing_next_round","ep_next"]
    for k in keep:
        if k not in df: df[k] = None
    df = df[keep].rename(columns={"web_name":"name"})
    for c in ["points_per_game","form","ep_next","chance_of_playing_next_round"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def opponent_map(fixtures: pd.DataFrame, teams: pd.DataFrame):
    opp = {}
    short_by_id = {int(r.team_id): r.team_short for _, r in teams.iterrows()}
    for _, row in fixtures.iterrows():
        try:
            home = int(row["team_h"]); away = int(row["team_a"])
        except Exception:
            continue
        home_team = short_by_id.get(home, "?")
        away_team = short_by_id.get(away, "?")
        opp.setdefault(home, []).append(f"{away_team} (H)")
        opp.setdefault(away, []).append(f"{home_team} (A)")
    return opp
