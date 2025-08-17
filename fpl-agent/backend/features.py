import pandas as pd

# FPL fixture difficulty values (1 easiest .. 5 hardest) → multiplicative factor
FDR_FACTOR = {1: 1.10, 2: 1.05, 3: 1.00, 4: 0.92, 5: 0.85}

def _team_fdr_factors(fixtures: pd.DataFrame) -> dict[int, float]:
    """Return team_id -> average FDR factor for the current GW."""
    fdr = {}
    if fixtures is None or fixtures.empty:
        return {}
    for _, row in fixtures.iterrows():
        try:
            th = int(row["team_h"]); ta = int(row["team_a"])
            dh = int(row.get("team_h_difficulty", 3)); da = int(row.get("team_a_difficulty", 3))
        except Exception:
            continue
        fdr.setdefault(th, []).append(FDR_FACTOR.get(dh, 1.0))
        fdr.setdefault(ta, []).append(FDR_FACTOR.get(da, 1.0))
    return {tid: float(pd.Series(v).mean()) for tid, v in fdr.items()}

def compute_expected_points(
    players: pd.DataFrame,
    teams: pd.DataFrame,
    fixtures: pd.DataFrame,
    form_weight: float = 0.6,
    fdr_weight: float = 0.3,
) -> pd.DataFrame:
    """
    exp_pts = play_prob * (form_weight*form + (1-form_weight)*ppg) * mix(fdr)
    where mix(fdr) = (1 - fdr_weight) + fdr_weight * FDR_FACTOR
    """
    df = players.copy()
    form_weight = float(min(max(form_weight, 0.0), 1.0))
    fdr_weight = float(min(max(fdr_weight, 0.0), 1.0))

    # Base from recent form & season PPG
    ppg = pd.to_numeric(df["points_per_game"], errors="coerce").fillna(0)
    form = pd.to_numeric(df["form"], errors="coerce").fillna(0)
    base = form_weight * form + (1.0 - form_weight) * ppg

    # Probability of playing
    play_prob = pd.to_numeric(df["chance_of_playing_next_round"], errors="coerce").fillna(100.0) / 100.0
    combined = base.clip(lower=0) * play_prob

    # Fixture difficulty factor (by team for the chosen GW)
    fdr_map = _team_fdr_factors(fixtures)
    fdr_factor = df["team_id"].map(lambda tid: fdr_map.get(int(tid), 1.0) if pd.notna(tid) else 1.0)
    mixed = (1.0 - fdr_weight) + fdr_weight * fdr_factor

    df["exp_pts"] = (combined * mixed).clip(lower=0)

    # tiny positional bias to stabilize ranks
    bias = df["pos"].map({"GKP":0.6,"DEF":0.4,"MID":0.2,"FWD":0.3}).fillna(0)
    df["exp_pts"] = (df["exp_pts"] + bias).clip(lower=0)
    return df
