"""Table 1: hold-out regimes on the Bandhavgarh Terra-day channel.

usage: python scripts/run_table1.py --data DATA_DIR --out results [--regime R] [--seed S]
Without --regime/--seed runs all 3 regimes x seeds 7..11 (slow, ~30 min).
"""
import argparse, os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lstdark import io, holdout, estimators as E, metrics as Mt

def one(T, A, regime, sd, ranks):
    H = {"random": lambda: holdout.random_entries(T, sd),
         "cloud-shaped": lambda: holdout.cloud_shaped(T, sd),
         "blackout": lambda: holdout.blackout(T, A, sd)}[regime]()
    Tm = T.copy(); Tm[H] = np.nan; I = E.time_interp(Tm)
    row = dict(regime=regime, seed=sd, held_frac=float(H.sum() / np.isfinite(T).sum()),
               spatial_mean=Mt.rmse(E.spatial_mean(Tm), T, H), time_interp=Mt.rmse(I, T, H),
               climatology=Mt.rmse(E.climatology(Tm), T, H))
    for r in range(1, 9):
        row[f"svd_r{r}"] = Mt.rmse(E.low_rank(Tm, r, I), T, H) if r in ranks else np.nan
    return row

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="results")
    ap.add_argument("--regime"); ap.add_argument("--seed", type=int); ap.add_argument("--ranks", default="1,3")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    M, _ = io.load_tile(a.data, "bandhavgarh"); T, A = M["T_day"], M["A_day"]
    ranks = [int(x) for x in a.ranks.split(",")]
    jobs = [(a.regime, a.seed)] if a.regime else [(r, s) for r in ["random", "cloud-shaped", "blackout"] for s in range(7, 12)]
    f = os.path.join(a.out, "table1_bandhavgarh.csv")
    for reg, sd in jobs:
        row = pd.DataFrame([one(T, A, reg, sd, ranks)])
        row.to_csv(f, mode="a", header=not os.path.exists(f), index=False); print(row.round(3).to_string(index=False), flush=True)
