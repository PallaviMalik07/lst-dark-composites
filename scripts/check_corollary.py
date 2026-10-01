"""Corollary 1.1 check: RMS difference between low-rank and interpolation
predictions at blacked-out entries, vs RMS difference at partially observed gaps."""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lstdark import io, holdout, estimators as E
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--tile", required=True); ap.add_argument("--out", default="results")
a = ap.parse_args()
M, _ = io.load_tile(a.data, a.tile); T, A = M["T_day"], M["A_day"]
H = holdout.blackout(T, A); Tm = T.copy(); Tm[H] = np.nan
I = E.time_interp(Tm); L = E.low_rank(Tm, 3, I)
gap = ~np.isfinite(Tm) & ~H & np.isfinite(I)
r = {"tile": a.tile,
     "rms_lowrank_minus_interp_dark": float(np.sqrt(np.mean((L[H] - I[H]) ** 2))),
     "rms_lowrank_minus_interp_partial_gaps": float(np.sqrt(np.mean((L[gap] - I[gap]) ** 2))),
     "corr_lowrank_interp_dark": float(np.corrcoef(L[H], I[H])[0, 1])}
json.dump(r, open(os.path.join(a.out, f"corollary_{a.tile}.json"), "w")); print(r)
