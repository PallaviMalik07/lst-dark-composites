"""Quick example: the paper's central result on synthetic data in a few seconds.

    python examples/quick_example.py

Low-rank completion wins clearly when entries are missing at random, and
loses its advantage when whole composites are missing (dark composites)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lstdark import holdout, estimators as E, metrics as M
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from test_quick import synthetic

T = synthetic(n=400, p=230, noise=1.0, seed=0)
A = T + np.random.default_rng(1).normal(0, 1.0, T.shape).astype(np.float32)  # partner sensor
print(f"synthetic tile: {T.shape[0]} pixels x {T.shape[1]} composites")
print(f"{'hold-out':10s} {'interp':>8s} {'low-rank r=3':>13s}")
for name, H in [("random", holdout.random_entries(T, seed=7)), ("blackout", holdout.blackout(T, A, seed=7))]:
    Tm = T.copy(); Tm[H] = np.nan
    print(f"{name:10s} {M.rmse(E.time_interp(Tm), T, H):8.3f} {M.rmse(E.low_rank(Tm, 3), T, H):13.3f}")
