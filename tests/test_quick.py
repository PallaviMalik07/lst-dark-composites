"""Quick tests on small synthetic data (no download needed, ~10 s).
Run:  python -m pytest tests -q"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lstdark import holdout, estimators as E, metrics as M


def synthetic(n=300, p=138, r=3, noise=0.3, seed=0):
    """Low-rank 'LST' matrix: pixel patterns x seasonal temporal modes + noise."""
    g = np.random.default_rng(seed); t = np.arange(p) * 2 * np.pi / 46
    V = np.stack([np.sin(t), np.cos(t), g.normal(0, 1, p)], 1)[:, :r]
    U = g.normal(0, 3, (n, r))
    return (30 + U @ V.T + g.normal(0, noise, (n, p))).astype(np.float32)


def test_interp_exact_on_linear_rows():
    X = np.tile(np.arange(20, dtype=np.float32), (3, 1)); Xm = X.copy(); Xm[:, 5:9] = np.nan
    assert np.allclose(E.time_interp(Xm), X)


def test_blackout_removes_whole_composites_and_15pct():
    T = synthetic(); A = T + 1.0
    H = holdout.blackout(T, A, seed=7)
    cols = H.any(0)
    assert H[:, cols].all()                            # every chosen composite fully removed
    assert 0.15 <= H.sum() / np.isfinite(T).sum() < 0.17


def test_lowrank_beats_interp_under_random_holdout():
    T = synthetic(); H = holdout.random_entries(T, seed=1); Tm = T.copy(); Tm[H] = np.nan
    assert M.rmse(E.low_rank(Tm, 3), T, H) < 0.5 * M.rmse(E.time_interp(Tm), T, H)


def test_proposition1_dark_column_set_by_initialisation():
    """Two fills that differ only on a dark column give different outputs there
    while fitting the observed entries equally well (Proposition 1, Corollary 1.1)."""
    T = synthetic(); Tm = T.copy(); j = 70; Tm[:, j] = np.nan
    I = E.time_interp(Tm); I2 = I.copy(); I2[:, j] += 5.0
    X1, X2 = E.low_rank(Tm, 3, I), E.low_rank(Tm, 3, I2)
    assert np.abs(X1[:, j] - X2[:, j]).mean() > 1.0            # answer moves with the seed
    obs = np.isfinite(Tm)
    assert np.allclose(X1[obs], X2[obs])                         # observations identical
    # Corollary 1.1: dark column = projection of fill onto [spatial basis, 1]
    mu = X1.mean(0, keepdims=True); U, s, Vt = np.linalg.svd(X1 - mu, full_matrices=False)
    B = np.hstack([U[:, :3], np.ones((U.shape[0], 1))])
    c, *_ = np.linalg.lstsq(B, X1[:, j], rcond=None)
    assert np.sqrt(np.mean((B @ c - X1[:, j]) ** 2)) < 1e-3


def test_cooccurrence_independent_is_one():
    g = np.random.default_rng(3)
    a = np.where(g.random((200, 200)) < 0.3, np.nan, 1.0); b = np.where(g.random((200, 200)) < 0.3, np.nan, 1.0)
    assert abs(M.cooccurrence(a, b) - 1) < 0.05
