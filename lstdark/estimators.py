"""Reconstruction estimators. Input Xm: matrix with NaN at missing entries."""
import numpy as np


def time_interp(Xm):
    """Per-pixel linear interpolation on the composite index (constant beyond ends)."""
    x = np.arange(Xm.shape[1], dtype=float); out = Xm.copy()
    for i in range(Xm.shape[0]):
        m = np.isfinite(Xm[i])
        if m.any():
            out[i] = np.interp(x, x[m], Xm[i, m])
    return out


def spatial_mean(Xm):
    """Mean of the observed pixels of each composite; tile mean if none observed."""
    obs = np.isfinite(Xm); s = np.where(obs, Xm, 0).sum(0); n = obs.sum(0)
    col = np.where(n > 0, s / np.maximum(n, 1), np.nanmean(Xm))
    return np.where(obs, Xm, col[None, :]).astype(np.float32)


def climatology(Xm, period=46):
    """Per-pixel mean of the same 8-day slot across years; pixel mean as fallback."""
    out = Xm.copy(); slot = np.arange(Xm.shape[1]) % period
    pm = np.nanmean(Xm, 1)
    for s in range(period):
        c = slot == s
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            v = np.nanmean(Xm[:, c], 1)
        v = np.where(np.isfinite(v), v, pm)
        blk = out[:, c]; out[:, c] = np.where(np.isfinite(blk), blk, v[:, None])
    return out


def low_rank(Xm, r=3, init=None, iters=20, basis=None):
    """Iterative hard-impute: column-centred rank-r SVD, observed entries held
    fixed, missing entries seeded with `init` (default: time_interp), float32.
    If `basis` (p x k) is given the temporal factor is projected onto its span
    (harmonic constraint)."""
    obs = np.isfinite(Xm)
    if init is None:
        init = time_interp(Xm)
    X = np.where(obs, Xm, init).astype(np.float32)
    P = None if basis is None else basis @ np.linalg.pinv(basis)
    for _ in range(iters):
        mu = X.mean(0, keepdims=True)
        U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
        Vr = Vt[:r]
        if P is not None:
            Vr = (P @ Vr.T).T
            mu = (P @ mu.T).T
        L = (U[:, :r] * s[:r]) @ Vr + mu
        X = np.where(obs, Xm, L).astype(np.float32)
    return X


def harmonic_basis(p, period=46, k=2):
    t = np.arange(p) * 2 * np.pi / period
    cols = [np.ones(p)] + [f(h * t) for h in range(1, k + 1) for f in (np.sin, np.cos)]
    return np.stack(cols, 1)


def joint_stack(Tm, A, r=3):
    """Shared-spatial-factor stack [T | A] (Proposition 2); returns T part."""
    p = Tm.shape[1]
    return low_rank(np.hstack([Tm, A]), r)[:, :p]


def factor_interp(Xm, r=3, dark=None):
    """Low-rank completion, then replace the temporal coefficients of dark
    composites by linear interpolation of the neighbouring coefficients."""
    X = low_rank(Xm, r)
    mu = X.mean(0, keepdims=True); U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    V = Vt[:r].copy(); m = mu[0].copy()
    if dark is None:
        dark = ~np.isfinite(Xm).any(0)
    x = np.arange(Xm.shape[1]); ok = ~dark
    for k in range(r):
        V[k, dark] = np.interp(x[dark], x[ok], V[k, ok])
    m[dark] = np.interp(x[dark], x[ok], m[ok])
    L = (U[:, :r] * s[:r]) @ V + m
    return np.where(np.isfinite(Xm), Xm, L)


def pixel_regression(Tm, A, min_pairs=3):
    """Per-pixel OLS T = a*A + b on co-observed composites (Proposition 3).
    Returns slope, intercept, R^2 per pixel."""
    both = np.isfinite(Tm) & np.isfinite(A); n = Tm.shape[0]
    a, b, r2 = (np.full(n, np.nan) for _ in range(3))
    for i in range(n):
        m = both[i]
        if m.sum() >= min_pairs and np.std(A[i, m]) > 0:
            a[i], b[i] = np.polyfit(A[i, m].astype(float), Tm[i, m].astype(float), 1)
            pr = a[i] * A[i, m] + b[i]; y = Tm[i, m]
            ss = ((y - y.mean()) ** 2).sum()
            r2[i] = 1 - ((y - pr) ** 2).sum() / ss if ss > 0 else np.nan
    return a, b, r2


def coupling(Tm, A, r=3):
    """Cross-sensor coupling with low-rank projection ('reg+polish').
    Regression predictions at fully missing composites are projected onto the
    spatial basis [U_r, 1] of the single-sensor low-rank completion."""
    a, b, r2 = pixel_regression(Tm, A)
    R = a[:, None] * A + b[:, None]
    P = low_rank(Tm, r)
    mu = P.mean(0, keepdims=True); U, s, Vt = np.linalg.svd(P - mu, full_matrices=False)
    B = np.hstack([U[:, :r], np.ones((U.shape[0], 1))]); out = P.copy()
    for j in np.where(~np.isfinite(Tm).any(0))[0]:
        m = np.isfinite(R[:, j])
        if m.sum() > r + 1:
            c, *_ = np.linalg.lstsq(B[m], R[m, j], rcond=None)
            out[:, j] = B @ c
    return out, r2
