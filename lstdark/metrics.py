import numpy as np


def rmse(P, T, H):
    return float(np.sqrt(np.mean((P[H] - T[H]) ** 2)))


def cooccurrence(a, b):
    """P(both missing) / (P(a missing) P(b missing)); 1 = independence."""
    ma, mb = ~np.isfinite(a), ~np.isfinite(b)
    return float((ma & mb).mean() / (ma.mean() * mb.mean()))


def dark_counts(M):
    """Strict dark = no pixel observed. Returns dict of counts."""
    o = {k: np.isfinite(M[k]) for k in ["T_day", "A_day", "T_night", "A_night"]}
    td = ~o["T_day"].any(0); both = td & ~o["A_day"].any(0)
    night = o["T_night"].any(0) | o["A_night"].any(0)
    near = (o["T_day"].mean(0) <= 0.05) & ~td
    return dict(terra_dark=int(td.sum()), both_dark=int(both.sum()),
                both_dark_with_night=int((both & night).sum()),
                terra_near_dark=int(near.sum()))
