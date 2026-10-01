"""Hold-out regimes. All withhold ~15% of the observed Terra-day entries."""
import numpy as np


def blackout(T, A, seed=7, frac=0.15, min_obs=0.5):
    """Remove whole composites. Candidates: composites with >min_obs of pixels
    observed in both T and A. Shuffle candidates and add whole composites until
    >= frac of observed T entries are withheld. Returns boolean held mask."""
    oT, oA = np.isfinite(T), np.isfinite(A)
    cand = np.where((oT.mean(0) > min_obs) & (oA.mean(0) > min_obs))[0]
    cnt, tot, held, cols = oT.sum(0), oT.sum(), 0, []
    for j in np.random.default_rng(seed).permutation(cand):
        cols.append(j); held += cnt[j]
        if held >= frac * tot:
            break
    H = np.zeros_like(oT); H[:, cols] = oT[:, cols]
    return H


def random_entries(T, seed=7, frac=0.15):
    """Remove frac of observed entries uniformly at random."""
    obs = np.flatnonzero(np.isfinite(T))
    pick = np.random.default_rng(seed).choice(obs, int(round(frac * obs.size)), replace=False)
    H = np.zeros(T.shape, bool); H.flat[pick] = True
    return H


def cloud_shaped(T, seed=7, frac=0.15, min_obs=0.95, donor_range=(0.2, 0.8)):
    """Stamp the gap pattern of a partially cloudy donor composite onto a
    well-observed target composite, repeating until >= frac is withheld."""
    oT = np.isfinite(T); f = oT.mean(0)
    targets = np.where(f > min_obs)[0]
    donors = np.where((f > donor_range[0]) & (f < donor_range[1]))[0]
    rng = np.random.default_rng(seed); H = np.zeros_like(oT)
    tot, held = oT.sum(), 0
    for j in rng.permutation(targets):
        d = rng.choice(donors)
        h = oT[:, j] & ~oT[:, d]
        H[:, j] = h; held += h.sum()
        if held >= frac * tot:
            break
    return H
