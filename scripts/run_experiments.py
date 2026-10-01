"""Run all experiments of the paper on the exported GeoTIFFs.

usage: python scripts/run_experiments.py --data DATA_DIR --out results
DATA_DIR holds <prefix>_<year>.tif for the five tiles (see README).
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lstdark import io, holdout, estimators as E, metrics as Mt

TILES = {"thar_arid": "Thar (arid)", "deccan_semiarid": "Deccan (semiarid)",
         "bandhavgarh": "Bandhavgarh (subhumid)", "ne_humid": "NE India (humid)",
         "konkan_coastal": "Konkan (coastal)"}


def masked(T, H):
    Tm = T.copy(); Tm[H] = np.nan; return Tm


def gap_structure(M, dates):
    mon = np.array([int(d[5:7]) for d in dates]); o = np.isfinite(M["T_day"])
    out = {f"terra_obs_month_{m:02d}": float(o[:, mon == m].mean()) for m in range(1, 13)}
    out["cooc_TA_day"] = Mt.cooccurrence(M["T_day"], M["A_day"])
    out["cooc_Tday_Tnight"] = Mt.cooccurrence(M["T_day"], M["T_night"])
    out["night_obs_frac"] = float(np.isfinite(M["T_night"]).mean())
    out.update(Mt.dark_counts(M)); return out


def blackout_suite(M):
    T, A = M["T_day"], M["A_day"]
    H = holdout.blackout(T, A); Tm = masked(T, H)
    r = {"blacked": int(H.any(0).sum()), "held": int(H.sum())}
    r["interp"] = Mt.rmse(E.time_interp(Tm), T, H)
    r["lowrank_r3"] = Mt.rmse(E.low_rank(Tm, 3), T, H)
    r["joint_stack"] = Mt.rmse(E.joint_stack(Tm, A), T, H)
    C, r2 = E.coupling(Tm, A)
    r["coupling"] = Mt.rmse(C, T, H)
    r["coupling_R2_median"] = float(np.nanmedian(r2))
    r["harmonic"] = Mt.rmse(E.low_rank(Tm, 3, basis=E.harmonic_basis(T.shape[1])), T, H)
    r["factor_interp"] = Mt.rmse(E.factor_interp(Tm, 3), T, H)
    _, _, r2n = E.pixel_regression(T, M["T_night"])
    r["night_day_R2_median"] = float(np.nanmedian(r2n))
    r["night_day_pairs_median"] = float(np.median((np.isfinite(T) & np.isfinite(M["T_night"])).sum(1)))
    return r


def spectrum(T, seed=7):
    """Energy share of the leading components of the rank-8 completion (random hold-out)."""
    Tm = masked(T, holdout.random_entries(T, seed)); X = E.low_rank(Tm, 8)
    s = np.linalg.svd(X - X.mean(0), compute_uv=False); e = s**2 / (s**2).sum()
    return {"energy_first_centred": float(e[0]), "energy_first3_centred": float(e[:3].sum())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results"); ap.add_argument("--tiles", nargs="*", default=list(TILES))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    for t in a.tiles:
        t0 = time.time(); M, dates = io.load_tile(a.data, t)
        res = {"tile": t, "n_pixels": M["T_day"].shape[0], "n_composites": len(dates)}
        res.update(gap_structure(M, dates)); res.update(blackout_suite(M))
        json.dump(res, open(os.path.join(a.out, f"{t}.json"), "w"), indent=1)
        print(t, f"{time.time()-t0:.0f}s", flush=True)
        if t == "bandhavgarh":
            json.dump(spectrum(M["T_day"]), open(os.path.join(a.out, "spectrum_bandhavgarh.json"), "w"))
