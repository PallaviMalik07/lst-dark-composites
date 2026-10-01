# lst-dark-composites

Code for the paper **"Dark composites are unrecoverable: an identifiability limit on
gap-filling of MODIS land surface temperature"**.

Authors: Pallavi Malik, Shivam Chauhan
Contact: imshivamhere@gmail.com (corresponding author)
License: MIT (see `LICENSE`)

## What the code does

When cloud removes every retrieval from an 8-day MODIS composite ("dark composite"),
no gap-filling method that uses only the MODIS record can recover it. This repository
contains everything used to show that on 11 years (2015–2025) of MOD11A2/MYD11A2
daytime LST at five Indian tiles:

- gap statistics: monthly availability, Terra/Aqua and day/night gap co-occurrence,
  dark-composite counts (paper §5.1, Figs 1 and 4);
- reconstruction under random, cloud-shaped and whole-composite (blackout) hold-out
  (§5.2, Table 1, Fig. 2);
- the four remedies: Terra+Aqua stacking, cross-sensor coupling, harmonic
  constraint, temporal-factor interpolation (§5.3, Table 2, Fig. 3);
- a direct check of Corollary 1.1 (§5.3).

## Repository layout

```
lstdark/               Python package
  io.py                read the exported GeoTIFFs
  holdout.py           random, cloud-shaped and blackout hold-out
  estimators.py        interpolation, climatology, spatial mean, low-rank, remedies
  metrics.py           RMSE, gap co-occurrence, dark-composite counts
scripts/
  gee_export.js        Google Earth Engine script that downloads the input data
  run_experiments.py   all per-tile results (Table 2, §5.1)
  run_table1.py        hold-out regimes, Bandhavgarh, 5 seeds (Table 1)
  check_corollary.py   Corollary 1.1 check
  make_figures.py      Figs 1-4
tests/test_quick.py    quick tests on synthetic data (no download needed)
examples/quick_example.py   10-second demonstration on synthetic data
results/               outputs used in the paper (for comparison)
figures/               Figs 1-4 as used in the paper
```

## Installation

Python 3.9 or newer.

```
git clone https://github.com/PallaviMalik07/lst-dark-composites.git
cd lst-dark-composites
pip install -r requirements.txt
```

## Quick test (no data download)

```
python -m pytest tests -q
```

Expected: `5 passed` in a few seconds. The tests check the hold-out schemes, the
low-rank estimator and Proposition 1 / Corollary 1.1 on a synthetic matrix.

Quick example:

```
python examples/quick_example.py
```

Expected output: low-rank completion is far better than interpolation under random
hold-out (about 1.0 vs 3.7) and gives almost no gain under blackout (about 3.6 vs 3.7).

## Reproducing the paper

1. **Get the data.** Open https://code.earthengine.google.com, paste
   `scripts/gee_export.js`, run it, and start the 55 export tasks (5 tiles × 11 years).
   Download the GeoTIFFs from the Google Drive folder `lst_dark_composites` into
   a local folder, e.g. `data/`. Files are named `<tile>_<year>.tif`.
2. **Run the experiments** (about 1–2 min per tile):
   ```
   python scripts/run_experiments.py --data data --out results
   ```
3. **Table 1** (3 regimes × 5 seeds, about 30 min; can be run one job at a time
   with `--regime random --seed 7`):
   ```
   python scripts/run_table1.py --data data --out results
   ```
4. **Corollary 1.1 check** (one tile per call):
   ```
   python scripts/check_corollary.py --data data --tile bandhavgarh
   ```
5. **Figures:**
   ```
   python scripts/make_figures.py --results results --out figures
   ```

Compare your `results/` with the files shipped in this repository. Low-rank
results use float32 SVD, so values may differ in the last decimal across
machines or BLAS libraries.

## Key settings

- LST in °C from QC "LST produced" (QC bits 0–1 = 00 or 01); error class exported.
- All hold-out regimes withhold 15 % of observed Terra-day entries; default seed 7.
- Blackout candidates: composites with more than 50 % of pixels observed in both
  Terra and Aqua.
- Low-rank: column-centred iterative hard-impute, rank 3, 20 iterations,
  float32, initialised with per-pixel linear interpolation.
- "Dark" composite: no pixel of the tile observed.
