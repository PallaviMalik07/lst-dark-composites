"""Make Figs 1-4 of the paper from the results/ folder.
usage: python scripts/make_figures.py --results results --out figures"""
import argparse, json, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

ORDER = ["thar_arid", "deccan_semiarid", "bandhavgarh", "ne_humid", "konkan_coastal"]
LAB = ["Thar\n(arid)", "Deccan\n(semiarid)", "Bandhavgarh\n(subhumid)", "NE India\n(humid)", "Konkan\n(coastal)"]
NAME = ["Thar (arid)", "Deccan (semiarid)", "Bandhavgarh (subhumid)", "NE India (humid)", "Konkan (coastal)"]
plt.rcParams.update({"font.family": "serif", "font.size": 9})


def clean(ax, labels=None):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if labels is not None:
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7.5)


def main(res, out):
    os.makedirs(out, exist_ok=True)
    R = {t: json.load(open(os.path.join(res, f"{t}.json"))) for t in ORDER}
    x = np.arange(5)
    # Fig 1
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), dpi=600)
    for t, n in zip(ORDER, NAME):
        ax[0].plot(range(1, 13), [R[t][f"terra_obs_month_{m:02d}"] for m in range(1, 13)], "o-", ms=3, lw=1, label=n)
    ax[0].set_ylim(0, 1.05); ax[0].set_xticks(range(1, 13)); ax[0].set_xlabel("month")
    ax[0].set_ylabel("observed fraction (Terra day)"); ax[0].legend(frameon=False, fontsize=7)
    ax[0].set_title("Daytime availability collapses in monsoon", fontsize=9); clean(ax[0])
    w = 0.38
    ax[1].bar(x - w/2, [R[t]["cooc_TA_day"] for t in ORDER], w, color="#595959", label="Terra day vs Aqua day")
    ax[1].bar(x + w/2, [R[t]["cooc_Tday_Tnight"] for t in ORDER], w, color="#bfbfbf", edgecolor="k", lw=0.6, label="Terra day vs Terra night")
    ax[1].axhline(1, ls="--", c="k", lw=0.8); ax[1].text(3.6, 1.1, "independent", fontsize=7)
    ax[1].set_ylabel("gap co-occurrence ratio"); ax[1].legend(frameon=False, fontsize=7)
    ax[1].set_title("Same-overpass gaps coincide; day/night do not", fontsize=9); clean(ax[1], LAB)
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig1_gap_structure.png"), dpi=600); plt.close()
    # Fig 2
    d = pd.read_csv(os.path.join(res, "table1_bandhavgarh.csv"))
    cols = ["spatial_mean", "time_interp", "climatology", "svd_r1", "svd_r3"]
    m = d.groupby("regime")[cols].mean(); s = d.groupby("regime")[cols].std()
    fig, ax = plt.subplots(figsize=(5.6, 3.0), dpi=600); w = 0.26
    for k, (rg, c) in enumerate(zip(["random", "cloud-shaped", "blackout"], ["#404040", "#8c8c8c", "#d9d9d9"])):
        ax.bar(np.arange(5) + (k-1)*w, m.loc[rg], w, yerr=s.loc[rg], color=c, edgecolor="k", lw=0.6,
               error_kw=dict(lw=0.6, capsize=1.5), label=rg)
    ax.set_xticks(range(5)); ax.set_xticklabels(["spatial mean", "time-interp", "climatology", "SVD r=1", "SVD r=3"], rotation=15)
    ax.set_ylabel("RMSE (K)"); ax.legend(frameon=False, title="hold-out regime", fontsize=8, title_fontsize=8)
    ax.set_title("Low-rank advantage survives cloud shapes, vanishes under blackout", fontsize=9); clean(ax)
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig2_regimes.png"), dpi=600); plt.close()
    # Fig 3
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=600); w = 0.18
    keys = [("interp", "linear interpolation", "#262626"), ("lowrank_r3", "low-rank r=3", "#595959"),
            ("joint_stack", "joint Terra+Aqua stack", "#8c8c8c"), ("coupling", "cross-sensor coupling", "#cccccc")]
    for k, (key, lab, c) in enumerate(keys):
        ax.bar(x + (k-1.5)*w, [R[t][key] for t in ORDER], w, color=c, label=lab)
    ax.scatter(x, [R[t]["harmonic"] for t in ORDER], marker="^", c="k", s=18, zorder=3, label="harmonic time-mode")
    ax.set_ylabel("RMSE on blacked-out composites (K)", fontsize=8); ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_title("No method family recovers dark composites", fontsize=9); clean(ax, LAB)
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig3_blackout_remedies.png"), dpi=600); plt.close()
    # Fig 4
    T = [R[t]["terra_dark"] for t in ORDER]; B = [R[t]["both_dark"] for t in ORDER]; N = [R[t]["both_dark_with_night"] for t in ORDER]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), dpi=600); w = 0.55
    ax[0].bar(x, T, w, color="#bfbfbf", edgecolor="k", lw=0.6, label="Terra dark")
    ax[0].bar(x, B, w, color="#404040", edgecolor="k", lw=0.6, label="both sensors dark")
    ax[0].set_title("Blackouts are largely simultaneous", fontsize=9)
    ax[1].bar(x, B, w, color="#bfbfbf", edgecolor="k", lw=0.6, label="both sensors dark")
    ax[1].bar(x, N, w, color="#737373", edgecolor="k", lw=0.6, label="with any night-time retrieval")
    ax[1].set_title("Night-time retrievals at dark composites", fontsize=9)
    for a, top in zip(ax, [max(T)*1.25, max(B)*1.4]):
        a.set_ylabel("composites of 506"); a.set_ylim(0, top); a.legend(frameon=False, fontsize=8, loc="upper left"); clean(a, LAB)
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig4_dark_composites.png"), dpi=600); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); ap.add_argument("--out", default="figures")
    a = ap.parse_args(); main(a.results, a.out)
