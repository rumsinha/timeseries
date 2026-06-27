"""
Weekend 9 (analysis): the calibration plots.

Reads the artifacts written by weekend_9.py and produces the
charts that decide whether the intervals are honest:

    01_reliability_diagram.png   target vs empirical coverage (the climax plot)
    02_coverage_by_horizon.png   does coverage hold at every horizon?
    03_interval_width.png        the price of the guarantee (width vs horizon)
    04_fan_chart.png             a week of test forecasts with the 80% band

Run after the pipeline:
    python weekend_9_analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "probabilistic_weekend_9"
RAW_DIR = DATA_DIR / "raw"
OUT = SCRIPT_DIR / "images" / "weekend_9"

COLORS = {"xgboost": "#FF6B35", "nbeats": "#4C78A8"}
METHOD_STYLE = {"conformal": "-", "quantile_regression": "--",
                "quantile": "--"}

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})
sns.set_palette("deep")


def display_label(model, method, meta=None):
    """Human label that makes sampled vs full-resolution evaluations explicit."""
    method_txt = method.replace("_", " ")
    if meta is None or meta.empty:
        return f"{model} · {method_txt}"
    row = meta[(meta["model"] == model) & (meta["method"] == method)]
    if row.empty or "n_origins" not in row or "origin_stride" not in row:
        return f"{model} · {method_txt}"
    n = int(row["n_origins"].iloc[0])
    stride = int(row["origin_stride"].iloc[0])
    return f"{model} · {method_txt} (n={n:,}, stride={stride})"


def plot_reliability(save):
    """THE plot. Diagonal = perfect calibration. Conformal should hug it;
    quantile regression typically sits off it (often under-covering). Drawing
    BOTH methods as curves is the whole argument of the post in one figure."""
    path = DATA_DIR / "reliability_curve.csv"
    if not path.exists():
        print(f"  SKIP reliability: {path} missing")
        return
    rc = pd.read_csv(path)
    meta_path = DATA_DIR / "coverage_summary.csv"
    meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot([0, 1], [0, 1], color="#444", linestyle=":", linewidth=1.5,
            label="perfect calibration", zorder=1)
    for model in sorted(rc["model"].unique()):
        for method in ("conformal", "quantile_regression"):
            sub = rc[(rc["model"] == model) & (rc["method"] == method)]
            if sub.empty:
                continue
            sub = sub.sort_values("target_coverage")
            ax.plot(sub["target_coverage"], sub["empirical_coverage"],
                    marker="o" if method == "conformal" else "s",
                    markersize=6,
                    color=COLORS.get(model, None),
                    linestyle="-" if method == "conformal" else "--",
                    linewidth=2,
                    label=display_label(model, method, meta))
    ax.set_xlabel("Target coverage (what the interval claims)")
    ax.set_ylabel("Empirical coverage (what actually happened on 2016)")
    ax.set_title("Reliability diagram — conformal (solid) shifts coverage upward;\n"
                 "native quantile regression (dashed) under-covers here")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(save, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save}")


def plot_coverage_by_horizon(save):
    """For the headline 80% interval, does coverage hold at EVERY horizon, for
    both methods? Flat near 0.80 = calibrated everywhere."""
    path = DATA_DIR / "coverage_summary.csv"
    if not path.exists():
        print(f"  SKIP coverage-by-horizon: {path} missing")
        return
    cs = pd.read_csv(path)
    cs = cs[np.isclose(cs["target_coverage"], 0.80)]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axhline(0.80, color="#444", linestyle=":", linewidth=1.5,
               label="target 0.80")
    for (model, method), sub in cs.groupby(["model", "method"]):
        sub = sub.sort_values("horizon")
        ax.plot(sub["horizon"], sub["empirical_coverage"],
                marker="o", markersize=4,
                color=COLORS.get(model, None),
                linestyle=METHOD_STYLE.get(method, "-"),
                label=display_label(model, method, cs))
    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel("Empirical coverage of the 80% interval")
    ax.set_title("Coverage at every horizon — full XGBoost vs sampled N-BEATS check")
    ax.set_xticks(range(1, 25))
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(save, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save}")


def plot_interval_width(save):
    """The price of the guarantee: how wide are the 80% intervals by horizon?"""
    path = DATA_DIR / "coverage_summary.csv"
    if not path.exists():
        print(f"  SKIP width: {path} missing")
        return
    cs = pd.read_csv(path)
    cs = cs[np.isclose(cs["target_coverage"], 0.80)]
    fig, ax = plt.subplots(figsize=(12, 5))
    for (model, method), sub in cs.groupby(["model", "method"]):
        sub = sub.sort_values("horizon")
        ax.plot(sub["horizon"], sub["mean_width"], marker="o", markersize=4,
                color=COLORS.get(model, None),
                linestyle=METHOD_STYLE.get(method, "-"),
                label=display_label(model, method, cs))
    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel("Mean 80% interval width (°C)")
    ax.set_title("Interval width grows with horizon — the cost of honest uncertainty")
    ax.set_xticks(range(1, 25))
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(save, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save}")


def plot_fan_chart(save, model="xgboost"):
    """A week of 24h-ahead conformal forecasts with the 80% band drawn as a fan.
    Picks consecutive non-overlapping origins so the picture is readable."""
    npz = RAW_DIR / f"{model}_conformal.npz"
    if not npz.exists():
        print(f"  SKIP fan chart: {npz} missing")
        return
    d = np.load(npz)
    lower, upper, truth = d["lower"], d["upper"], d["truth"]
    # Take 7 consecutive origins spaced 24h apart -> one week, no overlap.
    step = 24
    picks = list(range(0, min(7 * step, lower.shape[0]), step))
    fig, ax = plt.subplots(figsize=(13, 5))
    t = 0
    for k, o in enumerate(picks):
        xs = np.arange(t, t + 24)
        ax.fill_between(xs, lower[o], upper[o], color=COLORS.get(model, "#FF6B35"),
                        alpha=0.25, label="80% interval" if k == 0 else None)
        ax.plot(xs, truth[o], color="#222", linewidth=1.5,
                label="actual" if k == 0 else None)
        t += 24
    ax.set_xlabel("Hours into the test week")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"{model}: a week of 24h conformal forecasts — "
                 "the band should contain the black line ~80% of the time")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save}")


def print_headline_table():
    """The number the whole post turns on: 80% interval coverage per model/method."""
    path = DATA_DIR / "coverage_summary.csv"
    if not path.exists():
        return
    cs = pd.read_csv(path)
    cs = cs[np.isclose(cs["target_coverage"], 0.80)]
    print("\n--- Headline: 80% interval, averaged across horizons ---")
    agg = (cs.groupby(["model", "method"])
           .agg(coverage=("empirical_coverage", "mean"),
                width=("mean_width", "mean"))
           .reset_index())
    for _, r in agg.iterrows():
        gap = r["coverage"] - 0.80
        flag = "OK" if abs(gap) <= 0.03 else ("UNDER" if gap < 0 else "OVER")
        print(f"  {r['model']:8s} {r['method']:20s}  "
              f"coverage {r['coverage']:.3f}  gap {gap:+.3f}  "
              f"width {r['width']:.2f}°C  [{flag}]")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Weekend 9 analysis: calibration plots")
    print("=" * 60)
    plot_reliability(OUT / "01_reliability_diagram.png")
    plot_coverage_by_horizon(OUT / "02_coverage_by_horizon.png")
    plot_interval_width(OUT / "03_interval_width.png")
    plot_fan_chart(OUT / "04_fan_chart.png", model="xgboost")
    print_headline_table()
    print("\nPlots in:", OUT.resolve())


if __name__ == "__main__":
    main()
