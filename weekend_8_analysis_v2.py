"""
Weekend 8 (analysis): build the cross-model comparison plots.

Separate from weekend_8_tuning_v9.py so you can iterate on charts without re-running
the 2-hour tuning job. Reads the artifacts the tuning script wrote, plus the
per-horizon CSVs from Weekends 6-7, and produces:

    01_cv_fold_diagram.png        how the 3 expanding-window folds are laid out
    02_per_horizon_all_models.png MAE-by-horizon for every model on one chart
    03_tuning_before_after.png    untuned vs tuned MAE bars
    04_leaderboard_tuned.png      the updated leaderboard

Run after weekend_8_tuning_v9.py:
    python weekend_8_analysis_v2.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths (mirror the tuning script's anchors)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOK_ROOT = SCRIPT_DIR
TUNING_DIR = NOTEBOOK_ROOT / "tuning_weekend_8"
PER_HORIZON_DIR = TUNING_DIR / "per_horizon"
OUTPUT_DIR = NOTEBOOK_ROOT / "images" / "weekend_8"
LEADERBOARD_PATH = NOTEBOOK_ROOT / "data" / "leaderboard.csv"

OUTPUT_HORIZON = 24
EXPECTED_ORIGINS = 8593
EXPECTED_MODELS = {"sarima", "xgboost", "lightgbm", "lstm", "nbeats"}

COLORS = {
    "sarima": "#4C78A8", "SARIMA": "#4C78A8",
    "xgboost": "#FF6B35", "XGBoost": "#FF6B35",
    "lightgbm": "#1E8E5A", "LightGBM": "#1E8E5A",
    "lstm": "#7B4F8B", "LSTM": "#7B4F8B",
    "nbeats": "#D97757", "N-BEATS": "#D97757",
}

plt.rcParams.update({
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
})
sns.set_palette("deep")


def plot_cv_fold_diagram(save_path: Path):
    """Visualize the 3 expanding-window folds + sealed test. Pure schematic."""
    fig, ax = plt.subplots(figsize=(12, 4))
    years = list(range(2009, 2017))
    rows = [
        ("Fold 1", 2009, 2012, 2013),
        ("Fold 2", 2009, 2013, 2014),
        ("Fold 3", 2009, 2014, 2015),
        ("Final",  2009, 2015, 2016),
    ]
    seen = set()  # ensures each legend label is registered exactly once

    def once(key):
        # Return the label the first time we see it, None afterwards, so the
        # legend lists Train / Validate / Test (sealed) one time each.
        if key in seen:
            return None
        seen.add(key)
        return key

    for i, (label, tr_s, tr_e, va) in enumerate(rows):
        y = len(rows) - i
        # training span
        ax.barh(y, tr_e - tr_s + 1, left=tr_s, height=0.6,
                color="#4C78A8", alpha=0.85, label=once("Train"))
        # validation / test year — label by WHAT THE BAR IS, not the row index
        is_test = (label == "Final")
        bar_kind = "Test (sealed)" if is_test else "Validate"
        ax.barh(y, 1, left=va, height=0.6,
                color="#C04A4A" if is_test else "#E89B3B", alpha=0.9,
                label=once(bar_kind))
        ax.text(tr_s - 0.15, y, label, ha="right", va="center", fontsize=10)

    ax.set_xlim(2008.3, 2017)
    ax.set_ylim(0.3, len(rows) + 0.7)
    ax.set_yticks([])
    ax.set_xticks(years)
    ax.set_xlabel("Year")
    ax.set_title("Expanding-window time-series cross-validation "
                 "(validation always follows training)")
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def plot_per_horizon_all(save_path: Path):
    """Every model's tuned per-horizon MAE on one chart — the headline plot."""
    fig, ax = plt.subplots(figsize=(13, 6))
    horizons = np.arange(1, OUTPUT_HORIZON + 1)

    plotted_any = False
    if PER_HORIZON_DIR.exists():
        # Tuned models
        for csv in sorted(PER_HORIZON_DIR.glob("*_tuned_horizon.csv")):
            model = csv.stem.replace("_tuned_horizon", "")
            dfm = pd.read_csv(csv)
            ax.plot(dfm["horizon"], dfm["mae"], marker="o", markersize=5,
                    linewidth=2, label=f"{model} (tuned)",
                    color=COLORS.get(model, None))
            plotted_any = True
        # Baselines — dashed grey, the sanity check Weekend 7 lacked. If
        # persistence sits below a model at h=1, that model's short-horizon
        # "win" is losing to a one-liner.
        for name, style in (("persistence", ":"), ("seasonal_naive", "--")):
            bcsv = PER_HORIZON_DIR / f"{name}_horizon.csv"
            if bcsv.exists():
                dfb = pd.read_csv(bcsv)
                ax.plot(dfb["horizon"], dfb["mae"], linestyle=style,
                        linewidth=1.8, color="#888888", label=name)
                plotted_any = True

    if not plotted_any:
        ax.text(0.5, 0.5, "No per-horizon CSVs found.\nRun weekend_8_tuning_v9.py first.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)

    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel("MAE (°C)")
    ax.set_title("Per-horizon error across all tuned models — "
                 "which model to use at which horizon")
    ax.set_xticks(horizons)
    ax.tick_params(axis="x", labelsize=8)
    if plotted_any:
        ax.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def plot_before_after(save_path: Path):
    """Prior published vs Weekend 8 MAE, side by side per model."""
    tuned_csv = TUNING_DIR / "tuned_leaderboard.csv"
    if not tuned_csv.exists():
        print(f"  SKIP before/after: {tuned_csv} not found")
        return
    tuned = pd.read_csv(tuned_csv).set_index("model")["mae"]

    # Prior published numbers from the persistent leaderboard.
    name_map = {"SARIMA": "sarima", "XGBoost": "xgboost", "LightGBM": "lightgbm",
                "LSTM": "lstm", "N-BEATS": "nbeats"}
    untuned = {}
    if LEADERBOARD_PATH.exists():
        lb = pd.read_csv(LEADERBOARD_PATH)
        for _, row in lb.iterrows():
            key = name_map.get(row["model"])
            if key:
                untuned[key] = float(row["mae"])

    models = [m for m in tuned.index if m in untuned]
    if not models:
        print("  SKIP before/after: no overlapping models with prior leaderboard")
        return

    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, [untuned[m] for m in models], w, label="Prior published",
           color="#B0B0B0")
    ax.bar(x + w/2, [tuned[m] for m in models], w, label="Weekend 8",
           color=[COLORS.get(m, "#444") for m in models])
    for i, m in enumerate(models):
        delta = tuned[m] - untuned[m]
        ax.annotate(f"{delta:+.3f}", (i + w/2, tuned[m]),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8,
                    color="#1E8E5A" if delta < 0 else "#C04A4A")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Test MAE (°C)")
    ax.set_title("Prior published implementation vs Weekend 8 test MAE")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def plot_leaderboard_tuned(save_path: Path):
    """Updated leaderboard bar chart using tuned numbers where available."""
    tuned_csv = TUNING_DIR / "tuned_leaderboard.csv"
    if not tuned_csv.exists():
        print(f"  SKIP leaderboard: {tuned_csv} not found")
        return
    tuned = pd.read_csv(tuned_csv)
    pretty = {"sarima": "SARIMA", "xgboost": "XGBoost", "lightgbm": "LightGBM",
              "lstm": "LSTM", "nbeats": "N-BEATS"}
    tuned["display"] = tuned["model"].map(lambda m: pretty.get(m, m))
    tuned = tuned.sort_values("mae")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLORS.get(m, "#888") for m in tuned["model"]]
    bars = ax.barh(tuned["display"], tuned["mae"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Test MAE (°C) — lower is better")
    ax.set_title("Weekend 8 leaderboard: tuned models "
                 "(MAE averaged across horizons h=1..24)")
    for bar, val in zip(bars, tuned["mae"]):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 24):
    """Diebold-Mariano test on per-origin MAE loss differentials.

    e1, e2: (n_origins, 24) absolute-error matrices for two models on the
    SAME origins. Loss per origin = mean abs error across the 24 horizons.
    Uses HAC (Newey-West) variance with h-1 lags to account for the serial
    correlation that overlapping multi-step forecasts induce, plus the
    Harvey-Leybourne-Newbold small-sample correction.

    Returns (dm_stat, p_value). Negative dm => model 1 more accurate.
    """
    from scipy import stats as scipy_stats
    d = e1.mean(axis=1) - e2.mean(axis=1)        # loss differential per origin
    n = len(d)
    dbar = d.mean()
    # Newey-West long-run variance with h-1 lags
    gamma0 = np.var(d, ddof=0)
    s = gamma0
    for lag in range(1, h):
        cov = np.mean((d[lag:] - dbar) * (d[:-lag] - dbar))
        s += 2 * (1 - lag / h) * cov
    var_dbar = s / n
    if var_dbar <= 0:
        return float("nan"), float("nan")
    dm = dbar / np.sqrt(var_dbar)
    # HLN small-sample correction
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_corr = dm * hln
    p = 2 * scipy_stats.t.sf(abs(dm_corr), df=n - 1)
    return float(dm_corr), float(p)


def run_dm_tests(save_path: Path):
    """Pairwise DM tests between every model with saved raw residuals.

    This converts the series' three hand-waved 'they're tied' claims into a
    measured statement: either the differences are significant or they aren't.
    """
    raws = {}
    truths = {}
    for npz in sorted(PER_HORIZON_DIR.glob("*_raw.npz")):
        model = npz.stem.replace("_raw", "")
        data = np.load(npz)
        truths[model] = data["y_true"]
        raws[model] = np.abs(data["y_true"] - data["y_pred"])
    if len(raws) < 2:
        print("  SKIP DM tests: need >=2 models with raw residuals")
        return

    rows = []
    models = sorted(raws.keys())
    # DM requires same origins; only compare models with equal origin counts
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            if raws[m1].shape != raws[m2].shape:
                rows.append({"model_1": m1, "model_2": m2, "dm": np.nan,
                             "p_value": np.nan,
                             "note": "origin counts differ (protocol mismatch)"})
                continue
            if not np.allclose(truths[m1], truths[m2], rtol=0, atol=1e-5):
                rows.append({"model_1": m1, "model_2": m2, "dm": np.nan,
                             "p_value": np.nan,
                             "note": "actual targets differ (origin mismatch)"})
                continue
            dm, p = diebold_mariano(raws[m1], raws[m2])
            rows.append({"model_1": m1, "model_2": m2, "dm": dm,
                         "p_value": p, "note": ""})
    out = pd.DataFrame(rows)
    out["p_value_holm"] = np.nan
    valid = out["p_value"].notna()
    if valid.any():
        ordered = out.loc[valid, "p_value"].sort_values()
        m = len(ordered)
        adjusted = []
        running_max = 0.0
        for rank, (_, p_value) in enumerate(ordered.items()):
            candidate = min(1.0, (m - rank) * p_value)
            running_max = max(running_max, candidate)
            adjusted.append(running_max)
        out.loc[ordered.index, "p_value_holm"] = adjusted
    out["significant_holm_0_05"] = out["p_value_holm"] < 0.05
    out.to_csv(save_path, index=False)
    print(f"  saved DM tests -> {save_path}")
    for _, r in out.iterrows():
        if pd.notna(r["dm"]):
            verdict = ("significant" if r["p_value_holm"] < 0.05 else
                       "NOT significant")
            print(f"    {r['model_1']:10s} vs {r['model_2']:10s}  "
                  f"DM={r['dm']:+.2f}  raw p={r['p_value']:.3f}  "
                  f"Holm p={r['p_value_holm']:.3f}  ({verdict})")
        else:
            print(f"    {r['model_1']:10s} vs {r['model_2']:10s}  "
                  f"skipped: {r['note']}")


def validate_artifacts():
    """Fail before plotting if model outputs do not share the v9 protocol."""
    leaderboard_path = TUNING_DIR / "tuned_leaderboard.csv"
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Missing {leaderboard_path}; run v9 tuning first")
    leaderboard = pd.read_csv(leaderboard_path)
    models = set(leaderboard["model"])
    if models != EXPECTED_MODELS:
        missing = sorted(EXPECTED_MODELS - models)
        extra = sorted(models - EXPECTED_MODELS)
        raise ValueError(f"Incomplete leaderboard; missing={missing}, extra={extra}")
    if "version" not in leaderboard.columns or not (leaderboard["version"] == "v9").all():
        raise ValueError(
            "Leaderboard mixes stale artifacts. Re-run every model with "
            "weekend_8_tuning_v9.py."
        )
    bad_protocol = leaderboard.loc[leaderboard["protocol"] != "stride-1", "model"]
    if not bad_protocol.empty:
        raise ValueError(
            "Stale/noncanonical artifacts for: " + ", ".join(bad_protocol) +
            ". Re-run weekend_8_tuning_v9.py."
        )

    reference = None
    for model in leaderboard["model"]:
        path = PER_HORIZON_DIR / f"{model}_raw.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw predictions for {model}: {path}")
        data = np.load(path)
        y_true, y_pred = data["y_true"], data["y_pred"]
        expected = (EXPECTED_ORIGINS, OUTPUT_HORIZON)
        if y_true.shape != expected or y_pred.shape != expected:
            raise ValueError(f"{model} has shape {y_true.shape}; expected {expected}")
        if reference is None:
            reference = y_true
        elif not np.allclose(reference, y_true, rtol=0, atol=1e-5):
            raise ValueError(f"{model} actual targets do not align with other models")


def plot_nonoverlapping_acf(save_path: Path):
    """Settle the recurring W6/W7 claim: is short-lag residual ACF real
    signal, or an artifact of stride-1 origins sharing 167/168 input rows?

    Take each model's h=24 residuals at every 24th origin (non-overlapping
    forecast windows) and plot that ACF. If the short-lag structure survives
    de-overlapping, it's real; if it vanishes, it was the artifact.
    """
    from statsmodels.graphics.tsaplots import plot_acf

    raws = sorted(PER_HORIZON_DIR.glob("*_raw.npz"))
    if not raws:
        print("  SKIP non-overlapping ACF: no raw residuals found")
        return
    n_models = len(raws)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4),
                             squeeze=False)
    for ax, npz in zip(axes[0], raws):
        model = npz.stem.replace("_raw", "")
        data = np.load(npz)
        resid_h24 = (data["y_true"] - data["y_pred"])[:, -1]
        non_overlap = resid_h24[::24]            # 24h-spaced origins
        plot_acf(non_overlap, lags=min(30, len(non_overlap) // 3), ax=ax,
                 alpha=0.05)
        ax.set_title(f"{model}: h=24 residual ACF,\n"
                     f"non-overlapping origins (n={len(non_overlap)})")
        ax.set_xlabel("Lag (days)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)

    validate_artifacts()
    print("Weekend 8 analysis: cross-model comparison plots")
    print("=" * 60)

    plot_cv_fold_diagram(OUTPUT_DIR / "01_cv_fold_diagram.png")
    plot_per_horizon_all(OUTPUT_DIR / "02_per_horizon_all_models.png")
    plot_before_after(OUTPUT_DIR / "03_tuning_before_after.png")
    plot_leaderboard_tuned(OUTPUT_DIR / "04_leaderboard_tuned.png")
    plot_nonoverlapping_acf(OUTPUT_DIR / "05_nonoverlapping_acf.png")
    print("\n--- Diebold-Mariano significance tests ---")
    run_dm_tests(OUTPUT_DIR / "dm_tests.csv")

    # Print the best configs if present
    cfg_path = TUNING_DIR / "best_configs.json"
    if cfg_path.exists():
        print("\nBest configs found by tuning:")
        with open(cfg_path) as fh:
            for model, cfg in json.load(fh).items():
                print(f"  {model}: {cfg}")

    print("\n" + "=" * 60)
    print("Plots in:", OUTPUT_DIR.resolve())
    print("=" * 60)


if __name__ == "__main__":
    main()