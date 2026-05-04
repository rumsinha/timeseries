"""
Weekend 2: Making Time Series Behave — Differencing, ACF, and PACF

Picks up exactly where Weekend 1 ended: we have a trend-stationary series
(per ADF/KPSS disagreement). This script:

1. Loads and resamples Jena to hourly (same pipeline as Weekend 1)
2. Applies first-order differencing — the simplest fix
3. Re-runs ADF and KPSS to verify
4. Applies seasonal differencing for the daily cycle
5. Plots ACF and PACF on the transformed series
6. Interprets which lags matter — preview of model order selection

Run end-to-end:
    python weekend_2_differencing_acf.py

Outputs five plots to ./images/weekend_2/

Optional: pass --seasonal-period <int> to experiment with different
seasonal periods (e.g., 168 for weekly, 8766 for yearly).
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = Path("/Users/rumasinha/random/timeseries_analysis/data/jena_climate_2009_2016.csv")
OUTPUT_DIR = Path("images/weekend_2")
RANDOM_SEED = 42  # not strictly needed here, but kept for reproducibility
                  # of any sampling we do downstream

# Plot styling — shared across the series for consistency
plt.rcParams.update({
    "figure.figsize": (12, 4),
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})
sns.set_palette("deep")
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Step 1: Load & resample (identical to Weekend 1)
# ---------------------------------------------------------------------------
def load_hourly_temperature(path: Path) -> pd.Series:
    """Load Jena CSV, parse timestamps, resample to hourly mean."""
    if not path.exists():
        sys.exit(
            f"\nERROR: data file not found at {path.resolve()}\n"
            "Download from https://www.kaggle.com/datasets/mnassrib/jena-climate "
            "and place it in ./data/\n"
        )
    df = pd.read_csv(path)
    df["Date Time"] = pd.to_datetime(
        df["Date Time"], format="%d.%m.%Y %H:%M:%S"
    )
    df = df.set_index("Date Time")
    hourly = df.resample("1h").mean()
    return hourly["T (degC)"].dropna()


# ---------------------------------------------------------------------------
# Step 2: Stationarity testing utility
# ---------------------------------------------------------------------------
def stationarity_report(series: pd.Series, label: str) -> dict:
    """Run ADF and KPSS, print results, return them as a dict for later use."""
    adf_stat, adf_p, *_ = adfuller(series.dropna())
    kpss_stat, kpss_p, *_ = kpss(series.dropna(), regression="c", nlags="auto")

    adf_verdict = "stationary" if adf_p < 0.05 else "non-stationary"
    kpss_verdict = "non-stationary" if kpss_p < 0.05 else "stationary"

    print(f"\n--- {label} ---")
    print(f"ADF  statistic = {adf_stat:>9.4f}   p = {adf_p:.4f}   -> {adf_verdict}")
    print(f"KPSS statistic = {kpss_stat:>9.4f}   p = {kpss_p:.4f}   -> {kpss_verdict}")

    return {
        "label": label,
        "adf_stat": adf_stat, "adf_p": adf_p, "adf_verdict": adf_verdict,
        "kpss_stat": kpss_stat, "kpss_p": kpss_p, "kpss_verdict": kpss_verdict,
    }


# ---------------------------------------------------------------------------
# Step 3: Differencing operations
# ---------------------------------------------------------------------------
def first_difference(series: pd.Series) -> pd.Series:
    """y_t - y_{t-1}.  Removes trend.

    Note: predicting that this difference equals zero IS the Random Walk
    forecast. The math we use here for transformation is the same math
    that produces the strongest baseline most fancy models fail to beat.
    """
    return series.diff().dropna()


def seasonal_difference(series: pd.Series, period: int) -> pd.Series:
    """y_t - y_{t-period}.  Removes seasonality at the given period."""
    return series.diff(period).dropna()


# ---------------------------------------------------------------------------
# Step 4: Plotting helpers
# ---------------------------------------------------------------------------
def plot_series_comparison(
    original: pd.Series,
    differenced: pd.Series,
    title_orig: str,
    title_diff: str,
    save_path: Path,
):
    """Two-panel: original above, differenced below.  Same x-axis."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(original.index, original.values, linewidth=0.5, color="#2E5077")
    axes[0].set_title(title_orig)
    axes[0].set_ylabel("Temperature (°C)")
    axes[1].plot(differenced.index, differenced.values, linewidth=0.5, color="#C04A4A")
    axes[1].set_title(title_diff)
    axes[1].set_ylabel("Δ Temperature (°C)")
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  saved {save_path}")


def plot_acf_pacf(series: pd.Series, lags: int, title: str, save_path: Path):
    """Side-by-side ACF and PACF."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title(f"ACF — {title}")
    axes[0].set_xlabel("Lag (hours)")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], alpha=0.05, method="ywm")
    axes[1].set_title(f"PACF — {title}")
    axes[1].set_xlabel("Lag (hours)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  saved {save_path}")


# ---------------------------------------------------------------------------
# Bonus: tiny Random Walk baseline so the connection is concrete
# ---------------------------------------------------------------------------
def random_walk_mae(series: pd.Series) -> float:
    """MAE of the Random Walk forecast (predict y_t = y_{t-1}).

    This is exactly the mean absolute value of the first-differenced series.
    Demonstrating it numerically reinforces the post's point: differencing
    and the Random Walk are the same operation viewed two ways.
    """
    return series.diff().abs().mean()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main(seasonal_period: int = 24):
    print("=" * 60)
    print("Weekend 2: Differencing, ACF, and PACF")
    print(f"Seasonal period: {seasonal_period} hours")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    temp = load_hourly_temperature(DATA_PATH)
    print(f"\nLoaded {len(temp):,} hourly observations from "
          f"{temp.index.min()} to {temp.index.max()}")

    # Random Walk baseline as a single number — concrete reference for later weekends
    rw_mae = random_walk_mae(temp)
    print(f"\nRandom Walk baseline MAE: {rw_mae:.4f} °C "
          "(every fancy model later in this series must beat this)")

    # Baseline test (callback to Weekend 1 — confirms the diagnosis)
    stationarity_report(temp, "Original hourly temperature")

    # First differencing — removes trend
    temp_d1 = first_difference(temp)
    stationarity_report(temp_d1, "After first differencing (d=1)")

    plot_series_comparison(
        temp, temp_d1,
        "Original hourly temperature (2009–2016)",
        "First-differenced series  (Δy_t = y_t − y_{t−1})",
        OUTPUT_DIR / "01_first_differencing.png",
    )

    # Seasonal differencing on top of first difference
    temp_d1_seasonal = seasonal_difference(temp_d1, period=seasonal_period)
    stationarity_report(
        temp_d1_seasonal,
        f"After first + seasonal-{seasonal_period} differencing "
        f"(d=1, D=1, s={seasonal_period})"
    )

    plot_series_comparison(
        temp_d1, temp_d1_seasonal,
        "First-differenced series",
        f"First + seasonal-{seasonal_period} differenced  "
        "(daily cycle removed)" if seasonal_period == 24 else
        f"First + seasonal-{seasonal_period} differenced",
        OUTPUT_DIR / "02_seasonal_differencing.png",
    )

    # ACF/PACF on the transformed series — this is what we'd feed SARIMA
    plot_acf_pacf(
        temp_d1_seasonal, lags=72,
        title=f"transformed (d=1, D=1, s={seasonal_period})",
        save_path=OUTPUT_DIR / "03_acf_pacf_differenced.png",
    )

    # ACF/PACF on the original series for contrast
    # Use a sample because computing ACF on 70k points is slow and the
    # interpretation doesn't need full resolution
    sample = temp.loc["2014-01-01":"2014-12-31"]
    plot_acf_pacf(
        sample, lags=72,
        title="original (one-year sample)",
        save_path=OUTPUT_DIR / "04_acf_pacf_original.png",
    )

    # The "memory map" — ACF up to 168 hours (one week) showing the daily echoes
    fig, ax = plt.subplots(figsize=(14, 4))
    plot_acf(sample, lags=168, ax=ax, alpha=0.05)
    ax.set_title("ACF up to 168 hours — the rhythm of the week")
    ax.set_xlabel("Lag (hours)")
    # Annotate the 24h echoes
    for k in range(1, 8):
        ax.axvline(24 * k, color="red", alpha=0.2, linestyle="--", linewidth=0.8)
    ax.text(24, ax.get_ylim()[1] * 0.9, "24h", color="red", fontsize=9)
    ax.text(48, ax.get_ylim()[1] * 0.9, "48h", color="red", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_acf_weekly.png", bbox_inches="tight")
    plt.close()
    print(f"  saved {OUTPUT_DIR / '05_acf_weekly.png'}")

    # Save the transformed series so Weekend 3 doesn't have to re-difference
    transformed_path = Path("data/weekend_2_transformed.parquet")
    transformed_path.parent.mkdir(exist_ok=True)
    temp_d1_seasonal.to_frame("temp_transformed").to_parquet(transformed_path)
    print(f"\n  saved transformed series to {transformed_path} for Weekend 3")

    print("\n" + "=" * 60)
    print("Done. Plots in:", OUTPUT_DIR.resolve())
    print(f"Random Walk MAE to beat: {rw_mae:.4f} °C")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasonal-period",
        type=int,
        default=24,
        help="Seasonal period in hours. Default 24 (daily). "
             "Try 168 for weekly or 8766 for yearly.",
    )
    args = parser.parse_args()
    main(seasonal_period=args.seasonal_period)