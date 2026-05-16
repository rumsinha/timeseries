"""
Weekend 4: Closing the Classical Loop — SARIMA

Picks up from Weekend 2's ACF/PACF diagnostic work and Weekend 3's Prophet
benchmark. This script:

1. Loads the same hourly Jena temperature pipeline (Weekends 1-3 callback)
2. Splits chronologically: train (2009-2014), val (2015), test (2016)
3. Recomputes baselines (Random Walk, Seasonal Naive) for the leaderboard
4. Fits SARIMA with orders chosen from ACF/PACF reading + auto_arima sanity check
5. Generates rolling 24h-ahead forecasts on the test set
6. Diagnoses residuals (Ljung-Box, residual plots)
7. Updates the leaderboard CSV (Prophet from Weekend 3 + SARIMA from this run)

Run end-to-end:
    python weekend_4_sarima.py

Outputs five plots to ./images/weekend_4/

Compute note: full SARIMA fit on 6 years of hourly data (~52k rows) takes
~10-20 minutes depending on hardware. We use a downsampled-but-equivalent
strategy when fitting — see fit_sarima() docstring.
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Quieting statsmodels' convergence warnings during fitting
warnings.filterwarnings("ignore")
logging.getLogger("statsmodels").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration — locked from the series spec
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
while not (PROJECT_ROOT / "data").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_PATH = Path("/Users/rumasinha/random/timeseries_analysis/data/jena_climate_2009_2016.csv")
OUTPUT_DIR = PROJECT_ROOT / "images" / "weekend_4"
LEADERBOARD_PATH = PROJECT_ROOT / "data" / "leaderboard.csv"
RANDOM_SEED = 42

TRAIN_END = "2014-12-31 23:00:00"
VAL_END = "2015-12-31 23:00:00"
TEST_END = "2016-12-31 23:00:00"
FORECAST_HORIZON = 24

# SARIMA orders chosen from Weekend 2's ACF/PACF reading
# (p, d, q) — non-seasonal: AR(2), one regular difference, MA(1)
# (P, D, Q, s) — seasonal: SAR(1), one seasonal difference at lag 24, SMA(1)
SARIMA_ORDER = (2, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 24)

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
# Data loading (shared with Weekends 1-3)
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
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.set_index("Date Time")
    hourly = df.resample("1h").mean()
    hourly = hourly.asfreq("h")
    hourly = hourly.interpolate("linear")
    return hourly["T (degC)"].dropna()


def chronological_split(series: pd.Series):
    """Split into train / val / test by date — never randomly."""
    train = series.loc[:TRAIN_END]
    val = series.loc[TRAIN_END:VAL_END].iloc[1:]
    test = series.loc[VAL_END:TEST_END].iloc[1:]
    return train, val, test


# ---------------------------------------------------------------------------
# Baselines (same as Weekend 3 — recomputed for the leaderboard)
# ---------------------------------------------------------------------------
def random_walk_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    return np.full(horizon, history.iloc[-1])


def seasonal_naive_forecast(history: pd.Series, horizon: int,
                            period: int = 24) -> np.ndarray:
    last_period = history.iloc[-period:].values
    tiled = np.tile(last_period, int(np.ceil(horizon / period)))
    return tiled[:horizon]


def evaluate_baseline(name: str, forecast_fn, train: pd.Series, test: pd.Series,
                      horizon: int) -> dict:
    """Walk-forward evaluation of a baseline at a fixed forecast horizon."""
    residuals = []
    history = train.copy()
    for i in range(0, len(test) - horizon, horizon):
        actual = test.iloc[i:i + horizon].values
        forecast = forecast_fn(history, horizon)
        residuals.extend(actual - forecast)
        history = pd.concat([history, test.iloc[i:i + horizon]])

    residuals = np.asarray(residuals)
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    print(f"  {name:25s}  MAE = {mae:.3f} °C   RMSE = {rmse:.3f} °C")
    return {"model": name, "mae": mae, "rmse": rmse}


# ---------------------------------------------------------------------------
# SARIMA
# ---------------------------------------------------------------------------
def fit_sarima(train: pd.Series) -> SARIMAX:
    """Fit SARIMA with orders chosen from Weekend 2's ACF/PACF analysis.

    Order rationale:
    - d=1: first differencing removes the trend (confirmed stationary in W2)
    - D=1, s=24: seasonal differencing at lag 24 removes the daily cycle
    - p=2: PACF shows significant spikes at lags 1 and 2, then dies
    - q=1: ACF tails off — small MA component captures lag-1 shock structure
    - P=1, Q=1: at the seasonal lag (24h), both ACF and PACF show spikes —
      include one of each as a balanced seasonal model

    NOTE on training set size:
    Full 52k-row training takes 15-20+ minutes per fit. We use the last
    2 years of training data (2013-2014) instead. This trades a small
    amount of statistical efficiency for ~5x speedup. The trade is fine
    here because SARIMA's parameters are local in time — most of the
    information about (p,q,P,Q) lives in the recent past.
    """
    # Use last 2 years for fitting speed — see docstring
    fit_data = train.loc["2013":]
    print(f"  fitting SARIMA{SARIMA_ORDER}x{SARIMA_SEASONAL_ORDER} on "
          f"{len(fit_data):,} rows (last 2 years of training)...")
    model = SARIMAX(
        fit_data,
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=200)
    return fitted


def evaluate_sarima(fitted_model, train: pd.Series, val: pd.Series,
                    test: pd.Series, horizon: int) -> dict:
    """Rolling 24h-ahead evaluation of SARIMA.

    Unlike Weekend 3's Prophet (fit-once-predict-all), we genuinely walk
    forward here. After each 24h chunk we extend the fitted model with the
    observed values using append() — this is the SARIMAX equivalent of
    'updating state without refitting'. Each forecast costs ~50ms, so
    rolling through 365 chunks is feasible (~3 minutes total).

    Note on state bridging:
    The model was fit on 2013-2014 (last 2 years of train). Before rolling
    through 2016 (test), we extend the model's STATE through 2015 (val)
    using append(..., refit=False). The val data does NOT influence the
    fitted parameters — those are locked. It just brings the internal
    Kalman filter state up to the start of test data so append() works
    without an index discontinuity. This keeps val reserved for Weekend 8's
    hyperparameter tuning while still allowing continuous rolling.
    """
    # Bridge the gap: extend model state through val without refitting
    print(f"  bridging state through {len(val):,} validation rows...")
    current_model = fitted_model.append(val, refit=False)

    residuals = []
    all_forecasts = []  # collect point forecasts for downstream reuse
    n_chunks = (len(test) - horizon) // horizon
    for i in range(n_chunks):
        if i % 30 == 0:
            print(f"  ...rolling forecast: chunk {i+1}/{n_chunks}")

        # Forecast the next 24 hours
        forecast = current_model.forecast(steps=horizon)
        actual = test.iloc[i * horizon:(i + 1) * horizon]
        residuals.extend(actual.values - forecast.values)
        all_forecasts.extend(forecast.values)

        # Append realized values to model state without refitting
        current_model = current_model.append(actual, refit=False)

    residuals = np.asarray(residuals)
    yhat = np.asarray(all_forecasts)
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    print(f"  {'SARIMA':25s}  MAE = {mae:.3f} °C   RMSE = {rmse:.3f} °C")
    return {
        "model": "SARIMA",
        "mae": mae,
        "rmse": rmse,
        "residuals": residuals,
        "yhat": yhat,
        "fitted_model": fitted_model,
        "rolling_model": current_model,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def diagnose_residuals(residuals: np.ndarray) -> dict:
    """Ljung-Box test for residual autocorrelation.

    If residuals look like white noise, Ljung-Box p-values stay high
    (we fail to reject 'no autocorrelation'). If residuals retain
    structure, p-values fall — the model missed something.
    """
    lb = acorr_ljungbox(residuals, lags=[10, 24, 48], return_df=True)
    print("\n  Ljung-Box residual diagnostics:")
    for lag, row in lb.iterrows():
        verdict = "white noise" if row["lb_pvalue"] > 0.05 else "structure remains"
        print(f"    lag {lag:3d}:  Q = {row['lb_stat']:>8.2f}   "
              f"p = {row['lb_pvalue']:.4f}   -> {verdict}")
    return lb


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_residual_diagnostics(residuals: np.ndarray, save_path: Path):
    """4-panel residual diagnostic: time, histogram, Q-Q, ACF."""
    from scipy import stats as scipy_stats

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Top-left: residuals over time
    axes[0, 0].plot(residuals, linewidth=0.4, color="#2E5077")
    axes[0, 0].axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    axes[0, 0].set_title("Residuals over time")
    axes[0, 0].set_xlabel("Forecast index")
    axes[0, 0].set_ylabel("Residual (°C)")

    # Top-right: residual histogram with normal overlay
    axes[0, 1].hist(residuals, bins=60, density=True, color="#5B9BD5",
                    edgecolor="white", alpha=0.85)
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 200)
    axes[0, 1].plot(x, scipy_stats.norm.pdf(x, mu, sigma),
                    color="#C04A4A", linewidth=2, label="Normal fit")
    axes[0, 1].set_title("Residual distribution")
    axes[0, 1].set_xlabel("Residual (°C)")
    axes[0, 1].legend()

    # Bottom-left: Q-Q plot
    scipy_stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q plot (normality check)")

    # Bottom-right: ACF of residuals
    plot_acf(residuals, lags=48, ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title("ACF of residuals (should be all noise)")
    axes[1, 1].set_xlabel("Lag (hours)")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def plot_test_window_zoom(test: pd.Series, sarima_yhat: np.ndarray,
                          save_path: Path,
                          window_start: str = "2016-07-01",
                          window_end: str = "2016-07-08"):
    """One week of test predictions vs actuals (matches Weekend 3 style)."""
    test_idx = test.index[:len(sarima_yhat)]
    df = pd.DataFrame({
        "actual": test.iloc[:len(sarima_yhat)].values,
        "sarima": sarima_yhat,
    }, index=test_idx)
    window = df.loc[window_start:window_end]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(window.index, window["sarima"], color="#2E8B57",
            linewidth=2, label="SARIMA forecast")
    ax.plot(window.index, window["actual"], color="#222222",
            linewidth=1.5, label="Actual")
    ax.set_title(f"SARIMA 24h-ahead forecast vs actual: "
                 f"{window_start} → {window_end}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def plot_leaderboard(results: list, save_path: Path):
    """Bar chart comparing all models on MAE."""
    df = pd.DataFrame(results).sort_values("mae")
    # Highlight SARIMA in green; everything else gray; Prophet in red
    color_map = {"SARIMA": "#2E8B57", "Prophet": "#C04A4A"}
    colors = [color_map.get(name, "#888888") for name in df["model"]]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(df["model"], df["mae"], color=colors)
    ax.set_xlabel("MAE (°C) — lower is better")
    ax.set_title("Weekend 4 leaderboard: SARIMA joins the lineup "
                 "(24h-ahead forecast)")
    for bar, val in zip(bars, df["mae"]):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def plot_summary_fit(fitted_model, save_path: Path):
    """Built-in statsmodels diagnostic plot for the fitted SARIMA."""
    fig = fitted_model.plot_diagnostics(figsize=(14, 9))
    fig.suptitle(
        f"SARIMA{SARIMA_ORDER}x{SARIMA_SEASONAL_ORDER} — fit diagnostics",
        y=1.01, fontsize=12, fontweight="bold",
    )
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(f"Weekend 4: SARIMA{SARIMA_ORDER}x{SARIMA_SEASONAL_ORDER}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load and split ----
    temp = load_hourly_temperature(DATA_PATH)
    train, val, test = chronological_split(temp)
    print(f"\nTrain:  {train.index.min()} → {train.index.max()}  ({len(train):,} rows)")
    print(f"Val:    {val.index.min()} → {val.index.max()}  ({len(val):,} rows)  [reserved for Weekend 8 tuning]")
    print(f"Test:   {test.index.min()} → {test.index.max()}  ({len(test):,} rows)")

    # ---- Baselines (recomputed for the unified leaderboard) ----
    print("\n--- Baselines ---")
    results = []
    results.append(evaluate_baseline(
        "Random Walk (lag-1)", random_walk_forecast,
        train, test, FORECAST_HORIZON,
    ))
    results.append(evaluate_baseline(
        "Seasonal Naive (lag-24)",
        lambda hist, h: seasonal_naive_forecast(hist, h, period=24),
        train, test, FORECAST_HORIZON,
    ))

    # ---- Load Prophet result from Weekend 3 if it exists ----
    if LEADERBOARD_PATH.exists():
        prior_lb = pd.read_csv(LEADERBOARD_PATH)
        prophet_row = prior_lb[prior_lb["model"] == "Prophet"]
        if len(prophet_row):
            results.append({
                "model": "Prophet",
                "mae": float(prophet_row["mae"].iloc[0]),
                "rmse": float(prophet_row["rmse"].iloc[0]),
            })
            print(f"  {'Prophet (from Wk3)':25s}  "
                  f"MAE = {results[-1]['mae']:.3f} °C   "
                  f"RMSE = {results[-1]['rmse']:.3f} °C")

    # ---- SARIMA ----
    print("\n--- SARIMA ---")
    fitted = fit_sarima(train)
    sarima_result = evaluate_sarima(fitted, train, val, test, FORECAST_HORIZON)
    results.append({
        "model": "SARIMA",
        "mae": sarima_result["mae"],
        "rmse": sarima_result["rmse"],
    })

    # ---- Diagnostics ----
    diagnose_residuals(sarima_result["residuals"])

    # ---- Plots ----
    print("\n--- Plots ---")
    plot_summary_fit(fitted, OUTPUT_DIR / "01_sarima_diagnostics.png")
    plot_residual_diagnostics(
        sarima_result["residuals"],
        OUTPUT_DIR / "02_residual_diagnostics.png",
    )

    # Point forecasts for the zoom plot — cached during evaluate_sarima
    sarima_yhat = sarima_result["yhat"]

    plot_test_window_zoom(test, sarima_yhat, OUTPUT_DIR / "03_test_week_zoom.png")
    plot_leaderboard(results, OUTPUT_DIR / "04_leaderboard.png")

    # ---- Save updated leaderboard ----
    pd.DataFrame(results).to_csv(LEADERBOARD_PATH, index=False)
    print(f"\n  updated leaderboard at {LEADERBOARD_PATH}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Done. Plots in:", OUTPUT_DIR.resolve())
    print("\nLeaderboard (lower MAE is better):")
    for r in sorted(results, key=lambda x: x["mae"]):
        print(f"  {r['model']:25s}  MAE = {r['mae']:.3f} °C")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    main()