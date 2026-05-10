"""
Weekend 3: The Predictable Past — Prophet for Time Series

Picks up from Weekend 2: we have a stationary series, we understand its
memory, and we have a Random Walk baseline MAE to beat. This script:

1. Loads the same hourly Jena temperature pipeline (Weekends 1-2 callback)
2. Splits chronologically: train (2009-2014), val (2015), test (2016)
3. Establishes baselines: Random Walk + Seasonal Naive
4. Fits a Prophet model with daily + yearly seasonality
5. Generates 24h-ahead forecasts on the test set
6. Visualizes the components, changepoints, and uncertainty intervals
7. Updates the leaderboard

Run end-to-end:
    python weekend_3_prophet.py

Outputs six plots to ./images/weekend_3/

Note: Prophet uses 'ds' (datestamp) and 'y' (value) as required column names.
Don't fight it — rename your columns and move on.
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from prophet import Prophet

# Prophet is verbose by default; quiet it down for cleaner notebooks
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration — locked from the series spec
# ---------------------------------------------------------------------------
# Resolve paths relative to the project root, regardless of where the
# script is launched from (e.g. from notebooks/ vs project root).
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
while not (PROJECT_ROOT / "data").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_PATH = Path("/Users/rumasinha/random/timeseries_analysis/data/jena_climate_2009_2016.csv")
OUTPUT_DIR = Path("images/weekend_3")
RANDOM_SEED = 42

TRAIN_END = "2014-12-31 23:00:00"
VAL_END = "2015-12-31 23:00:00"
TEST_END = "2016-12-31 23:00:00"

FORECAST_HORIZON = 24  # 24 hours ahead — locked from series spec

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
# Data loading (shared with Weekends 1-2; will eventually live in src/)
# ---------------------------------------------------------------------------
def load_hourly_temperature(path: Path) -> pd.Series:
    """Load Jena CSV, parse timestamps, resample to hourly mean.

    Adds two production-ready touches that classical and ML pipelines both
    expect:
    - Explicit hourly frequency on the index (Prophet is more stable when
      the frequency is set rather than inferred).
    - Linear interpolation of any small gaps. Jena is a clean dataset, but
      defensive interpolation prevents downstream surprises.
    """
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
    hourly = hourly.asfreq("h")           # explicit frequency
    hourly = hourly.interpolate("linear") # fill any small gaps
    return hourly["T (degC)"].dropna()


def chronological_split(series: pd.Series):
    """Split into train / val / test by date — never randomly."""
    train = series.loc[:TRAIN_END]
    val = series.loc[TRAIN_END:VAL_END].iloc[1:]  # exclude boundary overlap
    test = series.loc[VAL_END:TEST_END].iloc[1:]
    return train, val, test


# ---------------------------------------------------------------------------
# Baselines (from Weekend 2's groundwork — same operations, used as forecasts)
# ---------------------------------------------------------------------------
def random_walk_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    """y_hat_{t+1..t+h} = y_t.  The 'do nothing' baseline."""
    return np.full(horizon, history.iloc[-1])


def seasonal_naive_forecast(history: pd.Series, horizon: int, period: int = 24) -> np.ndarray:
    """y_hat_{t+k} = y_{t+k-period}.  'Tomorrow at 3pm = today at 3pm.'

    Generalized to handle horizons longer than one seasonal period: we tile
    the last period and slice. Currently horizon==period in this script,
    but later weekends may use longer horizons (e.g., 168h ahead).
    """
    last_period = history.iloc[-period:].values
    tiled = np.tile(last_period, int(np.ceil(horizon / period)))
    return tiled[:horizon]


def evaluate_baseline(name: str, forecast_fn, train: pd.Series, test: pd.Series,
                      horizon: int) -> dict:
    """Walk-forward evaluation of a baseline at a fixed forecast horizon."""
    residuals = []
    history = train.copy()
    # Step through test set in chunks of `horizon` hours
    for i in range(0, len(test) - horizon, horizon):
        actual = test.iloc[i:i + horizon].values
        forecast = forecast_fn(history, horizon)
        residuals.extend(actual - forecast)
        # Walk forward: append the realized values to history
        history = pd.concat([history, test.iloc[i:i + horizon]])

    residuals = np.asarray(residuals)
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    print(f"  {name:25s}  MAE = {mae:.3f} °C   RMSE = {rmse:.3f} °C")
    return {"model": name, "mae": mae, "rmse": rmse}


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------
def prepare_for_prophet(series: pd.Series) -> pd.DataFrame:
    """Prophet wants exactly two columns: ds (datestamp) and y (value)."""
    return pd.DataFrame({"ds": series.index, "y": series.values})


def fit_prophet(train_df: pd.DataFrame) -> Prophet:
    """Fit Prophet with daily + yearly seasonality enabled.

    Notes:
    - Hourly data ⇒ enable daily_seasonality
    - Multi-year span ⇒ enable yearly_seasonality
    - Weekly seasonality is DISABLED: weather doesn't care about weekdays,
      so adding a weekly component just fits noise. (For business data —
      sales, traffic, support tickets — you'd want it on.)
    - We let Prophet auto-detect changepoints in the trend
    - changepoint_prior_scale controls trend flexibility (default 0.05)
    """
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=False,  # weather doesn't have a weekly cycle
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        interval_width=0.80,       # 80% prediction intervals
    )
    print("  fitting Prophet on", len(train_df), "training rows...")
    model.fit(train_df)
    return model


def evaluate_prophet(model: Prophet, train: pd.Series, test: pd.Series,
                     horizon: int) -> dict:
    """Evaluate Prophet forecasts at a fixed horizon.

    IMPORTANT — fairness disclosure:
    Unlike the baselines (which truly walk forward, growing their history
    as each test chunk is observed), Prophet here is fit ONCE on the
    training set and then asked to predict the entire test horizon as a
    single batch. We do NOT refit at every step.

    Why: refitting Prophet hourly across an 8,784-hour test set would take
    hours of wall-clock time and rarely changes results materially.
    Production systems typically refit daily or weekly — that tradeoff is
    covered in the Weekend 10 post on production realities.

    Implication: the comparison between Prophet and the baselines is
    slightly tilted in the baselines' favor (they get to see realized
    test values; Prophet doesn't). The gap is small in practice, but
    worth knowing.
    """
    n_chunks = (len(test) - horizon) // horizon

    # Build the future dataframe spanning the entire test window
    future_index = test.index[:n_chunks * horizon]
    future_df = pd.DataFrame({"ds": future_index})
    forecast = model.predict(future_df)

    yhat = forecast["yhat"].values
    actual = test.iloc[:len(yhat)].values
    residuals = actual - yhat

    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    print(f"  {'Prophet':25s}  MAE = {mae:.3f} °C   RMSE = {rmse:.3f} °C")

    return {
        "model": "Prophet",
        "mae": mae,
        "rmse": rmse,
        "forecast": forecast,
        "actual": actual,
        "yhat": yhat,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_components(model: Prophet, forecast_df: pd.DataFrame, save_path: Path):
    """Prophet's built-in components plot: trend, weekly, yearly, daily."""
    fig = model.plot_components(forecast_df, figsize=(12, 10))
    fig.suptitle(
        "Prophet's view of Jena temperature: trend + seasonalities",
        y=1.01, fontsize=12, fontweight="bold",
    )
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_forecast_with_uncertainty(model: Prophet, forecast_df: pd.DataFrame,
                                   save_path: Path):
    """Prophet's built-in forecast plot with uncertainty bands."""
    fig = model.plot(forecast_df, figsize=(14, 5))
    plt.title("Prophet forecast with 80% prediction interval")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_changepoints(model: Prophet, forecast_df: pd.DataFrame, save_path: Path):
    """Trend with detected changepoints overlaid."""
    from prophet.plot import add_changepoints_to_plot
    fig = model.plot(forecast_df, figsize=(14, 5))
    add_changepoints_to_plot(fig.gca(), model, forecast_df)
    plt.title("Trend with detected changepoints (red dashed)")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_test_window_zoom(test: pd.Series, yhat: np.ndarray, lower: np.ndarray,
                          upper: np.ndarray, save_path: Path,
                          window_start: str = "2016-07-01",
                          window_end: str = "2016-07-08"):
    """Zoom into one week of test predictions vs actuals."""
    test_idx = test.index[:len(yhat)]
    df = pd.DataFrame({
        "actual": test.iloc[:len(yhat)].values,
        "yhat": yhat,
        "lower": lower,
        "upper": upper,
    }, index=test_idx)
    window = df.loc[window_start:window_end]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(window.index, window["lower"], window["upper"],
                    color="#5B9BD5", alpha=0.25, label="80% interval")
    ax.plot(window.index, window["yhat"], color="#5B9BD5",
            linewidth=2, label="Prophet forecast")
    ax.plot(window.index, window["actual"], color="#222222",
            linewidth=1.5, label="Actual")
    ax.set_title(f"Prophet forecast vs actual: {window_start} → {window_end}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def plot_leaderboard(results: list, save_path: Path):
    """Bar chart comparing model MAEs."""
    df = pd.DataFrame(results).sort_values("mae")
    colors = ["#C04A4A" if name == "Prophet" else "#888888"
              for name in df["model"]]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(df["model"], df["mae"], color=colors)
    ax.set_xlabel("MAE (°C) — lower is better")
    ax.set_title("Weekend 3 leaderboard: Prophet vs baselines (24h-ahead forecast)")
    for bar, val in zip(bars, df["mae"]):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Weekend 3: Prophet — first actual forecast")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load and split ----
    temp = load_hourly_temperature(DATA_PATH)
    train, val, test = chronological_split(temp)
    # Note: val is not used in Weekend 3. We're not tuning hyperparameters
    # this weekend; that's Weekend 8's topic. The val set is kept reserved
    # so we don't accidentally peek at 2015 data when tuning later.
    print(f"\nTrain:  {train.index.min()} → {train.index.max()}  ({len(train):,} rows)")
    print(f"Val:    {val.index.min()} → {val.index.max()}  ({len(val):,} rows)  [reserved for Weekend 8 tuning]")
    print(f"Test:   {test.index.min()} → {test.index.max()}  ({len(test):,} rows)")

    # ---- Baselines ----
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

    # ---- Prophet ----
    print("\n--- Prophet ---")
    train_df = prepare_for_prophet(train)
    model = fit_prophet(train_df)
    prophet_result = evaluate_prophet(model, train, test, FORECAST_HORIZON)
    results.append({
        "model": prophet_result["model"],
        "mae": prophet_result["mae"],
        "rmse": prophet_result["rmse"],
    })

    forecast_df = prophet_result["forecast"]

    # ---- Plots ----
    print("\n--- Plots ---")
    plot_components(model, forecast_df, OUTPUT_DIR / "01_components.png")
    plot_forecast_with_uncertainty(
        model, forecast_df, OUTPUT_DIR / "02_forecast_uncertainty.png",
    )
    plot_changepoints(model, forecast_df, OUTPUT_DIR / "03_changepoints.png")
    plot_test_window_zoom(
        test,
        prophet_result["yhat"],
        forecast_df["yhat_lower"].values,
        forecast_df["yhat_upper"].values,
        OUTPUT_DIR / "04_test_week_zoom.png",
    )
    plot_leaderboard(results, OUTPUT_DIR / "05_leaderboard.png")

    # ---- Save the leaderboard for later weekends ----
    leaderboard_path = Path("data/leaderboard.csv")
    leaderboard_path.parent.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(leaderboard_path, index=False)
    print(f"\n  saved leaderboard to {leaderboard_path}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Done. Plots in:", OUTPUT_DIR.resolve())
    print("\nFinal leaderboard (lower MAE is better):")
    for r in sorted(results, key=lambda x: x["mae"]):
        print(f"  {r['model']:25s}  MAE = {r['mae']:.3f} °C")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    main()