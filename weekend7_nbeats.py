"""
Weekend 7: N-BEATS — Deep Learning Built for Time Series

Picks up from Weekend 6 (LSTM tied SARIMA at MAE 1.870 vs 1.886). The
Weekend 6 cliffhanger promised N-BEATS — the neural basis expansion
architecture from Oreshkin et al. (2019), the first deep-learning approach
to top the M4 forecasting competition. The published M4 result used an
ensemble of N-BEATS models; here we use a single instance, with the same
generic-architecture hyperparameter family as the paper's reported
single-model configuration.

This script delivers what was promised:
- Univariate input (just temperature; no other weather variables)
- No engineered features
- Generic N-BEATS architecture (single-model; not an ensemble)
- Same train/val/test boundaries and leaderboard CSV as Weekends 3-6

Pipeline:
1. Load Jena, resample to hourly, keep ONLY the temperature column
2. Build a darts TimeSeries with the chronological split
3. Standardize using the training set (Scaler from darts)
4. Fit N-BEATS with PyTorch backend (MPS-accelerated on Apple Silicon)
5. Predict on 2016 in 24-hour chunks (matches the Weekend 6 protocol)
6. Compute MAE/RMSE in degrees Celsius, flatten across all 24 horizons
7. Update leaderboard

Run end-to-end:
    python weekend_7_nbeats.py

Outputs four plots to ./images/weekend_7/

Compute: ~10-20 minutes on Apple MPS, longer on CPU.

Dependencies:
    pip install u8darts[torch]
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
import torch

# darts is the time series forecasting library; we use its NBEATSModel.
# u8darts[torch] installs darts + PyTorch backend for GPU/MPS support.
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel
from pytorch_lightning.callbacks import EarlyStopping as EarlyStoppingCallback_PL

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("darts").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration — locked from the series spec
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Two distinct anchors:
#   DATA_ROOT   — where the Jena CSV lives. Found by walking UP from the script
#                 until we hit the directory containing data/jena_climate_*.csv.
#                 On this project that resolves to the timeseries_analysis/ root.
#   NOTEBOOK_ROOT — where THIS series writes its outputs (leaderboard + images).
#                 Prior weekends (3-6) all wrote to notebooks/, so we keep that
#                 as the home for Weekend 7's outputs too. This is the directory
#                 the script itself lives in.
#
# Keeping these separate is what prevents the split-folder problem: previously
# the leaderboard was pinned to notebooks/ while images derived from a project
# root that resolved one level higher, so they landed in different trees.
_data_marker = "data/jena_climate_2009_2016.csv"
DATA_ROOT = SCRIPT_DIR
while not (DATA_ROOT / _data_marker).exists() and DATA_ROOT.parent != DATA_ROOT:
    DATA_ROOT = DATA_ROOT.parent

NOTEBOOK_ROOT = SCRIPT_DIR  # outputs live alongside the script, under notebooks/

DATA_PATH = DATA_ROOT / "data" / "jena_climate_2009_2016.csv"
OUTPUT_DIR = NOTEBOOK_ROOT / "images" / "weekend_7"
LEADERBOARD_PATH = NOTEBOOK_ROOT / "data" / "leaderboard.csv"
RANDOM_SEED = 42

TRAIN_END = "2014-12-31 23:00:00"
VAL_END = "2015-12-31 23:00:00"
TEST_END = "2016-12-31 23:00:00"

INPUT_WINDOW = 168       # 7 days of hourly history — matches Weekend 6 LSTM
OUTPUT_HORIZON = 24      # predict the next 24 hours
TARGET_COL = "T (degC)"

# N-BEATS hyperparameters — same family as the paper's generic-architecture
# single-model configuration. The M4 submission was an ensemble; this is one
# model.
NUM_STACKS = 30          # default in the paper's generic architecture
NUM_BLOCKS = 1           # blocks per stack (generic uses 1; interpretable uses more)
NUM_LAYERS = 4           # fully-connected layers within each block
LAYER_WIDTH = 512        # neurons per FC layer (paper default)
BATCH_SIZE = 256
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 1e-3

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
torch.manual_seed(RANDOM_SEED)


def get_pl_trainer_kwargs():
    """Configure PyTorch Lightning's trainer to use MPS / CUDA / CPU.

    darts wraps PyTorch Lightning; we pass accelerator settings through.
    """
    if torch.backends.mps.is_available():
        return {"accelerator": "mps", "devices": 1}
    if torch.cuda.is_available():
        return {"accelerator": "gpu", "devices": 1}
    return {"accelerator": "cpu", "devices": 1}


# ---------------------------------------------------------------------------
# Step 1: Load univariate temperature data
# ---------------------------------------------------------------------------
def load_hourly_temperature(path: Path) -> pd.Series:
    """Load Jena CSV, resample to hourly, keep only the target column."""
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
    hourly = hourly.asfreq("h").interpolate("linear")
    return hourly[TARGET_COL].dropna()


def chronological_split(series: pd.Series):
    """Split the temperature series by date (same boundaries as Weekends 3-6)."""
    train = series.loc[:TRAIN_END]
    val = series.loc[TRAIN_END:VAL_END].iloc[1:]
    test = series.loc[VAL_END:TEST_END].iloc[1:]
    return train, val, test


def to_darts(series: pd.Series) -> TimeSeries:
    """Convert a pandas Series to a darts TimeSeries.

    darts expects an explicit frequency. We already resampled to hourly,
    so 'h' is correct.
    """
    return TimeSeries.from_series(series, freq="h").astype(np.float32)


# ---------------------------------------------------------------------------
# Step 2: Build N-BEATS model
# ---------------------------------------------------------------------------
class EarlyStoppingCallback(EarlyStoppingCallback_PL):
    """Early stopping on validation loss with the configured patience.

    Subclasses PyTorch Lightning's EarlyStopping with our defaults so the
    NBEATSModel constructor stays uncluttered.
    """
    def __init__(self):
        super().__init__(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=1e-5,
            mode="min",
            verbose=True,
        )


def build_nbeats() -> NBEATSModel:
    """Construct the generic N-BEATS architecture.

    Why these settings:
    - generic_architecture=True: blocks are stacks of fully-connected layers
      with learned basis functions, not the trend+seasonality blocks of the
      'interpretable' variant. Generic was the higher-accuracy variant in
      Oreshkin et al. (2019) Table 18.
    - num_stacks=30: matches the paper's generic config
    - num_blocks=1 per stack: paper's generic uses 1 block per stack
      (interpretable uses 3)
    - layer_widths=512: paper default
    - input_chunk_length=168, output_chunk_length=24: same as LSTM in Weekend 6
    """
    early_stopping = EarlyStoppingCallback()
    pl_kwargs = get_pl_trainer_kwargs()
    pl_kwargs["callbacks"] = [early_stopping]

    model = NBEATSModel(
        input_chunk_length=INPUT_WINDOW,
        output_chunk_length=OUTPUT_HORIZON,
        generic_architecture=True,
        num_stacks=NUM_STACKS,
        num_blocks=NUM_BLOCKS,
        num_layers=NUM_LAYERS,
        layer_widths=LAYER_WIDTH,
        batch_size=BATCH_SIZE,
        n_epochs=MAX_EPOCHS,
        random_state=RANDOM_SEED,
        optimizer_kwargs={"lr": LEARNING_RATE},
        pl_trainer_kwargs=pl_kwargs,
        save_checkpoints=True,   # required for load_best=True after early stopping
        force_reset=True,        # wipe any pre-existing checkpoint dir
        model_name="weekend_7_nbeats",
    )
    return model


# ---------------------------------------------------------------------------
# Step 3: Walk-forward evaluation
# ---------------------------------------------------------------------------
def evaluate_nbeats(model: NBEATSModel, train_scaled: TimeSeries,
                    val_scaled: TimeSeries, test_scaled: TimeSeries,
                    scaler: Scaler) -> dict:
    """Generate rolling 24-hour-ahead predictions matching Weekend 6's protocol.

    Origin convention (matches Weekend 6 LSTM exactly):
    - The LSTM's WeatherSequenceDataset built sequences entirely from rows
      INSIDE the test split: input window [t, t+167] -> target [t+168, t+191],
      with t ranging over rows of test_scaled. That gave 8,593 sequences and
      8,593 24-hour forecasts.
    - For apples-to-apples comparison, we start N-BEATS forecasts at
      `test_start + INPUT_WINDOW` (so the 168-hour input window lives entirely
      inside test) and predict the next 24 hours. That gives the same
      8,593 origins.

    Why we use historical_forecasts:
    A naive Python loop calling model.predict() 8,593 times has crippling
    per-call overhead in darts (input formatting, MPS transfer, tensor
    extraction). historical_forecasts() batches the inputs through the
    model once per stride and is the idiomatic darts way to do walk-forward
    evaluation.
    """
    print(f"  generating rolling 24h-ahead forecasts via historical_forecasts...")

    # Concatenate all splits so the historical_forecasts call has full context.
    # darts uses the input_chunk_length window from the supplied series to make
    # each forecast — and it picks the appropriate window automatically from
    # the start position we specify.
    full_series = train_scaled.concatenate(val_scaled).concatenate(test_scaled)
    test_start_idx = len(train_scaled) + len(val_scaled)

    # First origin we forecast: test_start_idx + INPUT_WINDOW. Matches Weekend 6.
    # historical_forecasts with start=<timestamp> starts predicting from that
    # timestamp onwards.
    first_origin_idx = test_start_idx + INPUT_WINDOW
    first_origin_ts = full_series.time_index[first_origin_idx]

    # forecast_horizon=24, stride=1, last_points_only=False returns a list of
    # 24-element TimeSeries — one per origin. Exactly what we need.
    forecasts = model.historical_forecasts(
        series=full_series,
        start=first_origin_ts,
        forecast_horizon=OUTPUT_HORIZON,
        stride=1,
        retrain=False,
        last_points_only=False,
        verbose=False,
    )

    # Each element of `forecasts` is a TimeSeries of length 24, scaled.
    # Inverse-transform and stack into (n_origins, 24).
    y_pred_scaled = np.stack([f.values().flatten() for f in forecasts])
    # Recover the actuals at the same origins/horizons from the unscaled series.
    # We need the inverse-scaled actuals to compute MAE/RMSE in °C.
    n_origins = len(forecasts)

    # Pull actuals in °C from the full (inverse-scaled) series
    full_unscaled_values = scaler.inverse_transform(full_series).values().flatten()

    y_true = np.zeros((n_origins, OUTPUT_HORIZON), dtype=np.float32)
    for i, fc in enumerate(forecasts):
        # Each forecast i starts at first_origin_idx + i and runs for 24 steps
        start = first_origin_idx + i
        y_true[i, :] = full_unscaled_values[start:start + OUTPUT_HORIZON]

    # Inverse-transform the predictions in one batch
    y_pred = np.zeros_like(y_true)
    for i, fc in enumerate(forecasts):
        y_pred[i, :] = scaler.inverse_transform(fc).values().flatten()

    residuals = (y_true - y_pred).flatten()
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    print(f"  {'N-BEATS':25s}  MAE = {mae:.3f} °C   RMSE = {rmse:.3f} °C")
    print(f"  Forecast origins evaluated: {n_origins:,} "
          f"(matches Weekend 6 LSTM convention)")
    return {
        "model": "N-BEATS",
        "mae": mae,
        "rmse": rmse,
        "y_true_deg": y_true,
        "y_pred_deg": y_pred,
        "residuals": residuals,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_test_week_zoom(test_series: pd.Series, y_true_deg: np.ndarray,
                        y_pred_deg: np.ndarray, save_path: Path,
                        window_start: str = "2016-07-01",
                        window_end: str = "2016-07-08"):
    """Show true 24-hour-ahead predictions for one summer week.

    Same convention as Weekend 6: pull the LAST column of each 24-step
    forecast and align the x-axis to the timestamp of that 24th-hour value.

    Origin indexing:
    Forecast origins in this script start at INPUT_WINDOW within the test
    series (matching Weekend 6 LSTM). So sample i predicts hours
    [INPUT_WINDOW + i, INPUT_WINDOW + i + 23] of test_series. The 24th-hour
    target lives at index INPUT_WINDOW + i + (OUTPUT_HORIZON - 1).
    """
    h = OUTPUT_HORIZON - 1
    sample_indices = np.arange(len(y_true_deg))
    target_indices = sample_indices + INPUT_WINDOW + h
    timestamps = test_series.index[target_indices]

    actual_h = y_true_deg[:, h]
    pred_h = y_pred_deg[:, h]

    df_plot = pd.DataFrame(
        {"actual": actual_h, "nbeats": pred_h},
        index=timestamps,
    )
    window = df_plot.loc[window_start:window_end]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(window.index, window["actual"], color="#222222",
            linewidth=1.5, label="Actual")
    ax.plot(window.index, window["nbeats"], color="#D97757",
            linewidth=2, label="N-BEATS forecast (24h-ahead)", alpha=0.9)
    ax.set_title(f"N-BEATS true 24-hour-ahead forecast vs actual: "
                 f"{window_start} → {window_end}")
    ax.set_xlabel("Date (target time, i.e. the date being predicted)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def plot_residual_diagnostics(residuals: np.ndarray, save_path: Path,
                              horizon_label: str = "24h-ahead"):
    """4-panel residual diagnostic for a SINGLE forecast horizon."""
    from scipy import stats as scipy_stats
    from statsmodels.graphics.tsaplots import plot_acf

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(residuals, linewidth=0.4, color="#D97757")
    axes[0, 0].axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    axes[0, 0].set_title(f"Residuals over time ({horizon_label})")
    axes[0, 0].set_xlabel("Forecast index")
    axes[0, 0].set_ylabel("Residual (°C)")

    axes[0, 1].hist(residuals, bins=60, density=True, color="#5B9BD5",
                    edgecolor="white", alpha=0.85)
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 200)
    axes[0, 1].plot(x, scipy_stats.norm.pdf(x, mu, sigma),
                    color="#C04A4A", linewidth=2, label="Normal fit")
    axes[0, 1].set_title(f"Residual distribution ({horizon_label})")
    axes[0, 1].set_xlabel("Residual (°C)")
    axes[0, 1].legend()

    scipy_stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q plot (normality check)")

    plot_acf(residuals, lags=48, ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title(f"ACF of {horizon_label} residuals (first 48 lags)")
    axes[1, 1].set_xlabel("Lag (hours of forecast-origin time)")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def plot_mae_by_horizon(y_true_deg: np.ndarray, y_pred_deg: np.ndarray,
                        save_path: Path):
    """MAE at each forecast horizon, with the Weekend 6 LSTM curve as comparison.

    Like Weekend 6's plot_mae_by_horizon, but if we have the LSTM's per-horizon
    CSV from Weekend 6, we overlay it for direct comparison.
    """
    mae_per_h = np.mean(np.abs(y_true_deg - y_pred_deg), axis=0)
    horizons = np.arange(1, OUTPUT_HORIZON + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(horizons, mae_per_h, marker="o", color="#D97757",
            linewidth=2, markersize=7, label="N-BEATS")

    # Overlay Weekend 6's LSTM per-horizon profile if available.
    # The LSTM CSV location moved across weekends, so check the likely spots:
    # NOTEBOOK_ROOT/images/weekend_6 (current convention) and a couple of
    # fallbacks. First hit wins.
    lstm_candidates = [
        NOTEBOOK_ROOT / "images" / "weekend_6" / "lstm_horizon_metrics.csv",
        DATA_ROOT / "images" / "weekend_6" / "lstm_horizon_metrics.csv",
        NOTEBOOK_ROOT.parent / "images" / "weekend_6" / "lstm_horizon_metrics.csv",
    ]
    lstm_csv = next((p for p in lstm_candidates if p.exists()), None)
    if lstm_csv is not None:
        lstm_metrics = pd.read_csv(lstm_csv)
        ax.plot(lstm_metrics["horizon"], lstm_metrics["mae"],
                marker="s", color="#7B4F8B", linewidth=2, markersize=6,
                alpha=0.8, label="LSTM (Weekend 6)")
        print(f"  overlaid LSTM profile from {lstm_csv}")
    else:
        print("  NOTE: LSTM horizon CSV not found — plotting N-BEATS only. "
              "(Looked under images/weekend_6/lstm_horizon_metrics.csv)")

    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel("MAE (°C)")
    ax.set_title("Per-horizon error: N-BEATS vs LSTM")
    ax.set_xticks(horizons)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend()
    ax.annotate(f"h=1: {mae_per_h[0]:.3f}°C",
                xy=(1, mae_per_h[0]), xytext=(2.5, mae_per_h[0] - 0.15),
                fontsize=9, color="#444444")
    ax.annotate(f"h=24: {mae_per_h[-1]:.3f}°C",
                xy=(24, mae_per_h[-1]), xytext=(19, mae_per_h[-1] + 0.05),
                fontsize=9, color="#444444")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def plot_leaderboard(results: list, save_path: Path):
    """Bar chart showing the full leaderboard across 7 weekends."""
    df = pd.DataFrame(results).sort_values("mae")
    color_map = {
        "N-BEATS": "#D97757",
        "LSTM": "#7B4F8B",
        "XGBoost": "#FF6B35",
        "LightGBM": "#1E8E5A",
        "SARIMA": "#2E8B57",
        "Prophet": "#C04A4A",
    }
    colors = [color_map.get(name, "#888888") for name in df["model"]]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(df["model"], df["mae"], color=colors)
    ax.set_xlabel("MAE (°C) — lower is better")
    ax.set_title("Weekend 7 leaderboard: N-BEATS joins the lineup "
                 "(MAE averaged across forecast horizons h=1..24)")
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
    print("Weekend 7: N-BEATS (univariate, generic architecture)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pl_kwargs = get_pl_trainer_kwargs()
    print(f"Device: {pl_kwargs['accelerator']}")

    # ---- Load + split ----
    temp = load_hourly_temperature(DATA_PATH)
    print(f"\nLoaded {len(temp):,} hourly temperature values (univariate)")
    train, val, test = chronological_split(temp)
    print(f"Train:  {train.index.min()} → {train.index.max()}  ({len(train):,} rows)")
    print(f"Val:    {val.index.min()} → {val.index.max()}  "
          f"({len(val):,} rows)  [used for early stopping; no hyperparameter tuning]")
    print(f"Test:   {test.index.min()} → {test.index.max()}  ({len(test):,} rows)")

    # ---- Convert to darts TimeSeries and scale ----
    print("\n--- Standardization (train-only stats) ---")
    train_ts = to_darts(train)
    val_ts = to_darts(val)
    test_ts = to_darts(test)

    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_ts)
    val_scaled = scaler.transform(val_ts)
    test_scaled = scaler.transform(test_ts)
    print(f"  scaler fitted on train ({len(train):,} rows)")

    # ---- Build and train N-BEATS ----
    print("\n--- N-BEATS training ---")
    model = build_nbeats()
    print(f"  Architecture: generic N-BEATS, {NUM_STACKS} stacks × "
          f"{NUM_BLOCKS} blocks × {NUM_LAYERS} layers × {LAYER_WIDTH} width")
    # Actual parameter count will be printed after the first fit() call,
    # since darts builds the internal model lazily on first fit. We'll print
    # it from model.model.parameters() after training completes.
    print(f"  Training on {pl_kwargs['accelerator']} for up to {MAX_EPOCHS} epochs "
          f"(early stopping patience={EARLY_STOPPING_PATIENCE})")
    model.fit(series=train_scaled, val_series=val_scaled, verbose=False)

    # Restore best-validation weights (matches Weekend 6 LSTM behavior).
    #
    # PyTorch 2.6 changed torch.load's default to weights_only=True, which
    # rejects darts checkpoints because they pickle the optimizer state
    # (torch.optim.adam.Adam). We allowlist the trusted globals the darts
    # checkpoint needs, then reload. If the reload still fails for any reason,
    # we fall back to the in-memory model (last-epoch weights) with a warning,
    # so the run completes either way.
    best_restored = False
    try:
        import torch.serialization as _ts
        from torch.optim.adam import Adam as _Adam
        _safe = [_Adam]
        # pytorch_lightning bundles a couple of helper classes in the ckpt too;
        # add them if present in this version.
        try:
            from pytorch_lightning.utilities.types import OptimizerConfig as _OptCfg
            _safe.append(_OptCfg)
        except Exception:
            pass
        with _ts.safe_globals(_safe):
            model = NBEATSModel.load_from_checkpoint("weekend_7_nbeats", best=True)
        best_restored = True
        print("  training complete; best-validation checkpoint restored")
    except Exception as e_safe:
        # Second attempt: the checkpoint is OUR OWN file, written seconds ago by
        # this same script — it is trusted. Temporarily force torch.load back to
        # weights_only=False (the pre-2.6 behavior) just for this reload.
        try:
            import torch as _torch
            _orig_load = _torch.load
            def _trusting_load(*args, **kwargs):
                kwargs["weights_only"] = False
                return _orig_load(*args, **kwargs)
            _torch.load = _trusting_load
            try:
                model = NBEATSModel.load_from_checkpoint("weekend_7_nbeats", best=True)
                best_restored = True
                print("  training complete; best-validation checkpoint restored "
                      "(via trusted-load fallback)")
            finally:
                _torch.load = _orig_load
        except Exception as e_force:
            print(f"  WARNING: could not reload best checkpoint "
                  f"({type(e_safe).__name__} then {type(e_force).__name__}). "
                  f"Using last-epoch in-memory model instead.")
            print(f"           (detail: {str(e_force)[:120]})")

    if not best_restored:
        print("  NOTE: results below reflect the final training epoch, not the "
              "best-validation epoch. With early stopping patience="
              f"{EARLY_STOPPING_PATIENCE}, these are close but not identical.")

    # Print actual parameter count (the formula-based estimate was rough)
    try:
        n_params = sum(p.numel() for p in model.model.parameters())
        print(f"  Total trainable parameters: {n_params:,}")
    except Exception:
        # darts internals occasionally change; not worth crashing the script over
        pass

    # ---- Evaluate on test set ----
    print("\n--- Test evaluation ---")
    nbeats_result = evaluate_nbeats(model, train_scaled, val_scaled,
                                    test_scaled, scaler)

    # ---- Build leaderboard ----
    print("\n--- Leaderboard ---")
    results = []
    if LEADERBOARD_PATH.exists():
        prior_lb = pd.read_csv(LEADERBOARD_PATH)
        prior_lb = prior_lb[~prior_lb["model"].isin(["N-BEATS"])]
        for _, row in prior_lb.iterrows():
            results.append({"model": row["model"],
                            "mae": float(row["mae"]),
                            "rmse": float(row["rmse"])})
            print(f"  loaded prior: {row['model']:25s}  "
                  f"MAE = {row['mae']:.3f} °C")
    results.append({"model": "N-BEATS",
                    "mae": nbeats_result["mae"],
                    "rmse": nbeats_result["rmse"]})

    # ---- Per-horizon metrics for auditability ----
    y_true_deg = nbeats_result["y_true_deg"]
    y_pred_deg = nbeats_result["y_pred_deg"]
    horizon_metrics = pd.DataFrame({
        "horizon": np.arange(1, OUTPUT_HORIZON + 1),
        "mae": np.mean(np.abs(y_true_deg - y_pred_deg), axis=0),
        "rmse": np.sqrt(np.mean((y_true_deg - y_pred_deg) ** 2, axis=0)),
    })
    horizon_csv_path = OUTPUT_DIR / "nbeats_horizon_metrics.csv"
    horizon_metrics.to_csv(horizon_csv_path, index=False)
    print(f"\n  saved per-horizon metrics: {horizon_csv_path}")
    print(f"  MAE by horizon (first / middle / last):")
    print(f"    h= 1: MAE={horizon_metrics.iloc[0]['mae']:.3f} °C")
    print(f"    h=12: MAE={horizon_metrics.iloc[11]['mae']:.3f} °C")
    print(f"    h=24: MAE={horizon_metrics.iloc[23]['mae']:.3f} °C")

    # ---- Horizon-24-only residuals for diagnostic plot ----
    h_last = OUTPUT_HORIZON - 1
    residuals_h24 = y_true_deg[:, h_last] - y_pred_deg[:, h_last]

    # ---- Plots ----
    print("\n--- Plots ---")
    plot_test_week_zoom(test, y_true_deg, y_pred_deg,
                        OUTPUT_DIR / "01_test_week_zoom.png")
    plot_residual_diagnostics(residuals_h24,
                              OUTPUT_DIR / "02_residual_diagnostics_h24.png",
                              horizon_label="24h-ahead")
    plot_mae_by_horizon(y_true_deg, y_pred_deg,
                        OUTPUT_DIR / "03_mae_by_horizon_vs_lstm.png")
    plot_leaderboard(results, OUTPUT_DIR / "04_leaderboard.png")

    # ---- Save updated leaderboard ----
    pd.DataFrame(results).to_csv(LEADERBOARD_PATH, index=False)
    print(f"\n  updated leaderboard at {LEADERBOARD_PATH}")

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