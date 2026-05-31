"""
Weekend 6: Deep Learning Joins the Lineup — LSTM on Raw Sequences

Picks up from Weekend 5 (XGBoost 2.385 / LightGBM 2.388, both lost to SARIMA).
Weekend 5's cliffhanger promised:
- LSTM architecture
- Raw multivariate sequences (no hand-engineered features)
- The model sees data "the way a human observer would" - as a sequence
  unfolding over time, not as a flat row of pre-engineered features

This script delivers all three.

Pipeline:
1. Load Jena, resample to hourly, keep all 14 features
2. Standardize using training-set statistics only (no leakage)
3. Build sequences: 168 hours of history -> next 24 hours of temperature
4. Train a 2-layer LSTM with single-shot 24-output head
5. Evaluate on 2016 test set, compute MAE/RMSE
6. Diagnose residuals
7. Update leaderboard

Run end-to-end:
    python weekend_6_lstm.py

Outputs five plots plus lstm_horizon_metrics.csv to ./images/weekend_6/

Compute: ~10-30 minutes on CPU, ~3-10 minutes on Apple MPS or CUDA.
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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration — locked from the series spec
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
while not (PROJECT_ROOT / "data").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_PATH = Path("/Users/rumasinha/random/timeseries_analysis/data/jena_climate_2009_2016.csv")
OUTPUT_DIR = PROJECT_ROOT / "images" / "weekend_6"
LEADERBOARD_PATH = PROJECT_ROOT / "data" / "leaderboard.csv"
RANDOM_SEED = 42

TRAIN_END = "2014-12-31 23:00:00"
VAL_END = "2015-12-31 23:00:00"
TEST_END = "2016-12-31 23:00:00"

INPUT_WINDOW = 168       # 7 days of hourly history as input
OUTPUT_HORIZON = 24      # predict the next 24 hours
TARGET_COL = "T (degC)"

# LSTM hyperparameters - kept simple, will be tuned in Weekend 8
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7

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


def get_device() -> torch.device:
    """Pick the fastest available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Step 1: Load multivariate hourly data
# ---------------------------------------------------------------------------
def load_hourly_multivariate(path: Path) -> pd.DataFrame:
    """Load Jena CSV, keep all 14 weather variables, resample hourly."""
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
    return hourly


def chronological_split(df: pd.DataFrame):
    """Split the raw dataframe by date (same boundaries as Weekends 3-5)."""
    train = df.loc[:TRAIN_END]
    val = df.loc[TRAIN_END:VAL_END].iloc[1:]
    test = df.loc[VAL_END:TEST_END].iloc[1:]
    return train, val, test


# ---------------------------------------------------------------------------
# Step 2: Standardization (training statistics ONLY)
# ---------------------------------------------------------------------------
class Standardizer:
    """Z-score normalization using training-set means and stds.

    Why this matters for LSTMs: neural networks train poorly when input
    features span different scales (pressure ~1000 mbar vs wind speed
    ~5 m/s). Standardization puts everything on a common scale so the
    optimizer can learn balanced weights.

    Critical correctness rule: stats are computed on TRAIN only. Using
    val or test statistics is leakage.
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, df: pd.DataFrame):
        self.mean_ = df.mean()
        self.std_ = df.std()
        # Avoid divide-by-zero on near-constant columns
        self.std_ = self.std_.replace(0, 1e-8)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df - self.mean_) / self.std_

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        """Reverse the z-score for just the target column."""
        return values * self.std_[TARGET_COL] + self.mean_[TARGET_COL]


# ---------------------------------------------------------------------------
# Step 3: Sequence dataset
# ---------------------------------------------------------------------------
class WeatherSequenceDataset(Dataset):
    """Turn a standardized DataFrame into (input_window, target_window) pairs.

    For each starting index i:
      input  = all 14 features at times [i, i+1, ..., i+167]   shape (168, 14)
      target = temperature at times [i+168, i+169, ..., i+191] shape (24,)

    The LSTM sees the input_window as a sequence and emits the target_window
    as a single 24-element vector (single-shot prediction). No autoregression
    during inference — that's a different post.
    """
    def __init__(self, df: pd.DataFrame, target_col: str = TARGET_COL,
                 input_window: int = INPUT_WINDOW,
                 output_horizon: int = OUTPUT_HORIZON):
        self.df = df.copy()
        self.target_col = target_col
        self.input_window = input_window
        self.output_horizon = output_horizon
        self.X = self.df.values.astype(np.float32)
        # Index of the target column in the (n_rows, n_features) array
        self.target_idx = list(self.df.columns).index(target_col)
        # Valid starting indices i: need at least input_window + output_horizon
        # rows of history available from i forward
        self.n = len(self.df) - self.input_window - self.output_horizon + 1
        if self.n <= 0:
            raise ValueError(
                f"DataFrame too short: needs at least "
                f"{self.input_window + self.output_horizon} rows, has {len(self.df)}"
            )

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.X[idx:idx + self.input_window]                 # (168, 14)
        y_start = idx + self.input_window
        y_end = y_start + self.output_horizon
        y = self.X[y_start:y_end, self.target_idx]              # (24,)
        return torch.from_numpy(x), torch.from_numpy(y)


# ---------------------------------------------------------------------------
# Step 4: LSTM model
# ---------------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    """Two-layer LSTM with a single-shot prediction head.

    Architecture:
    - Input: (batch, 168, 14) sequence of 14-dim hourly observations
    - LSTM: 2 stacked layers, 64 hidden units each, dropout 0.2
    - Take only the LAST hidden state of the top layer: (batch, 64)
    - Dense layer maps (batch, 64) -> (batch, 24): all 24 forecasts at once

    Why use only the last hidden state: by the time the LSTM has read all
    168 input steps, the final hidden state should summarize everything it
    needs to know about the recent past. Predicting all 24 future steps
    from this single vector is the simplest viable architecture.
    """
    def __init__(self, n_features: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT,
                 output_horizon: int = OUTPUT_HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, output_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Take the last hidden state of the top layer: (batch, hidden_size)
        final_hidden = lstm_out[:, -1, :]
        # Predict all output_horizon steps at once
        return self.head(final_hidden)


# ---------------------------------------------------------------------------
# Step 5: Training loop with early stopping
# ---------------------------------------------------------------------------
def train_lstm(model: nn.Module, train_loader: DataLoader,
               val_loader: DataLoader, device: torch.device,
               max_epochs: int = MAX_EPOCHS,
               patience: int = EARLY_STOPPING_PATIENCE,
               lr: float = LEARNING_RATE) -> dict:
    """Train with Adam, MSE loss, early stopping on validation loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = None
    best_state = None
    epochs_since_improvement = 0

    print(f"  Training on {device} for up to {max_epochs} epochs "
          f"(early stopping patience={patience})")

    for epoch in range(1, max_epochs + 1):
        # ----- Train -----
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            yhat = model(x_batch)
            loss = criterion(yhat, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ----- Validate -----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                yhat = model(x_batch)
                val_losses.append(criterion(yhat, y_batch).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"    epoch {epoch:3d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}")

        # ----- Early stopping -----
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            # Save the best state on CPU to avoid GPU memory issues
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"    early stopping at epoch {epoch} "
                      f"(no val improvement for {patience} epochs)")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  restored best weights from epoch {best_epoch} "
              f"(val_loss={best_val_loss:.4f})")

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    return history


# ---------------------------------------------------------------------------
# Step 6: Evaluation on test set
# ---------------------------------------------------------------------------
def evaluate_on_test(model: nn.Module, test_loader: DataLoader,
                     standardizer: Standardizer,
                     device: torch.device) -> dict:
    """Run model on test set, inverse-transform back to °C, compute MAE/RMSE.

    The LSTM outputs standardized values; we have to undo the z-score to get
    back to degrees Celsius for an apples-to-apples MAE comparison with the
    other models in the leaderboard.
    """
    model.eval()
    all_y_true = []
    all_y_pred = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            yhat = model(x_batch)
            all_y_true.append(y_batch.numpy())
            all_y_pred.append(yhat.cpu().numpy())

    y_true = np.concatenate(all_y_true, axis=0)  # (n_samples, 24)
    y_pred = np.concatenate(all_y_pred, axis=0)  # (n_samples, 24)

    # Inverse-transform from z-score back to °C
    y_true_deg = standardizer.inverse_transform_target(y_true)
    y_pred_deg = standardizer.inverse_transform_target(y_pred)

    # MAE/RMSE flatten across all (sample, horizon) pairs — this matches how
    # the other models in the series are evaluated
    residuals = (y_true_deg - y_pred_deg).flatten()
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    print(f"  {'LSTM':25s}  MAE = {mae:.3f} °C   RMSE = {rmse:.3f} °C")
    return {
        "model": "LSTM",
        "mae": mae,
        "rmse": rmse,
        "y_true_deg": y_true_deg,
        "y_pred_deg": y_pred_deg,
        "residuals": residuals,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_training_curves(history: dict, save_path: Path):
    """Train loss vs val loss over epochs — diagnose overfitting at a glance.

    Also marks the best-validation epoch (the epoch whose weights are
    actually being used in the final model). Without this annotation, a
    reader might assume the final model uses epoch-10 weights when in
    fact early stopping rolled back to the best epoch.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train loss",
            color="#2E5077", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Validation loss",
            color="#C04A4A", linewidth=2)

    # Mark the best-validation epoch — this is the model that ends up used
    best_epoch = history.get("best_epoch")
    best_val_loss = history.get("best_val_loss")
    if best_epoch is not None:
        ax.axvline(best_epoch, linestyle="--", color="gray", alpha=0.7)
        # Place label just above the best val loss point
        ax.scatter([best_epoch], [best_val_loss], color="gray",
                   zorder=5, s=80, edgecolor="black", linewidth=1.2)
        ax.annotate(
            f"best epoch ({best_epoch}):\nrestored model",
            xy=(best_epoch, best_val_loss),
            xytext=(best_epoch + 1.2, best_val_loss + 0.005),
            fontsize=9, color="#222222",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (standardized space)")
    ax.set_title("LSTM training curves — convergence and overfitting diagnostic")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def plot_test_week_zoom(test_df: pd.DataFrame, y_true_deg: np.ndarray,
                        y_pred_deg: np.ndarray, save_path: Path,
                        window_start: str = "2016-07-01",
                        window_end: str = "2016-07-08"):
    """Plot the TRUE 24-hour-ahead predictions for one summer week.

    test_df has the dates; y_true_deg and y_pred_deg are arrays of shape
    (n_samples, 24) where row i predicts the 24 hours starting at
    test_df.index[i + INPUT_WINDOW].

    To show what the model knows about "the value 24 hours from now,"
    we use the LAST column of each forecast (horizon h=23, i.e. hour 24
    of the prediction). The x-axis date is then the timestamp of that
    24th-hour value, which is the date the prediction is FOR.

    Why this matters: plotting y_pred_deg[:, 0] would show 1-hour-ahead
    predictions, not 24-hour-ahead, despite the chart title saying
    otherwise. That was a real bug in v1 of this script.
    """
    h = OUTPUT_HORIZON - 1  # the 24th forecast hour (zero-indexed)
    sample_indices = np.arange(len(y_true_deg))
    # The 24th-hour-ahead target for sample i lives at index i + 168 + 23
    target_indices = sample_indices + INPUT_WINDOW + h
    timestamps = test_df.index[target_indices]

    actual_h = y_true_deg[:, h]
    pred_h = y_pred_deg[:, h]

    df_plot = pd.DataFrame(
        {"actual": actual_h, "lstm": pred_h},
        index=timestamps,
    )
    window = df_plot.loc[window_start:window_end]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(window.index, window["actual"], color="#222222",
            linewidth=1.5, label="Actual")
    ax.plot(window.index, window["lstm"], color="#7B4F8B",
            linewidth=2, label="LSTM forecast (24h-ahead)", alpha=0.9)
    ax.set_title(f"LSTM true 24-hour-ahead forecast vs actual: "
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
    """4-panel residual diagnostic for a SINGLE forecast horizon.

    Important: the residuals passed in should be from a single horizon h
    (e.g., all the 24-hour-ahead errors), NOT a flattened mix of all
    horizons. Flattening overlapping multi-horizon forecast windows
    creates artificial structure in the ACF — the lag axis stops being
    interpretable as 'hours of real time' and starts encoding the
    flattening order. Same issue Weekend 4 flagged for Ljung-Box on
    rolling-forecast residuals: the right diagnostic depends on the
    structure of the residuals, not just the model.
    """
    from scipy import stats as scipy_stats
    from statsmodels.graphics.tsaplots import plot_acf

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(residuals, linewidth=0.4, color="#7B4F8B")
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
    """MAE at each forecast horizon (1-24 hours ahead).

    This plot reveals WHERE the LSTM gets its accuracy from. If MAE is
    low at horizon 1 and grows monotonically with h, the model is good
    at short-range and degrades for far-ahead predictions. If MAE is
    flat across horizons, the model isn't using the sequence structure
    to advantage at any particular range.
    """
    mae_per_h = np.mean(np.abs(y_true_deg - y_pred_deg), axis=0)
    horizons = np.arange(1, OUTPUT_HORIZON + 1)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(horizons, mae_per_h, marker="o", color="#7B4F8B",
            linewidth=2, markersize=7)
    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel("MAE (°C)")
    ax.set_title("LSTM error grows with forecast horizon")
    ax.set_xticks(horizons)
    ax.tick_params(axis="x", labelsize=8)

    # Annotate the first and last horizons for quick reading
    ax.annotate(f"h=1: {mae_per_h[0]:.3f}°C",
                xy=(1, mae_per_h[0]), xytext=(2.5, mae_per_h[0] - 0.1),
                fontsize=9, color="#444444")
    ax.annotate(f"h=24: {mae_per_h[-1]:.3f}°C",
                xy=(24, mae_per_h[-1]), xytext=(20, mae_per_h[-1] + 0.05),
                fontsize=9, color="#444444")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def plot_leaderboard(results: list, save_path: Path):
    """Bar chart showing the full leaderboard across 6 weekends."""
    df = pd.DataFrame(results).sort_values("mae")
    color_map = {
        "LSTM": "#7B4F8B",
        "XGBoost": "#FF6B35",
        "LightGBM": "#1E8E5A",
        "SARIMA": "#2E8B57",
        "Prophet": "#C04A4A",
    }
    colors = [color_map.get(name, "#888888") for name in df["model"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(df["model"], df["mae"], color=colors)
    ax.set_xlabel("MAE (°C) — lower is better")
    ax.set_title("Weekend 6 leaderboard: LSTM joins the lineup "
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
    print("Weekend 6: LSTM on raw multivariate sequences")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # ---- Load + split ----
    df = load_hourly_multivariate(DATA_PATH)
    print(f"\nLoaded {len(df):,} hourly rows × {df.shape[1]} variables")
    train_df, val_df, test_df = chronological_split(df)
    print(f"Train:  {train_df.index.min()} → {train_df.index.max()}  "
          f"({len(train_df):,} rows)")
    print(f"Val:    {val_df.index.min()} → {val_df.index.max()}  "
          f"({len(val_df):,} rows)  [used for early stopping; no hyperparameter tuning]")
    print(f"Test:   {test_df.index.min()} → {test_df.index.max()}  "
          f"({len(test_df):,} rows)")

    # ---- Standardize using TRAIN stats only ----
    print("\n--- Standardization (train-only stats) ---")
    standardizer = Standardizer().fit(train_df)
    train_std = standardizer.transform(train_df)
    val_std = standardizer.transform(val_df)
    test_std = standardizer.transform(test_df)
    print(f"  target mean (train): {standardizer.mean_[TARGET_COL]:.3f} °C")
    print(f"  target std  (train): {standardizer.std_[TARGET_COL]:.3f} °C")

    # ---- Build sequence datasets ----
    print("\n--- Sequence datasets (168h input -> 24h output) ---")
    train_ds = WeatherSequenceDataset(train_std)
    val_ds = WeatherSequenceDataset(val_std)
    test_ds = WeatherSequenceDataset(test_std)
    print(f"  train: {len(train_ds):,} sequences")
    print(f"  val:   {len(val_ds):,} sequences")
    print(f"  test:  {len(test_ds):,} sequences")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    # ---- Build and train the model ----
    print("\n--- LSTM training ---")
    n_features = train_df.shape[1]
    model = LSTMForecaster(n_features=n_features)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: 2-layer LSTM, hidden={HIDDEN_SIZE}, "
          f"dropout={DROPOUT}")
    print(f"  Parameters: {n_params:,}")
    history = train_lstm(model, train_loader, val_loader, device)

    # ---- Evaluate on the test set ----
    print("\n--- Test evaluation ---")
    lstm_result = evaluate_on_test(model, test_loader, standardizer, device)

    # ---- Build leaderboard ----
    results = []
    if LEADERBOARD_PATH.exists():
        prior_lb = pd.read_csv(LEADERBOARD_PATH)
        # Filter out prior runs of THIS weekend's model in case we re-ran
        prior_lb = prior_lb[~prior_lb["model"].isin(["LSTM"])]
        for _, row in prior_lb.iterrows():
            results.append({"model": row["model"],
                            "mae": float(row["mae"]),
                            "rmse": float(row["rmse"])})
            print(f"  loaded prior: {row['model']:25s}  "
                  f"MAE = {row['mae']:.3f} °C")
    results.append({"model": "LSTM",
                    "mae": lstm_result["mae"],
                    "rmse": lstm_result["rmse"]})

    # ---- Note on leaderboard comparability ----
    # The LSTM MAE/RMSE values reported above are averaged across ALL 24
    # forecast horizons (h=1 through h=24). This is the same convention
    # used by SARIMA, the baselines, and the gradient-boosting models in
    # prior weekends: each evaluation flattens 24-step forecast chunks
    # and computes MAE over the flat list. The leaderboard comparison is
    # apples-to-apples on that basis.

    # ---- Horizon-level metrics for auditability ----
    y_true_deg = lstm_result["y_true_deg"]
    y_pred_deg = lstm_result["y_pred_deg"]
    horizon_metrics = pd.DataFrame({
        "horizon": np.arange(1, OUTPUT_HORIZON + 1),
        "mae": np.mean(np.abs(y_true_deg - y_pred_deg), axis=0),
        "rmse": np.sqrt(np.mean((y_true_deg - y_pred_deg) ** 2, axis=0)),
    })
    horizon_csv_path = OUTPUT_DIR / "lstm_horizon_metrics.csv"
    horizon_metrics.to_csv(horizon_csv_path, index=False)
    print(f"\n  saved per-horizon metrics: {horizon_csv_path}")
    print(f"  MAE by horizon (first / middle / last):")
    print(f"    h= 1: MAE={horizon_metrics.iloc[0]['mae']:.3f} °C")
    print(f"    h=12: MAE={horizon_metrics.iloc[11]['mae']:.3f} °C")
    print(f"    h=24: MAE={horizon_metrics.iloc[23]['mae']:.3f} °C")

    # ---- Compute horizon-24-only residuals for the diagnostic plot ----
    # The flattened residuals from evaluate_on_test() are NOT a clean
    # hourly time series — they concatenate overlapping forecast windows.
    # Plotting their ACF would measure flattening artifacts, not real
    # residual structure. The honest diagnostic uses residuals from a
    # SINGLE horizon (here, h=24) so the lag axis means hours of real
    # forecast-origin time.
    h_last = OUTPUT_HORIZON - 1
    residuals_h24 = y_true_deg[:, h_last] - y_pred_deg[:, h_last]

    # ---- Plots ----
    print("\n--- Plots ---")
    plot_training_curves(history, OUTPUT_DIR / "01_training_curves.png")
    plot_test_week_zoom(test_std, y_true_deg, y_pred_deg,
                        OUTPUT_DIR / "02_test_week_zoom.png")
    plot_residual_diagnostics(residuals_h24,
                              OUTPUT_DIR / "03_residual_diagnostics_h24.png",
                              horizon_label="24h-ahead")
    plot_mae_by_horizon(y_true_deg, y_pred_deg,
                        OUTPUT_DIR / "04_mae_by_horizon.png")
    plot_leaderboard(results, OUTPUT_DIR / "05_leaderboard.png")

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