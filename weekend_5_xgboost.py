"""
Weekend 5: The Bridge to Modern ML — Feature Engineering + Gradient Boosting

Picks up from Weekend 4 (SARIMA wins at MAE 1.886). 
- Gradient-boosted trees (XGBoost, LightGBM)
- Engineered features: lags, rolling means, cyclical encodings
- ALL 14 weather variables, not just temperature
- The point that feature engineering matters more than the model

Pipeline:
1. Load Jena, resample to hourly, keep all 14 features
2. Engineer 50 features per row across four categories:
   - Target lags (8): lag-1, 2, 3, 6, 12, 24, 48, 168
   - Target rolling statistics (12): mean/std/min/max over 6h/24h/168h
   - Exogenous lags (26): 13 other weather variables at lag-1 and lag-24
   - Cyclical encodings (4): sin/cos of hour-of-day and day-of-year
3. Build supervised regression targets: predict T(t+24) from features at time t
4. Chronological train/val/test split BY TARGET TIMESTAMP (no boundary leak)
5. Fit XGBoost and LightGBM (both with importance_type="gain")
6. Compute feature importances on a comparable scale
7. Update leaderboard

Run end-to-end:
    python weekend_5_xgboost.py

Outputs five plots to ./images/weekend_5/

Compute: ~5 minutes wall time (feature engineering is the slowest step).
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
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Configuration — locked from the series spec
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
while not (PROJECT_ROOT / "data").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_PATH = Path("/Users/rumasinha/random/timeseries_analysis/data/jena_climate_2009_2016.csv")
OUTPUT_DIR = PROJECT_ROOT / "images" / "weekend_5"
LEADERBOARD_PATH = PROJECT_ROOT / "data" / "leaderboard.csv"
RANDOM_SEED = 42

TRAIN_END = "2014-12-31 23:00:00"
VAL_END = "2015-12-31 23:00:00"
TEST_END = "2016-12-31 23:00:00"
FORECAST_HORIZON = 24
TARGET_COL = "T (degC)"

# Lag windows for feature engineering — chosen for clear physical meaning
LAGS = [1, 2, 3, 6, 12, 24, 48, 168]      # 1h, 2h, ..., 1 week
ROLLING_WINDOWS = [6, 24, 168]             # 6h, 1d, 1 week

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
# Step 1: Load all 14 features (not just temperature like Weekends 1-4)
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
    return hourly  # all 14 columns


def chronological_split(df: pd.DataFrame):
    """Chronological split with NO target-boundary leakage.

    We split by the target timestamp, not the feature timestamp.  Why:
    each engineered row at feature time t has target = T(t+24).  If we
    naively split by feature time at 2014-12-31 23:00, the last 24 rows
    of train would have targets falling in 2015 — which is val data.
    Splitting by target_time fixes this cleanly.

    The data the split sees:
    - train: rows whose TARGET falls in 2009..2014
    - val:   rows whose TARGET falls in 2015 (reserved for Weekend 8)
    - test:  rows whose TARGET falls in 2016
    """
    train = df[df["target_time"] <= TRAIN_END]
    val = df[(df["target_time"] > TRAIN_END) & (df["target_time"] <= VAL_END)]
    test = df[(df["target_time"] > VAL_END) & (df["target_time"] <= TEST_END)]
    return train, val, test


# ---------------------------------------------------------------------------
# Step 2: Feature engineering — the heart of this weekend
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame, target_col: str = TARGET_COL,
                      lags: list = None, rolling_windows: list = None) -> pd.DataFrame:
    """Convert raw multivariate time series into a supervised regression dataset.

    For each row at time t, generate four categories of features:

    1. Target lags (8 features): lag-1, 2, 3, 6, 12, 24, 48, 168 of the target.
    2. Target rolling statistics (12 features): mean/std/min/max of the target
       over the past 6h, 24h, 168h, each shifted by 1 to avoid leaking T(t).
    3. Exogenous lags (26 features): for each of the 13 other weather variables,
       lag-1 and lag-24. We use only short and medium lags to keep the matrix
       at a reasonable width.
    4. Cyclical time encodings (4 features): sin/cos of hour-of-day and
       day-of-year, so the model treats hour 23 and hour 0 as adjacent on a
       circle rather than 23 units apart on a line.

    This is the conceptual leap from classical time series: instead of asking
    'how does y(t) depend on y(t-k)' inside one model, we engineer the
    dependence as input columns and let a tabular regressor figure out which
    columns matter.

    Returns a DataFrame with 50 feature columns + 'target' (T at t+24) +
    'target_time' (the timestamp the target lives at, used for honest splitting).
    """
    if lags is None:
        lags = LAGS
    if rolling_windows is None:
        rolling_windows = ROLLING_WINDOWS

    out = pd.DataFrame(index=df.index)

    # --- Lag features for the target ---
    for lag in lags:
        out[f"target_lag_{lag}"] = df[target_col].shift(lag)

    # --- Lag features for other variables (just a few key ones to control width)
    # Adding 14 cols * 8 lags = 112 features is overkill; restrict to a small
    # set of physically meaningful exogenous variables with selected lags
    exogenous_vars = [c for c in df.columns if c != target_col]
    for var in exogenous_vars:
        # Only short and medium lags for exogenous variables
        for lag in [1, 24]:
            out[f"{var}_lag_{lag}"] = df[var].shift(lag)

    # --- Rolling statistics on target (shift first to avoid leakage) ---
    target_shifted = df[target_col].shift(1)  # don't use the current value
    for w in rolling_windows:
        out[f"target_roll{w}_mean"] = target_shifted.rolling(w).mean()
        out[f"target_roll{w}_std"] = target_shifted.rolling(w).std()
        out[f"target_roll{w}_min"] = target_shifted.rolling(w).min()
        out[f"target_roll{w}_max"] = target_shifted.rolling(w).max()

    # --- Cyclical time encodings ---
    # Encoding hour-of-day with sin/cos preserves the cycle's continuity:
    # midnight (0) and 23:00 are 'close' on the circle, not 23 hours apart
    out["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    out["day_of_year_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    out["day_of_year_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

    # --- Target: temperature 24 hours from now ---
    out["target"] = df[target_col].shift(-FORECAST_HORIZON)

    # --- Target timestamp: when the prediction is FOR (not when it's made) ---
    # This lets us split chronologically by the date being predicted rather
    # than by the feature timestamp, preventing the 24-hour boundary leak
    # where training features at 2014-12-31 23:00 have targets in 2015.
    out["target_time"] = out.index + pd.Timedelta(hours=FORECAST_HORIZON)

    # Drop rows with any NaN (early lags, late targets, missing rolling values)
    out = out.dropna()
    return out


# ---------------------------------------------------------------------------
# Step 3: Train/eval helpers
# ---------------------------------------------------------------------------
def split_features_target(df: pd.DataFrame):
    """Separate features (X) from target (y).

    Both 'target' and 'target_time' are bookkeeping columns; neither
    should enter the model.
    """
    X = df.drop(columns=["target", "target_time"])
    y = df["target"]
    return X, y


def evaluate(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Predict on test set and return MAE/RMSE."""
    yhat = model.predict(X_test)
    residuals = y_test.values - yhat
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    print(f"  {name:25s}  MAE = {mae:.3f} °C   RMSE = {rmse:.3f} °C")
    return {"model": name, "mae": mae, "rmse": rmse,
            "yhat": yhat, "residuals": residuals}


# ---------------------------------------------------------------------------
# Step 4: Plot helpers
# ---------------------------------------------------------------------------
def plot_feature_importance(importance_dict: dict, save_path: Path,
                            top_n: int = 20):
    """Side-by-side bar chart of top features for XGBoost and LightGBM."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False)

    for ax, (model_name, imp) in zip(axes, importance_dict.items()):
        top = imp.head(top_n).iloc[::-1]  # reverse for top-down bar order
        ax.barh(top.index, top.values, color="#2E5077")
        ax.set_title(f"{model_name} — top {top_n} features by importance")
        ax.set_xlabel("Importance (gain)")
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def plot_test_window_zoom(test_y: pd.Series, predictions: dict,
                          save_path: Path,
                          window_start: str = "2016-07-01",
                          window_end: str = "2016-07-08"):
    """One-week zoom showing each model's forecast vs actual.

    Important indexing note:
    test_y is indexed at FEATURE time t — its values are the targets
    at t + FORECAST_HORIZON. To display the chart on the dates being
    predicted (which is what readers expect to see), we shift both the
    actual series and the prediction series forward by FORECAST_HORIZON.
    Without this, the chart's x-axis labels would be 24 hours earlier
    than the actual values being plotted.
    """
    # Shift both series so the index reflects the date being predicted
    shift = pd.Timedelta(hours=FORECAST_HORIZON)
    actual_at_target = pd.Series(test_y.values, index=test_y.index + shift)

    fig, ax = plt.subplots(figsize=(14, 5))
    actual_window = actual_at_target.loc[window_start:window_end]
    ax.plot(actual_window.index, actual_window.values,
            color="#222222", linewidth=1.5, label="Actual")

    colors = {"XGBoost": "#FF6B35", "LightGBM": "#1E8E5A"}
    for name, yhat in predictions.items():
        yhat_at_target = pd.Series(yhat, index=test_y.index + shift)
        window_pred = yhat_at_target.loc[window_start:window_end]
        ax.plot(window_pred.index, window_pred.values,
                color=colors.get(name, "#888888"), linewidth=2,
                alpha=0.85, label=name)

    ax.set_title(f"XGBoost vs LightGBM 24h-ahead forecast: "
                 f"{window_start} → {window_end}")
    ax.set_xlabel("Date (target time, i.e. the date being predicted)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def plot_leaderboard(results: list, save_path: Path):
    """Bar chart with full leaderboard across all five weekends."""
    df = pd.DataFrame(results).sort_values("mae")
    # Color code:
    color_map = {
        "XGBoost": "#FF6B35",
        "LightGBM": "#1E8E5A",
        "SARIMA": "#2E8B57",
        "Prophet": "#C04A4A",
    }
    colors = [color_map.get(name, "#888888") for name in df["model"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(df["model"], df["mae"], color=colors)
    ax.set_xlabel("MAE (°C) — lower is better")
    ax.set_title("Weekend 5 leaderboard: gradient boosting joins the lineup "
                 "(24h-ahead forecast)")
    for bar, val in zip(bars, df["mae"]):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def plot_feature_category_breakdown(importance_dict: dict, save_path: Path):
    """Group features into categories and show how much each category matters.

    Categories:
    - Target lags (autoregressive — the SARIMA-style information)
    - Target rolling stats (smoothed recent history)
    - Exogenous lags (other weather variables — the multivariate gain)
    - Cyclical time encodings (calendar features)
    """
    def categorize(feat):
        if feat.startswith("target_lag_"):
            return "Target lags"
        if feat.startswith("target_roll"):
            return "Target rolling stats"
        if feat in {"hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos"}:
            return "Cyclical time"
        return "Exogenous lags"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (model_name, imp) in zip(axes, importance_dict.items()):
        cats = imp.groupby(categorize).sum().sort_values(ascending=True)
        # Pre-2025-color palette
        category_colors = {
            "Target lags": "#2E5077",
            "Target rolling stats": "#5B9BD5",
            "Exogenous lags": "#C04A4A",
            "Cyclical time": "#888888",
        }
        bar_colors = [category_colors[c] for c in cats.index]
        ax.barh(cats.index, cats.values, color=bar_colors)
        total = cats.sum()
        for i, (name, val) in enumerate(cats.items()):
            pct = 100 * val / total
            ax.text(val + total * 0.01, i, f"{pct:.0f}%", va="center", fontsize=10)
        ax.set_title(f"{model_name} — importance by feature category")
        ax.set_xlabel("Total importance (gain)")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def plot_lag_importance_sweep(importance_dict: dict, save_path: Path):
    """Show how much each lag matters, side by side for both models.

    The dramatic story: which lags are doing the work? lag-1? lag-24?
    """
    import re

    def get_target_lag_imp(imp_series):
        # Filter to target_lag_X and extract the lag number
        lag_imp = {}
        for feat, val in imp_series.items():
            m = re.match(r"target_lag_(\d+)$", feat)
            if m:
                lag_imp[int(m.group(1))] = val
        return pd.Series(lag_imp).sort_index()

    fig, ax = plt.subplots(figsize=(12, 4))
    width = 0.4
    x = np.arange(len(LAGS))
    for i, (name, imp) in enumerate(importance_dict.items()):
        lag_imp = get_target_lag_imp(imp)
        # Align with LAGS order
        values = [lag_imp.get(l, 0) for l in LAGS]
        offset = (i - 0.5) * width
        color = "#FF6B35" if name == "XGBoost" else "#1E8E5A"
        ax.bar(x + offset, values, width, label=name, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"lag-{l}" for l in LAGS])
    ax.set_xlabel("Lag of the target variable")
    ax.set_ylabel("Importance (gain)")
    ax.set_title("Which target lags matter most? (XGBoost vs LightGBM)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {save_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Weekend 5: XGBoost + LightGBM with engineered features")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load multivariate hourly data ----
    df = load_hourly_multivariate(DATA_PATH)
    print(f"\nLoaded {len(df):,} hourly rows × {df.shape[1]} variables")
    print(f"  Variables: {', '.join(df.columns)}")

    # Sanity check: T and VPmax should be near-perfectly correlated
    # (Clausius-Clapeyron makes saturation vapor pressure a function of T).
    # The post claims ~0.95; let's print the actual number.
    t_vpmax_corr = df[["T (degC)", "VPmax (mbar)"]].corr().iloc[0, 1]
    print(f"  T vs VPmax correlation: {t_vpmax_corr:.3f}  "
          f"(near-equivalence drives the VPmax_lag_1 'shortcut')")

    # ---- Engineer features (this is THE work of this weekend) ----
    print("\n--- Feature engineering ---")
    fe = engineer_features(df, target_col=TARGET_COL,
                           lags=LAGS, rolling_windows=ROLLING_WINDOWS)
    print(f"  Engineered {fe.shape[1] - 1} features per row "
          f"({fe.shape[0]:,} usable rows after dropping NaN edges)")
    feature_categories = {
        "target lags": sum(c.startswith("target_lag_") for c in fe.columns),
        "target rolling stats": sum(c.startswith("target_roll") for c in fe.columns),
        "exogenous lags": sum("_lag_" in c and not c.startswith("target_") for c in fe.columns),
        "cyclical encodings": sum(c.endswith("_sin") or c.endswith("_cos") for c in fe.columns),
    }
    print(f"  Categories: {feature_categories}")

    # ---- Chronological split ----
    train_fe, val_fe, test_fe = chronological_split(fe)
    print(f"\nTrain:  {train_fe.index.min()} → {train_fe.index.max()}  ({len(train_fe):,} rows)")
    print(f"Val:    {val_fe.index.min()} → {val_fe.index.max()}  ({len(val_fe):,} rows)  [reserved for Weekend 8]")
    print(f"Test:   {test_fe.index.min()} → {test_fe.index.max()}  ({len(test_fe):,} rows)")

    X_train, y_train = split_features_target(train_fe)
    X_test, y_test = split_features_target(test_fe)

    # ---- Load Prophet + SARIMA from prior leaderboard ----
    results = []
    if LEADERBOARD_PATH.exists():
        prior_lb = pd.read_csv(LEADERBOARD_PATH)
        # Exclude any prior runs of THIS weekend's models so reruns don't pile up
        prior_lb = prior_lb[~prior_lb["model"].isin(["XGBoost", "LightGBM"])]
        for _, row in prior_lb.iterrows():
            results.append({"model": row["model"],
                            "mae": float(row["mae"]),
                            "rmse": float(row["rmse"])})
            print(f"  loaded prior: {row['model']:25s}  "
                  f"MAE = {row['mae']:.3f} °C")

    # ---- XGBoost ----
    print("\n--- XGBoost ---")
    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        verbosity=0,
        tree_method="hist",
    )
    xgb.fit(X_train, y_train)
    xgb_result = evaluate("XGBoost", xgb, X_test, y_test)
    results.append({"model": "XGBoost", "mae": xgb_result["mae"],
                    "rmse": xgb_result["rmse"]})

    # ---- LightGBM ----
    print("\n--- LightGBM ---")
    lgbm = LGBMRegressor(
        n_estimators=500,
        max_depth=-1,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        subsample_freq=1,        # LGBM row subsampling needs freq>0 to take effect
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        verbose=-1,
        importance_type="gain",  # default is "split" (count); use gain to match XGBoost
    )
    lgbm.fit(X_train, y_train)
    lgbm_result = evaluate("LightGBM", lgbm, X_test, y_test)
    results.append({"model": "LightGBM", "mae": lgbm_result["mae"],
                    "rmse": lgbm_result["rmse"]})

    # ---- Feature importances ----
    print("\n--- Feature importances ---")
    xgb_imp = pd.Series(xgb.feature_importances_,
                        index=X_train.columns).sort_values(ascending=False)
    lgbm_imp = pd.Series(lgbm.feature_importances_,
                         index=X_train.columns).sort_values(ascending=False)
    print("  Top 10 features (XGBoost):")
    for feat, val in xgb_imp.head(10).items():
        print(f"    {feat:35s}  {val:.4f}")
    print("  Top 10 features (LightGBM):")
    for feat, val in lgbm_imp.head(10).items():
        print(f"    {feat:35s}  {val:.0f}")

    importance_dict = {"XGBoost": xgb_imp, "LightGBM": lgbm_imp}

    # ---- Plots ----
    print("\n--- Plots ---")
    plot_feature_importance(importance_dict,
                            OUTPUT_DIR / "01_feature_importance_top20.png")
    plot_feature_category_breakdown(importance_dict,
                                    OUTPUT_DIR / "02_importance_by_category.png")
    plot_lag_importance_sweep(importance_dict,
                              OUTPUT_DIR / "03_lag_importance_sweep.png")
    plot_test_window_zoom(
        y_test,
        {"XGBoost": xgb_result["yhat"], "LightGBM": lgbm_result["yhat"]},
        OUTPUT_DIR / "04_test_week_zoom.png",
    )
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