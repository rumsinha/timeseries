"""
Weekend 8: Hyperparameter Tuning with Time-Series Cross-Validation

The validation set (2015) was reserved for this weekend in every prior post.
Now it gets used properly. This script tunes all five contender models with
Optuna, scoring each candidate configuration by mean MAE across THREE
expanding-window cross-validation folds, then evaluates the single best
configuration once on the held-out 2016 test set.

Models tuned: SARIMA, XGBoost, LightGBM, LSTM, N-BEATS.

Design (read before editing):
- ONE tuning protocol shared by all models. Each model supplies an "adapter"
  that knows three things: its Optuna search space, how to build+fit on a
  date range, and how to predict. The protocol never branches on model type.
- Cross-validation is EXPANDING WINDOW, never shuffled:
      fold 1: train 2009-2012 -> validate 2013
      fold 2: train 2009-2013 -> validate 2014
      fold 3: train 2009-2014 -> validate 2015
  Validation always comes chronologically AFTER training. Test (2016) is
  untouched until the final evaluation.
- SARIMA uses a fixed trailing two-year fit window inside each training fold;
  the other adapters use the complete training fold.
- Every final evaluation uses the same 8,593 stride-1 forecast origins. At
  origin o, observations end at o-1 and horizon h targets y[o+h-1].
- Neural nets train for a fixed epoch budget on the complete training fold.
  Optuna pruning occurs between folds, so no internal holdout is discarded and
  no un-restored early-stopping checkpoint can affect the score.

Outputs (to ./tuning_weekend_8/):
    best_configs.json                          best hyperparameters per model
    tuned_leaderboard.csv                      test MAE/RMSE for each tuned model
    per_horizon/{model}_tuned_horizon.csv      per-horizon profile, tuned model
    optuna_studies/{model}.pkl                 the Optuna study (inspectable)

Run:
    python weekend_8_tuning_v9.py
    python weekend_8_tuning_v9.py --models sarima xgboost
    python weekend_8_tuning_v9.py --quick

Dependencies:
    pip install optuna u8darts[torch] xgboost lightgbm statsmodels
"""

import argparse
import json
import logging
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    import optuna
except ImportError:
    optuna = None  # allows --help and import without the dep; main() checks

for noisy in ("optuna", "pytorch_lightning", "darts", "statsmodels"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Paths — two anchors, same pattern as Weekend 7 v4 (data vs notebook outputs)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
_data_marker = "data/jena_climate_2009_2016.csv"
DATA_ROOT = SCRIPT_DIR
while not (DATA_ROOT / _data_marker).exists() and DATA_ROOT.parent != DATA_ROOT:
    DATA_ROOT = DATA_ROOT.parent
NOTEBOOK_ROOT = SCRIPT_DIR

DATA_PATH = DATA_ROOT / "data" / "jena_climate_2009_2016.csv"
OUTPUT_DIR = NOTEBOOK_ROOT / "tuning_weekend_8"
PER_HORIZON_DIR = OUTPUT_DIR / "per_horizon"
STUDIES_DIR = OUTPUT_DIR / "optuna_studies"

RANDOM_SEED = 42
CODE_VERSION = "v9"

# Neural-net device. MPS degraded badly on the long real run: every LSTM fit
# hit the mid-training Timer and was scored UNDERTRAINED (test MAE 2.4 vs the
# untuned 1.87), and an N-BEATS trial diverged to ~1e13 MAE. CPU is slower per
# epoch but each fit TRAINS TO COMPLETION and is numerically stable, which is
# what these two models need. Default CPU; override with
# WEEKEND8_NEURAL_DEVICE=mps if you ever want to try the accelerator again.
import os
NEURAL_DEVICE = os.environ.get("WEEKEND8_NEURAL_DEVICE", "cpu").lower()

# ---------------------------------------------------------------------------
# Locked series spec
# ---------------------------------------------------------------------------
TARGET_COL = "T (degC)"
INPUT_WINDOW = 168
OUTPUT_HORIZON = 24

# Date boundaries. Test stays sealed until final evaluation.
CV_FOLDS = [
    # (train_start, train_end, val_year_start, val_year_end)
    ("2009-01-01 00:00:00", "2012-12-31 23:00:00", "2013-01-01 00:00:00", "2013-12-31 23:00:00"),
    ("2009-01-01 00:00:00", "2013-12-31 23:00:00", "2014-01-01 00:00:00", "2014-12-31 23:00:00"),
    ("2009-01-01 00:00:00", "2014-12-31 23:00:00", "2015-01-01 00:00:00", "2015-12-31 23:00:00"),
]
# Final retrain span (everything before test) and the sealed test year.
FINAL_TRAIN_END = "2015-12-31 23:00:00"
TEST_START = "2016-01-01 00:00:00"
TEST_END = "2016-12-31 23:00:00"

# Trial budgets per model — revised after the first run showed the original
# estimates were far too optimistic. SARIMA's rolling eval and the neural
# nets' 3x-fold training dominate. TIMEOUT_SECONDS is a study limit checked by
# Optuna between trials; an active trial is allowed to finish cleanly.
TRIAL_BUDGET = {
    "sarima": 12,
    "xgboost": 40,
    "lightgbm": 40,
    "lstm": 8,
    "nbeats": 6,
}
TIMEOUT_SECONDS = {
    "sarima": 30 * 60,
    "xgboost": 15 * 60,
    "lightgbm": 15 * 60,
    "lstm": 45 * 60,
    "nbeats": 45 * 60,
}
QUICK_BUDGET = {k: 2 for k in TRIAL_BUDGET}

# Set True by --quick. A smoke test must finish in minutes, so this shrinks the
# per-fold work itself (fit window, eval span, neural epochs) — not just the
# trial count. Quick-mode numbers are throwaway; the goal is "does the whole
# pipeline run end to end without crashing," nothing more.
QUICK = False

np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_full() -> pd.DataFrame:
    """Load Jena, hourly, all 14 variables (some models use exogenous ones)."""
    if not DATA_PATH.exists():
        sys.exit(f"\nERROR: data not found at {DATA_PATH}\n"
                 "Place jena_climate_2009_2016.csv under data/.\n")
    df = pd.read_csv(DATA_PATH)
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.set_index("Date Time").resample("1h").mean()
    return df.asfreq("h").interpolate("linear")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# Adapter contract
# ---------------------------------------------------------------------------
@dataclass
class FoldResult:
    """Flattened predictions plus the horizon labels represented by each row."""
    y_true: np.ndarray
    y_pred: np.ndarray
    horizons: np.ndarray = field(
        default_factory=lambda: np.arange(1, OUTPUT_HORIZON + 1)
    )


class ModelAdapter:
    """Base class. Each model subclasses this.

    The tuning protocol calls exactly two things:
      - suggest_config(trial): return a dict of hyperparameters from the trial
      - run_fold(config, df, train_start, train_end, eval_start, eval_end,
                 report_cb): fit on [train_start, train_end], forecast across
                 [eval_start, eval_end], return FoldResult in degC.

    The `final` flag controls evaluation resolution:
      final=False (tuning): SAMPLED evaluation — enough origins to RANK
        configurations reliably, cheap enough for trials x folds.
      final=True (sealed test): FULL resolution matching the series convention
        (stride-1 origins for every model), so reported numbers are comparable.
    Sampled eval for ranking is a standard tuning trick: config A vs config B
    ordering is stable under subsampling even though absolute MAE shifts a bit.
    """
    name: str = "base"
    prunable: bool = False  # only neural nets report intermediate values
    final: bool = False     # evaluate_on_test flips this to True

    def suggest_config(self, trial) -> dict:
        raise NotImplementedError

    def run_fold(self, config: dict, df: pd.DataFrame,
                 train_start: str, train_end: str,
                 eval_start: str, eval_end: str,
                 report_cb: Callable[[int, float], None] | None = None) -> FoldResult:
        raise NotImplementedError


def canonical_test_origins(n_test_rows: int) -> np.ndarray:
    """THE single forecast-origin convention shared by every model's final
    sealed-test evaluation, so all leaderboard MAEs are directly comparable
    and the Diebold-Mariano test can compare any pair.

    Convention: origin o indexes the test-year array. The model has observed
    rows [.. o-1] (a full INPUT_WINDOW of history sits before o) and forecasts
    the next OUTPUT_HORIZON hours, i.e. test_y[o : o + OUTPUT_HORIZON].

      first origin = INPUT_WINDOW         (a full 168h window precedes it)
      last  origin = n_test_rows - OUTPUT_HORIZON   (room for a full 24h target)

    On the 8,784-hour 2016 leap year this yields 8,784 - 168 - 24 + 1 = 8,593
    origins — exactly the Weekend 6-7 LSTM/N-BEATS count.
    """
    first = INPUT_WINDOW
    last = n_test_rows - OUTPUT_HORIZON
    if last < first:
        return np.array([], dtype=int)
    return np.arange(first, last + 1)


def direct_target_offset(horizon: int) -> int:
    """Row offset for a 1-based horizon under the canonical origin contract."""
    if not 1 <= horizon <= OUTPUT_HORIZON:
        raise ValueError(f"horizon must be in 1..{OUTPUT_HORIZON}, got {horizon}")
    return horizon - 1


def direct_train_cutoff(train_end: str, horizon: int) -> pd.Timestamp:
    """Latest feature timestamp whose shifted label remains pre-test."""
    return pd.Timestamp(train_end) - pd.Timedelta(hours=direct_target_offset(horizon))


# ---------------------------------------------------------------------------
# SARIMA adapter
# ---------------------------------------------------------------------------
class SarimaAdapter(ModelAdapter):
    name = "sarima"
    prunable = False

    def suggest_config(self, trial) -> dict:
        # Search a small neighborhood around the Weekend 4 orders
        # SARIMA(2,1,1)(1,1,1,24). Keep it tight: SARIMA fits are cheap but
        # not free, and wild orders rarely help on strongly seasonal data.
        return {
            "p": trial.suggest_int("p", 1, 3),
            "d": trial.suggest_int("d", 0, 1),
            "q": trial.suggest_int("q", 0, 2),
            "P": trial.suggest_int("P", 0, 2),
            "D": trial.suggest_int("D", 0, 1),
            "Q": trial.suggest_int("Q", 0, 1),
            "s": 24,
        }

    def run_fold(self, config, df, train_start, train_end,
                 eval_start, eval_end, report_cb=None):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y = df[TARGET_COL]
        # Keep the model's fit policy identical during tuning and final scoring.
        # A fixed two-year lookback controls SARIMA cost and is part of this
        # adapter's declared specification, not a test-time special case.
        fit_days = 90 if QUICK else 730
        fit_start = pd.Timestamp(train_end) - pd.Timedelta(days=fit_days)
        train = y.loc[max(pd.Timestamp(train_start), fit_start):train_end]

        order = (config["p"], config["d"], config["q"])
        # 's' is a fixed constant (24), not a searched param, so Optuna's
        # best_params won't contain it — default it here rather than KeyError.
        seasonal = (config["P"], config["D"], config["Q"], config.get("s", 24))
        model = SARIMAX(train, order=order, seasonal_order=seasonal,
                        enforce_stationarity=False, enforce_invertibility=False)
        maxiter = 20 if QUICK else 100
        fit = model.fit(disp=False, maxiter=maxiter, low_memory=True)
        if not fit.mle_retvals.get("converged", True):
            message = f"SARIMA failed to converge within {maxiter} iterations"
            if self.final:
                raise RuntimeError(message)
            print(f"    {message}; pruning trial")
            raise optuna.TrialPruned()

        # Forecast at stride-1 origins for final evaluation. Tuning samples one
        # origin every four days, while still appending all intervening observed
        # values so the state at each sampled origin is correct.
        origin_step = 1 if self.final else 4 * OUTPUT_HORIZON
        eval_series = y.loc[eval_start:eval_end]
        # Quick mode: only evaluate the first ~2 weeks so a fold is seconds.
        if QUICK:
            eval_series = eval_series.iloc[:INPUT_WINDOW + 14 * OUTPUT_HORIZON]
            origin_step = OUTPUT_HORIZON
        n = len(eval_series)
        eval_vals = eval_series.values
        preds, actuals = [], []
        state = fit.append(eval_series.iloc[:INPUT_WINDOW], refit=False)
        i = INPUT_WINDOW
        while i + OUTPUT_HORIZON <= n:
            fc = state.forecast(steps=OUTPUT_HORIZON)
            preds.append(np.asarray(fc))
            actuals.append(eval_vals[i:i + OUTPUT_HORIZON])
            next_i = min(i + origin_step, n)
            state = state.extend(eval_series.iloc[i:next_i])
            i = next_i
        if not preds:
            return FoldResult(np.array([]), np.array([]))
        return FoldResult(np.concatenate(actuals), np.concatenate(preds))


# ---------------------------------------------------------------------------
# Gradient-boosting adapters (XGBoost, LightGBM) share feature engineering
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature family as Weekend 5: target lags, rolling stats, exogenous
    lags, cyclical time encodings. Kept compact here for tuning speed."""
    out = pd.DataFrame(index=df.index)
    t = df[TARGET_COL]
    for lag in (1, 2, 3, 24, 48, 168):
        out[f"t_lag_{lag}"] = t.shift(lag)
    for win in (3, 24):
        out[f"t_rollmean_{win}"] = t.shift(1).rolling(win).mean()
        out[f"t_rollstd_{win}"] = t.shift(1).rolling(win).std()
    # A couple of strong exogenous signals from Weekend 5
    for col in ("VPmax (mbar)", "rh (%)", "p (mbar)"):
        if col in df.columns:
            out[f"{col}_lag1"] = df[col].shift(1)
    # Cyclical encodings
    hour = df.index.hour
    doy = df.index.dayofyear
    out["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    out["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    out["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    out["target"] = t
    return out


class _BoostingAdapter(ModelAdapter):
    """Shared logic for XGBoost and LightGBM.

    Tuning and final evaluation both use direct multi-horizon models. Tuning
    samples five representative horizons and daily origins for tractability;
    final evaluation trains all 24 horizons and scores every canonical origin.
    """
    prunable = False

    def _make_estimator(self, config):
        raise NotImplementedError

    def run_fold(self, config, df, train_start, train_end,
                 eval_start, eval_end, report_cb=None):
        horizons = [1, 12, 24] if QUICK else [1, 6, 12, 18, 24]
        return self._run_direct(config, df, train_start, train_end,
                                eval_start, eval_end, horizons, origin_stride=24)

    def run_fold_direct(self, config, df, train_start, train_end,
                        eval_start, eval_end) -> FoldResult:
        """Honest direct multi-horizon: one model per horizon h=1..24.

        Builds targets T(t+h-1) for each 1-based horizon h, trains a separate
        estimator, and
        assembles a (n_origins, 24) prediction matrix on the CANONICAL test
        origins so the result is directly comparable to the neural/SARIMA
        models and usable in the Diebold-Mariano test.

        Two correctness properties:
        - LEAK GUARD: training rows are capped at train_end - (h-1), so no
          shifted label reaches into the sealed test year.
        - SHARED ORIGINS: predictions are emitted only at canonical_test_origins,
          identical to the neural convention (first origin = INPUT_WINDOW into
          the test year), not at every test row.

        Quick smoke-test: train only horizons {1, 12, 24} on a 1-year window.
        """
        horizons = [1, 12, 24] if QUICK else list(range(1, OUTPUT_HORIZON + 1))
        return self._run_direct(config, df, train_start, train_end,
                                eval_start, eval_end, horizons, origin_stride=1)

    def _run_direct(self, config, df, train_start, train_end,
                    eval_start, eval_end, horizons, origin_stride) -> FoldResult:
        feats = build_features(df)
        feature_cols = [c for c in feats.columns if c != "target"]
        base = feats.copy()
        tgt_all = base["target"]

        if QUICK:
            q_start = pd.Timestamp(train_end) - pd.Timedelta(days=365)
            train_start = str(max(pd.Timestamp(train_start), q_start))

        # Canonical origins, expressed as timestamps in the test span.
        test_index = base.loc[eval_start:eval_end].index
        n_test = len(test_index)
        origins = canonical_test_origins(n_test)[::origin_stride]
        # Features at timestamp o contain target/exogenous lags ending at o-1,
        # exactly the information available when the forecast starts at o.
        origin_ts = test_index[origins]

        # Feature rows are timestamped at the forecast origin and contain only
        # lagged observations from before that origin.
        feat_at_origin = base.loc[origin_ts, feature_cols]
        valid_mask = ~feat_at_origin.isna().any(axis=1).values
        origins_v = origins[valid_mask]
        X_origin = feat_at_origin.values[valid_mask]

        test_y = base.loc[eval_start:eval_end, "target"].values
        pred_cols, true_cols = [], []
        for h in horizons:
            target_offset = direct_target_offset(h)
            tgt = tgt_all.shift(-target_offset) if target_offset else tgt_all
            # LEAK GUARD: latest training label must be <= train_end.
            train_cutoff = direct_train_cutoff(train_end, h)
            tr = base.loc[train_start:train_cutoff].assign(
                _y=tgt.loc[train_start:train_cutoff]).dropna()
            est = self._make_estimator(config)
            est.fit(tr[feature_cols].values, tr["_y"].values)

            # Predict T at horizon h from the origin's features.
            pred_cols.append(est.predict(X_origin))
            # CANONICAL truth: origin offset o, horizon h (1-based) -> test_y[o+h-1].
            # This matches the neural models' per-horizon column convention
            # (column j=h-1 is full[s+j]) exactly, so all models align.
            true_cols.append(np.array([test_y[o + h - 1] for o in origins_v]))

        y_pred = np.stack(pred_cols, axis=1)   # (n_valid_origins, n_horizons)
        y_true = np.stack(true_cols, axis=1)
        return FoldResult(y_true.flatten(), y_pred.flatten(), np.asarray(horizons))


class XGBoostAdapter(_BoostingAdapter):
    name = "xgboost"

    def suggest_config(self, trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

    def _make_estimator(self, config):
        from xgboost import XGBRegressor
        return XGBRegressor(
            **config, tree_method="hist", random_state=RANDOM_SEED,
            n_jobs=-1, importance_type="gain",
        )


class LightGBMAdapter(_BoostingAdapter):
    name = "lightgbm"

    def suggest_config(self, trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

    def _make_estimator(self, config):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            **config, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
            importance_type="gain",
        )


# ---------------------------------------------------------------------------
# Neural-net adapters (LSTM via darts BlockRNN, N-BEATS) — prunable
# ---------------------------------------------------------------------------
class _DartsNeuralAdapter(ModelAdapter):
    """Shared darts plumbing for the two neural models.

    Pruning: this happens at FOLD granularity, not mid-training. After each of
    the 3 CV folds completes, the objective reports the running mean MAE to
    Optuna's MedianPruner (trial.report + should_prune), so a trial that is
    clearly worse than the median after fold 1 or 2 is dropped before running
    the remaining folds. (We do not use an epoch-level Lightning pruning
    callback.) Each model is fit on the complete training span for a fixed
    epoch count, so partially trained timer-stopped models cannot win a study.
    """
    prunable = True

    def _trainer_kwargs(self):
        import torch
        # Default CPU for stability (see NEURAL_DEVICE note at top). Only use an
        # accelerator if the user explicitly opted in AND it's available.
        if NEURAL_DEVICE == "mps" and torch.backends.mps.is_available():
            acc = "mps"
        elif NEURAL_DEVICE == "gpu" and torch.cuda.is_available():
            acc = "gpu"
        else:
            acc = "cpu"
        # enable_progress_bar/model_summary off; logger off to stop the repeated
        # "GPU available / LitLogger tip" spam that showed the trainer was being
        # re-created every fold (a clue to the slowdown). Quiet trainer = less
        # per-fold setup overhead.
        return {"accelerator": acc, "devices": 1, "enable_progress_bar": False,
                "enable_model_summary": False, "logger": False,
                "enable_checkpointing": False}

    def _to_series(self, s):
        from darts import TimeSeries
        return TimeSeries.from_series(s, freq="h").astype(np.float32)

    def _build_model(self, config, trial):
        raise NotImplementedError

    def run_fold(self, config, df, train_start, train_end,
                 eval_start, eval_end, report_cb=None):
        from darts.dataprocessing.transformers import Scaler

        y = df[TARGET_COL]
        train_full = y.loc[train_start:train_end]

        scaler = Scaler()
        train_ts = scaler.fit_transform(self._to_series(train_full))

        model = self._build_model(config, trial=report_cb)
        model.fit(series=train_ts, verbose=False)

        eval_series = y.loc[eval_start:eval_end]
        full = self._to_series(pd.concat([train_full, eval_series]))
        full_scaled = scaler.transform(full)
        eval_start_idx = len(train_full)
        first_origin = eval_start_idx + INPUT_WINDOW

        if not self.final:
            # ---- TUNING: cheap, batched, NON-OVERLAPPING evaluation ----
            # The previous version called historical_forecasts(stride=24) which,
            # on a long concatenated series, could explode to hours per fold and
            # made the wall-clock cap unenforceable (one fold > the cap). Here we
            # instead forecast at NON-OVERLAPPING 24h origins via a single batched
            # predict() call: one forward pass over ~N/24 input windows, no
            # per-origin Python loop, no rolling. This ranks configs reliably and
            # cannot run away — cost is bounded by a single predict().
            # Origin o forecasts full_unscaled[o : o+OUTPUT_HORIZON], so the
            # last valid origin satisfies o + OUTPUT_HORIZON <= len(full_scaled).
            # range() is exclusive on the stop, so stop at len - OUTPUT_HORIZON + 1.
            origins = list(range(first_origin,
                                 len(full_scaled) - OUTPUT_HORIZON + 1,
                                 OUTPUT_HORIZON))
            # Drop any final origin that would overrun (guards odd lengths).
            origins = [o for o in origins
                       if o + OUTPUT_HORIZON <= len(full_scaled)]
            if not origins:
                return FoldResult(np.array([]), np.array([]))
            # Build the list of input-window series, predict them all at once.
            input_series = [full_scaled[o - INPUT_WINDOW:o] for o in origins]
            preds = model.predict(n=OUTPUT_HORIZON, series=input_series,
                                  verbose=False)
            full_unscaled = scaler.inverse_transform(full_scaled).values().flatten()
            y_pred = np.stack([scaler.inverse_transform(p).values().flatten()
                               for p in preds])
            y_true = np.stack([full_unscaled[o:o + OUTPUT_HORIZON]
                               for o in origins])
            return FoldResult(y_true.flatten(), y_pred.flatten())

        # ---- FINAL: honest stride-1 walk-forward (8,593-origin convention) ----
        # Quick smoke-test: cap to the first ~7 days of origins so this is
        # seconds, not the full 8,593-origin pass. Throwaway numbers.
        if QUICK:
            cap = first_origin + 7 * OUTPUT_HORIZON
            full_scaled = full_scaled[:cap + OUTPUT_HORIZON]
        first_ts = full_scaled.time_index[first_origin]
        fcs = model.historical_forecasts(
            series=full_scaled, start=first_ts,
            forecast_horizon=OUTPUT_HORIZON, stride=1, retrain=False,
            last_points_only=False, verbose=False,
        )
        full_unscaled = scaler.inverse_transform(full_scaled).values().flatten()
        y_pred = np.stack([scaler.inverse_transform(f).values().flatten()
                           for f in fcs])
        n_o = len(fcs)
        y_true = np.zeros((n_o, OUTPUT_HORIZON), dtype=np.float32)
        for i in range(n_o):
            s = first_origin + i
            y_true[i, :] = full_unscaled[s:s + OUTPUT_HORIZON]
        return FoldResult(y_true.flatten(), y_pred.flatten())


class LSTMAdapter(_DartsNeuralAdapter):
    name = "lstm"

    def suggest_config(self, trial) -> dict:
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            "n_rnn_layers": trial.suggest_int("n_rnn_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }

    def _build_model(self, config, trial=None):
        from darts.models import BlockRNNModel
        tk = self._trainer_kwargs()
        return BlockRNNModel(
            model="LSTM",
            input_chunk_length=INPUT_WINDOW,
            output_chunk_length=OUTPUT_HORIZON,
            hidden_dim=config["hidden_dim"],
            n_rnn_layers=config["n_rnn_layers"],
            dropout=config["dropout"],
            batch_size=config["batch_size"],
            n_epochs=2 if QUICK else 20,
            optimizer_kwargs={"lr": config["lr"]},
            random_state=RANDOM_SEED,
            pl_trainer_kwargs=tk,
            save_checkpoints=False,
            force_reset=True,
        )


class NBeatsAdapter(_DartsNeuralAdapter):
    name = "nbeats"

    def suggest_config(self, trial) -> dict:
        return {
            "num_stacks": trial.suggest_int("num_stacks", 10, 30, step=10),
            "num_layers": trial.suggest_int("num_layers", 2, 4),
            "layer_widths": trial.suggest_categorical("layer_widths", [256, 512]),
            # ceiling lowered 5e-3 -> 2e-3: lr=0.0044 diverged to ~1e13 on the
            # real run. N-BEATS is sensitive to high lr; keep the search stable.
            "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        }

    def _build_model(self, config, trial=None):
        from darts.models import NBEATSModel
        tk = self._trainer_kwargs()
        return NBEATSModel(
            input_chunk_length=INPUT_WINDOW,
            output_chunk_length=OUTPUT_HORIZON,
            generic_architecture=True,
            num_stacks=config["num_stacks"],
            num_blocks=1,
            num_layers=config["num_layers"],
            layer_widths=config["layer_widths"],
            batch_size=config["batch_size"],
            n_epochs=2 if QUICK else 20,
            optimizer_kwargs={"lr": config["lr"]},
            random_state=RANDOM_SEED,
            pl_trainer_kwargs=tk,
            save_checkpoints=False,
            force_reset=True,
        )


ADAPTERS = {
    "sarima": SarimaAdapter,
    "xgboost": XGBoostAdapter,
    "lightgbm": LightGBMAdapter,
    "lstm": LSTMAdapter,
    "nbeats": NBeatsAdapter,
}


# ---------------------------------------------------------------------------
# The shared tuning protocol
# ---------------------------------------------------------------------------
def make_objective(adapter: ModelAdapter, df: pd.DataFrame):
    """Build an Optuna objective: mean validation MAE across the 3 CV folds.

    Prints a line per fold so long runs are visibly alive — the first version
    of this script was silent for the whole study, which made a slow SARIMA
    trial indistinguishable from a hang.
    """
    def objective(trial):
        config = adapter.suggest_config(trial)
        t0 = time.time()
        print(f"  [{adapter.name}] trial {trial.number}: {config}")
        fold_maes = []
        for fold_i, (tr_s, tr_e, va_s, va_e) in enumerate(CV_FOLDS):
            tf = time.time()
            res = adapter.run_fold(config, df, tr_s, tr_e, va_s, va_e,
                                   report_cb=None)
            if res.y_true.size == 0:
                raise optuna.TrialPruned()
            fold_mae = mae(res.y_true, res.y_pred)
            # Divergence guard: an unstable lr can make training explode (we saw
            # a fold report ~1.5e13 °C). Such a trial is a failed config, not a
            # real (bad) score — prune it so it doesn't waste a slot or skew TPE.
            if not np.isfinite(fold_mae) or fold_mae > 100:
                print(f"    fold {fold_i + 1}/3 (val {va_s[:4]}): "
                      f"DIVERGED (MAE={fold_mae:.1f}) — pruning trial")
                raise optuna.TrialPruned()
            fold_maes.append(fold_mae)
            print(f"    fold {fold_i + 1}/3 (val {va_s[:4]}): "
                  f"MAE={fold_mae:.3f} °C  [{time.time() - tf:.0f}s]")
            # Report intermediate mean for pruning (prunable models benefit most)
            trial.report(float(np.mean(fold_maes)), step=fold_i)
            if trial.should_prune():
                print(f"    PRUNED after fold {fold_i + 1} "
                      f"(mean {np.mean(fold_maes):.3f} not competitive)")
                raise optuna.TrialPruned()
        score = float(np.mean(fold_maes))
        print(f"  [{adapter.name}] trial {trial.number} done: "
              f"mean MAE={score:.3f} °C  [{time.time() - t0:.0f}s total]")
        return score
    return objective


def _progress_callback(t_start: float):
    """Optuna callback: one line per completed trial with the running best."""
    def cb(study, trial):
        elapsed = (time.time() - t_start) / 60
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        best = min(t.value for t in completed) if completed else float("nan")
        print(f"  >>> {study.study_name}: {len(study.trials)} trials done, "
              f"best CV MAE so far {best:.3f} °C  [{elapsed:.1f} min elapsed]")
    return cb


def tune_one(model_key: str, df: pd.DataFrame, n_trials: int) -> dict:
    """Run an Optuna study for one model, return best config + summary."""
    adapter = ADAPTERS[model_key]()
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    pruner = (optuna.pruners.MedianPruner(n_warmup_steps=1)
              if adapter.prunable else optuna.pruners.NopPruner())
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                pruner=pruner, study_name=model_key)
    timeout = TIMEOUT_SECONDS.get(model_key)
    print(f"\n=== Tuning {model_key} ({n_trials} trials, "
          f"{'prunable' if adapter.prunable else 'no pruning'}, "
          f"study limit {timeout // 60} min, checked between trials) ===")

    t_start = time.time()
    objective = make_objective(adapter, df)
    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   callbacks=[_progress_callback(t_start)],
                   show_progress_bar=False)
    if (time.time() - t_start) >= timeout - 5:
        print(f"  NOTE: {model_key} reached its {timeout // 60}-min study limit; "
              f"keeping the {len(study.trials)} trials that completed.")

    STUDIES_DIR.mkdir(parents=True, exist_ok=True)
    with open(STUDIES_DIR / f"{model_key}.pkl", "wb") as fh:
        pickle.dump(study, fh)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"{model_key} produced no completed trials; inspect the saved study"
        )
    n_pruned = sum(1 for t in study.trials
                   if t.state == optuna.trial.TrialState.PRUNED)
    print(f"  best CV MAE: {study.best_value:.3f} °C   "
          f"(pruned {n_pruned}/{len(study.trials)} trials)")
    print(f"  best config: {study.best_params}")
    return {"model": model_key, "best_cv_mae": study.best_value,
            "best_params": study.best_params, "n_trials": len(study.trials),
            "n_pruned": n_pruned}


# ---------------------------------------------------------------------------
# Baselines, evaluated PER-HORIZON on the same origins as the models.
# The series has never measured baselines at individual horizons — Weekend 7
# crowned N-BEATS "best at h=1" without checking whether hourly persistence
# (predict T(t+h) = T(t)) already beats it there. This settles that.
# ---------------------------------------------------------------------------
def evaluate_baselines_per_horizon(df: pd.DataFrame) -> None:
    y = df[TARGET_COL].loc[TEST_START:TEST_END].values
    n = len(y)
    # CANONICAL convention (identical to the models):
    #   origin o, horizon h in 1..24  -> truth = y[o + h - 1]
    #   last observed value before the window = y[o - 1]
    #   persistence(h)    = y[o - 1]              (flat: last seen value)
    #   seasonal_naive(h) = y[o + h - 1 - 24]     (value one day earlier)
    # Bounds: o-1 >= 0 and o+23 <= n-1  -> first=INPUT_WINDOW, last=n-OUTPUT_HORIZON.
    origins = canonical_test_origins(n)          # 8,593 on the 2016 leap year
    H = np.arange(1, OUTPUT_HORIZON + 1)

    # actual matrix: row i, col (h-1) = y[o + h - 1]
    actual = np.stack([y[o:o + OUTPUT_HORIZON] for o in origins])

    PER_HORIZON_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    # Persistence: every horizon = last observed value y[o-1]
    pred_persist = np.repeat(y[origins - 1][:, None], OUTPUT_HORIZON, axis=1)
    # Seasonal naive: horizon h = y[(o + h - 1) - 24]
    pred_snaive = np.stack([y[o - 24:o - 24 + OUTPUT_HORIZON] for o in origins])
    for name, pred in (("persistence", pred_persist),
                       ("seasonal_naive", pred_snaive)):
        per_h = pd.DataFrame({
            "horizon": H,
            "mae": np.mean(np.abs(actual - pred), axis=0),
            "rmse": np.sqrt(np.mean((actual - pred) ** 2, axis=0)),
        })
        per_h.to_csv(PER_HORIZON_DIR / f"{name}_horizon.csv", index=False)
        results[name] = per_h
        print(f"  baseline {name:15s}  h=1: {per_h['mae'].iloc[0]:.3f} °C   "
              f"h=24: {per_h['mae'].iloc[-1]:.3f} °C   "
              f"mean: {per_h['mae'].mean():.3f} °C")


# ---------------------------------------------------------------------------
# Final evaluation on the SEALED test set
# ---------------------------------------------------------------------------
def evaluate_on_test(model_key: str, best_params: dict,
                     df: pd.DataFrame) -> dict:
    """Retrain the best config using pre-2016 data, evaluate once on 2016.

    This is the only place test data is touched. Returns flattened test MAE/RMSE
    plus the per-horizon profile (saved to CSV for the analysis script).
    """
    adapter = ADAPTERS[model_key]()
    adapter.final = True  # full-resolution stride-1 evaluation for every model
    print(f"\n--- Final test evaluation: {model_key} (full resolution) ---")
    # Boosting models use the honest direct multi-horizon path for the final
    # eval so they produce a true 24-wide per-horizon profile. Everything else
    # already returns 24-wide rows from run_fold.
    if isinstance(adapter, _BoostingAdapter):
        if QUICK:
            print("  (QUICK: direct multi-horizon on horizons {1,12,24} only, "
                  "1-yr window — throwaway numbers)")
        else:
            print(f"  (direct multi-horizon: training {OUTPUT_HORIZON} models, "
                  f"one per horizon)")
        res = adapter.run_fold_direct(best_params, df,
                                      CV_FOLDS[0][0], FINAL_TRAIN_END,
                                      TEST_START, TEST_END)
    else:
        res = adapter.run_fold(best_params, df,
                               CV_FOLDS[0][0], FINAL_TRAIN_END,
                               TEST_START, TEST_END, report_cb=None)
    test_mae = mae(res.y_true, res.y_pred)
    test_rmse = rmse(res.y_true, res.y_pred)
    print(f"  {model_key:10s}  test MAE = {test_mae:.3f} °C   "
          f"RMSE = {test_rmse:.3f} °C")

    # Per-horizon profile + RAW residual matrices. The raw matrices are what
    # enable the Diebold-Mariano significance test and the non-overlapping-
    # origin ACF in the analysis script — aggregates alone can't support them.
    PER_HORIZON_DIR.mkdir(parents=True, exist_ok=True)
    width = len(res.horizons)
    n = res.y_true.size
    if n % width == 0 and n >= width:
        yt = res.y_true.reshape(-1, width)
        yp = res.y_pred.reshape(-1, width)
        per_h = pd.DataFrame({
            "horizon": res.horizons,
            "mae": np.mean(np.abs(yt - yp), axis=0),
            "rmse": np.sqrt(np.mean((yt - yp) ** 2, axis=0)),
        })
        per_h.to_csv(PER_HORIZON_DIR / f"{model_key}_tuned_horizon.csv",
                     index=False)
        np.savez_compressed(PER_HORIZON_DIR / f"{model_key}_raw.npz",
                            y_true=yt, y_pred=yp, horizons=res.horizons)
        protocol = "stride-1 origins"
        print(f"  saved per-horizon profile + raw residuals "
              f"[protocol: {protocol}]")
    else:
        print(f"  NOTE: {model_key} eval is single-horizon proxy; "
              f"per-horizon profile skipped (run dedicated direct-h eval).")

    return {"model": model_key, "mae": test_mae, "rmse": test_rmse,
            "protocol": "stride-1", "version": CODE_VERSION}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _merge_json(path, key, value):
    """Read-merge-write a JSON dict so per-process runs don't clobber each
    other. Each model invocation adds/updates only its own key."""
    data = {}
    if path.exists():
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception:
            data = {}
    data[key] = value
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def _merge_leaderboard_row(path, row):
    """Append-or-replace one model's row in the tuned leaderboard CSV, so
    running models in separate processes accumulates instead of overwrites."""
    cols = ["model", "mae", "rmse", "protocol", "version"]
    if path.exists():
        df = pd.read_csv(path)
        df = df[df["model"] != row["model"]]
    else:
        df = pd.DataFrame(columns=cols)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values("mae")
    df.to_csv(path, index=False)
    return df


def main():
    global QUICK, OUTPUT_DIR, PER_HORIZON_DIR, STUDIES_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(ADAPTERS.keys()),
                        choices=list(ADAPTERS.keys()))
    parser.add_argument("--quick", action="store_true",
                        help="tiny trial counts for a smoke test")
    parser.add_argument("--baselines", action="store_true",
                        help="(re)compute the per-horizon baselines and exit")
    args = parser.parse_args()

    budget = QUICK_BUDGET if args.quick else TRIAL_BUDGET
    if args.quick:
        QUICK = True
        OUTPUT_DIR = NOTEBOOK_ROOT / "tuning_weekend_8_quick"
        PER_HORIZON_DIR = OUTPUT_DIR / "per_horizon"
        STUDIES_DIR = OUTPUT_DIR / "optuna_studies"
        print("QUICK smoke-test mode: shrunken fit/eval/epochs — "
              f"artifacts are isolated under {OUTPUT_DIR.name}/.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = OUTPUT_DIR / "best_configs.json"
    lb_path = OUTPUT_DIR / "tuned_leaderboard.csv"

    print("=" * 64)
    print("Weekend 8: Hyperparameter Tuning with Time-Series Cross-Validation")
    print("=" * 64)
    print(f"Models this process: {args.models}")
    print(f"Neural device: {NEURAL_DEVICE}")
    print(f"Folds: 3 expanding-window (val years 2013, 2014, 2015)")
    print(f"Test (2016) sealed until final evaluation.")

    df = load_full()
    print(f"\nLoaded {len(df):,} hourly rows × {df.shape[1]} variables")

    # Baselines-only mode (run once, cheap): the runner calls this first.
    if args.baselines:
        print("\n--- Per-horizon baselines (the check Weekend 7 skipped) ---")
        evaluate_baselines_per_horizon(df)
        print("baselines done.")
        return

    if optuna is None:
        sys.exit("ERROR: optuna is not installed. Run: pip install optuna")

    # ---- Tune + final-eval each model in THIS process ----
    # Designed so the shell runner can invoke one model per process: results
    # merge into shared best_configs.json / tuned_leaderboard.csv rather than
    # overwrite, while framework state is released between fresh processes.
    for model_key in args.models:
        summary = tune_one(model_key, df, budget[model_key])
        _merge_json(cfg_path, model_key, summary["best_params"])
        print(f"  merged best config for {model_key} -> {cfg_path}")

        print("\n" + "=" * 64)
        print(f"FINAL EVALUATION ON SEALED TEST SET (2016): {model_key}")
        print("=" * 64)
        result = evaluate_on_test(model_key, summary["best_params"], df)
        lb = _merge_leaderboard_row(lb_path, result)
        print(f"  merged leaderboard row for {model_key} -> {lb_path}")

        print("\nLeaderboard so far (lower MAE is better):")
        for _, r in lb.iterrows():
            print(f"  {r['model']:10s}  MAE = {r['mae']:.3f} °C   "
                  f"RMSE = {r['rmse']:.3f} °C")

    print("=" * 64)
    print("This process done. When all models + baselines are complete, "
          "run weekend_8_analysis_v2.py for the cross-model plots.")


if __name__ == "__main__":
    main()
