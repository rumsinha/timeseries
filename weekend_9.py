"""
Weekend 9: Probabilistic Forecasting — Quantile Regression vs Split Conformal.

This script is optimized for a laptop run:

  - XGBoost runs at the full Weekend 8 stride-1 convention by default.
  - N-BEATS is opt-in because full stride-1 probabilistic prediction is very
    expensive in Darts. The default N-BEATS mode evaluates every 24th origin;
    use --full-neural for the exact stride-1 run.
  - Results are checkpointed after each model, so a slow neural run never loses
    completed XGBoost artifacts.

Outputs:
    probabilistic_weekend_9/coverage_summary.csv
    probabilistic_weekend_9/reliability_curve.csv
    probabilistic_weekend_9/raw/{model}_{method}.npz

Examples:
    python weekend_9.py
    python weekend_9.py --models xgboost nbeats
    WEEKEND9_NEURAL_DEVICE=mps python weekend_9.py --models nbeats
    WEEKEND9_NEURAL_DEVICE=mps python weekend_9.py --models nbeats --full-neural
    python weekend_9.py --quick --models xgboost nbeats
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
_data_marker = "data/jena_climate_2009_2016.csv"
DATA_ROOT = SCRIPT_DIR
while not (DATA_ROOT / _data_marker).exists() and DATA_ROOT.parent != DATA_ROOT:
    DATA_ROOT = DATA_ROOT.parent
DATA_PATH = DATA_ROOT / "data" / "jena_climate_2009_2016.csv"
OUTPUT_DIR = SCRIPT_DIR / "probabilistic_weekend_9"
RAW_DIR = OUTPUT_DIR / "raw"


# ---------------------------------------------------------------------------
# Locked series spec
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
CODE_VERSION = "v2"
TARGET_COL = "T (degC)"
INPUT_WINDOW = 168
OUTPUT_HORIZON = 24

TRAIN_START = "2009-01-01 00:00:00"
TRAIN_END = "2014-12-31 23:00:00"
CALIB_START = "2015-01-01 00:00:00"
CALIB_END = "2015-12-31 23:00:00"
TEST_START = "2016-01-01 00:00:00"
TEST_END = "2016-12-31 23:00:00"

ALPHA_LEVELS = [0.05, 0.1, 0.2, 0.3, 0.5]
HEADLINE_ALPHA = 0.2

QUICK = False
NEURAL_DEVICE = os.environ.get("WEEKEND9_NEURAL_DEVICE", "cpu").lower()
np.random.seed(RANDOM_SEED)


@dataclass
class RunConfig:
    neural_origin_stride: int = 24
    neural_samples: int = 100
    neural_quantiles: bool = True
    neural_epochs: int = 10
    xgb_trees: int = 300


@dataclass
class Predictions:
    point: np.ndarray
    truth: np.ndarray
    origin_offsets: np.ndarray
    origin_stride: int
    qr_bands: dict | None = None


def log(msg: str) -> None:
    print(msg, flush=True)


def elapsed(t0: float) -> str:
    return f"{time.time() - t0:.1f}s"


def load_full() -> pd.DataFrame:
    if not DATA_PATH.exists():
        sys.exit(f"\nERROR: data not found at {DATA_PATH}\n")
    df = pd.read_csv(DATA_PATH)
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.set_index("Date Time").resample("1h").mean()
    return df.asfreq("h").interpolate("linear")


def canonical_origins(n_rows: int) -> np.ndarray:
    first, last = INPUT_WINDOW, n_rows - OUTPUT_HORIZON
    if last < first:
        return np.array([], dtype=int)
    return np.arange(first, last + 1, dtype=int)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    t = df[TARGET_COL]
    for lag in (1, 2, 3, 24, 48, 168):
        out[f"t_lag_{lag}"] = t.shift(lag)
    for win in (3, 24):
        out[f"t_rollmean_{win}"] = t.shift(1).rolling(win).mean()
        out[f"t_rollstd_{win}"] = t.shift(1).rolling(win).std()
    for col in ("VPmax (mbar)", "rh (%)", "p (mbar)"):
        if col in df.columns:
            out[f"{col}_lag1"] = df[col].shift(1)
    hour, doy = df.index.hour, df.index.dayofyear
    out["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    out["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    out["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    out["target"] = t
    return out


def coverage_levels() -> list[float]:
    return [1 - a for a in ALPHA_LEVELS]


def quantile_pairs(levels: list[float]) -> tuple[set[float], dict[float, tuple[float, float]]]:
    need_q: set[float] = set()
    pair_for: dict[float, tuple[float, float]] = {}
    for cov in levels:
        a = round((1 - cov) / 2, 4)
        b = round(1 - a, 4)
        pair_for[cov] = (a, b)
        need_q.update((a, b))
    return need_q, pair_for


class XGBoostQuantile:
    name = "xgboost"

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.params = dict(
            n_estimators=60 if QUICK else cfg.xgb_trees,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        self._fit_cache = {}

    def _fit_one(self, X, y, alpha=None, cache_key=None):
        from xgboost import XGBRegressor

        if cache_key is not None and cache_key in self._fit_cache:
            return self._fit_cache[cache_key]
        if alpha is None:
            model = XGBRegressor(**self.params)
        else:
            model = XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=alpha,
                **self.params,
            )
        model.fit(X, y)
        if cache_key is not None:
            self._fit_cache[cache_key] = model
        return model

    def predict_span(
        self,
        df,
        fit_start,
        fit_end,
        span_start,
        span_end,
        levels=None,
    ) -> Predictions:
        t0 = time.time()
        feats = build_features(df)
        feature_cols = [c for c in feats.columns if c != "target"]
        tgt_all = feats["target"]

        span_idx = feats.loc[span_start:span_end].index
        origins = canonical_origins(len(span_idx))
        if QUICK:
            origins = origins[: 7 * OUTPUT_HORIZON]
        origin_ts = span_idx[origins]
        X_origin_full = feats.loc[origin_ts, feature_cols]
        valid = ~X_origin_full.isna().any(axis=1).values
        origins = origins[valid]
        X_origin = X_origin_full.values[valid]
        span_y = feats.loc[span_start:span_end, "target"].values

        levels = levels or []
        need_q, pair_for = quantile_pairs(levels)
        horizons = [1, 12, 24] if QUICK else list(range(1, OUTPUT_HORIZON + 1))
        log(f"    xgboost span {span_start[:4]}: {len(origins):,} origins, "
            f"{len(horizons)} horizons, {len(need_q)} quantiles")

        point_cols, true_cols = [], []
        qpred = {q: [] for q in need_q}
        for i, h in enumerate(horizons, start=1):
            off = h - 1
            tgt = tgt_all.shift(-off) if off else tgt_all
            cutoff = pd.Timestamp(fit_end) - pd.Timedelta(hours=off)
            tr = feats.loc[fit_start:cutoff].assign(
                _y=tgt.loc[fit_start:cutoff]
            ).dropna()
            Xtr, ytr = tr[feature_cols].values, tr["_y"].values
            point = self._fit_one(
                Xtr, ytr, cache_key=(fit_start, fit_end, h, "point")
            ).predict(X_origin)
            point_cols.append(point)
            for q in need_q:
                qpred[q].append(
                    self._fit_one(
                        Xtr, ytr, alpha=q, cache_key=(fit_start, fit_end, h, q)
                    ).predict(X_origin)
                )
            true_cols.append(np.array([span_y[o + off] for o in origins]))
            if i == len(horizons) or i % 6 == 0:
                log(f"      horizon {i}/{len(horizons)} done [{elapsed(t0)}]")

        point = np.stack(point_cols, axis=1)
        truth = np.stack(true_cols, axis=1)
        qr_bands = None
        if levels:
            qmat = {q: np.stack(v, axis=1) for q, v in qpred.items()}
            qr_bands = {
                cov: ordered_interval(qmat[pair_for[cov][0]], qmat[pair_for[cov][1]])
                for cov in levels
            }
        return Predictions(point, truth, origins, 1, qr_bands)


class NBeatsQuantile:
    name = "nbeats"

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.n_epochs = 2 if QUICK else cfg.neural_epochs
        self._fit_cache = {}

    def _trainer_kwargs(self):
        import torch

        if NEURAL_DEVICE == "mps" and torch.backends.mps.is_available():
            acc = "mps"
        elif NEURAL_DEVICE == "gpu" and torch.cuda.is_available():
            acc = "gpu"
        else:
            acc = "cpu"
        return {
            "accelerator": acc,
            "devices": 1,
            "enable_progress_bar": True,
            "enable_model_summary": False,
            "logger": False,
            "enable_checkpointing": False,
        }

    def _series(self, s):
        from darts import TimeSeries

        return TimeSeries.from_series(s, freq="h").astype(np.float32)

    def _build(self, likelihood=None):
        from darts.models import NBEATSModel

        return NBEATSModel(
            input_chunk_length=INPUT_WINDOW,
            output_chunk_length=OUTPUT_HORIZON,
            generic_architecture=True,
            num_stacks=10 if QUICK else 20,
            num_blocks=1,
            num_layers=3,
            layer_widths=128 if QUICK else 256,
            n_epochs=self.n_epochs,
            batch_size=256,
            optimizer_kwargs={"lr": 1e-3},
            random_state=RANDOM_SEED,
            pl_trainer_kwargs=self._trainer_kwargs(),
            likelihood=likelihood,
            force_reset=True,
            save_checkpoints=False,
        )

    def _point_model(self, train_full):
        from darts.dataprocessing.transformers import Scaler

        cache_key = ("point", str(train_full.index[0]), str(train_full.index[-1]))
        if cache_key in self._fit_cache:
            return self._fit_cache[cache_key]
        scaler = Scaler()
        train_ts = scaler.fit_transform(self._series(train_full))
        model = self._build(likelihood=None)
        log(f"    nbeats point fit: {self.n_epochs} epochs on {len(train_full):,} hours")
        model.fit(train_ts, verbose=True)
        self._fit_cache[cache_key] = (scaler, model, train_ts)
        return scaler, model, train_ts

    def _quantile_model(self, train_ts, levels):
        from darts.utils.likelihood_models import QuantileRegression

        qset = sorted(
            {0.5}
            | {round((1 - c) / 2, 4) for c in levels}
            | {round(1 - (1 - c) / 2, 4) for c in levels}
        )
        cache_key = ("quantile", tuple(qset), self.n_epochs)
        if cache_key in self._fit_cache:
            return self._fit_cache[cache_key]
        model = self._build(likelihood=QuantileRegression(quantiles=qset))
        log(f"    nbeats quantile fit: {self.n_epochs} epochs, quantiles={qset}")
        model.fit(train_ts, verbose=True)
        self._fit_cache[cache_key] = model
        return model

    def predict_span(
        self,
        df,
        fit_start,
        fit_end,
        span_start,
        span_end,
        levels=None,
    ) -> Predictions:
        y = df[TARGET_COL]
        train_full = y.loc[fit_start:fit_end]
        scaler, point_model, train_ts = self._point_model(train_full)

        # Darts TimeSeries must be continuous. For the 2016 test span, using
        # train + test directly would skip all of 2015 and Darts would fill that
        # gap with NaNs. We may use 2015 as observed history/context for 2016
        # forecasts; the model was still fit only on 2009-2014.
        full_observed = y.loc[fit_start:span_end]
        full = self._series(full_observed)
        full_scaled = scaler.transform(full)
        full_unscaled = scaler.inverse_transform(full_scaled).values().flatten()

        stride = 24 if QUICK else max(1, self.cfg.neural_origin_stride)
        span_index = full_observed.loc[span_start:span_end].index
        span_start_pos = full_observed.index.get_loc(span_index[0])
        span_origins = canonical_origins(len(span_index))
        origins = span_start_pos + span_origins[::stride]
        if QUICK:
            origins = origins[: 7 * OUTPUT_HORIZON]

        n_full_stride = len(span_origins)
        log(f"    nbeats span {span_start[:4]}: {len(origins):,} origins "
            f"(stride={stride}; full stride-1 would be {n_full_stride:,})")

        input_series = [full_scaled[o - INPUT_WINDOW:o] for o in origins]
        log(f"    nbeats point predict: {len(input_series):,} windows x 24 horizons")
        point_pred = point_model.predict(
            n=OUTPUT_HORIZON, series=input_series, verbose=True
        )
        point = np.stack([
            scaler.inverse_transform(p).values().flatten() for p in point_pred
        ])
        truth = np.stack([full_unscaled[o:o + OUTPUT_HORIZON] for o in origins])

        qr_bands = None
        levels = levels or []
        if levels and self.cfg.neural_quantiles:
            qmodel = self._quantile_model(train_ts, levels)
            log(f"    nbeats quantile predict: {len(input_series):,} windows, "
                f"{self.cfg.neural_samples} samples/window")
            samples = qmodel.predict(
                n=OUTPUT_HORIZON,
                series=input_series,
                num_samples=self.cfg.neural_samples,
                verbose=True,
            )
            samples = [scaler.inverse_transform(p) for p in samples]
            qr_bands = {}
            for cov in levels:
                a = round((1 - cov) / 2, 4)
                b = round(1 - a, 4)
                lo = np.stack([p.quantile(a).values().flatten() for p in samples])
                hi = np.stack([p.quantile(b).values().flatten() for p in samples])
                qr_bands[cov] = ordered_interval(lo, hi)
        return Predictions(point, truth, origins - span_start_pos, stride, qr_bands)


ADAPTERS = {"xgboost": XGBoostQuantile, "nbeats": NBeatsQuantile}


def conformal_halfwidths(calib: Predictions, alpha: float) -> np.ndarray:
    resid = np.abs(calib.truth - calib.point)
    n = resid.shape[0]
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        log(f"    WARN: conformal k={k} > n={n} at alpha={alpha}; "
            "using max residual as the widest bounded interval.")
        k = n
    return np.sort(resid, axis=0)[k - 1]


def apply_conformal(test: Predictions, halfwidths: np.ndarray):
    return test.point - halfwidths[None, :], test.point + halfwidths[None, :]


def ordered_interval(lower, upper):
    return np.minimum(lower, upper), np.maximum(lower, upper)


def empirical_coverage(truth, lower, upper) -> np.ndarray:
    return ((truth >= lower) & (truth <= upper)).mean(axis=0)


def mean_width(lower, upper) -> np.ndarray:
    return (upper - lower).mean(axis=0)


def _summary_rows(model, method, alpha, pred: Predictions, cov, width):
    return [
        {
            "model": model,
            "method": method,
            "target_coverage": 1 - alpha,
            "horizon": h + 1,
            "empirical_coverage": float(cov[h]),
            "mean_width": float(width[h]),
            "n_origins": int(pred.truth.shape[0]),
            "origin_stride": int(pred.origin_stride),
            "version": CODE_VERSION,
        }
        for h in range(pred.truth.shape[1])
    ]


def _save_raw(model, method, pred: Predictions, lower, upper):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RAW_DIR / f"{model}_{method}.npz",
        lower=lower,
        upper=upper,
        truth=pred.truth,
        point=pred.point,
        origin_offsets=pred.origin_offsets,
        origin_stride=np.array(pred.origin_stride),
        version=np.array(CODE_VERSION),
    )


def evaluate_model(model_key: str, df: pd.DataFrame, cfg: RunConfig) -> dict:
    t0 = time.time()
    adapter = ADAPTERS[model_key](cfg)
    levels = coverage_levels()

    log(f"\n{'=' * 64}")
    log(f"{model_key}: fitting on TRAIN (2009-2014)")
    log(f"{'=' * 64}")
    log("  predicting calibration year (2015)...")
    calib = adapter.predict_span(
        df, TRAIN_START, TRAIN_END, CALIB_START, CALIB_END, levels=None
    )
    log("  predicting test year (2016) + native intervals...")
    test = adapter.predict_span(
        df, TRAIN_START, TRAIN_END, TEST_START, TEST_END, levels=levels
    )

    rows, reliability = [], []
    if test.qr_bands:
        for cov in levels:
            lo, hi = test.qr_bands[cov]
            cqr = empirical_coverage(test.truth, lo, hi)
            wqr = mean_width(lo, hi)
            reliability.append((cov, float(cqr.mean()), "quantile_regression"))
            if abs(cov - (1 - HEADLINE_ALPHA)) < 1e-9:
                _save_raw(model_key, "quantile", test, lo, hi)
                rows += _summary_rows(
                    model_key, "quantile_regression", 1 - cov, test, cqr, wqr
                )
                log(f"  quantile-regression 80% interval: mean coverage "
                    f"{cqr.mean():.3f}  mean width {wqr.mean():.2f}")
    else:
        log("  native quantile intervals skipped for this model/run.")

    for alpha in ALPHA_LEVELS:
        hw = conformal_halfwidths(calib, alpha)
        lo, hi = apply_conformal(test, hw)
        cov_emp = empirical_coverage(test.truth, lo, hi)
        w = mean_width(lo, hi)
        reliability.append((1 - alpha, float(cov_emp.mean()), "conformal"))
        if abs(alpha - HEADLINE_ALPHA) < 1e-9:
            _save_raw(model_key, "conformal", test, lo, hi)
            rows += _summary_rows(model_key, "conformal", alpha, test, cov_emp, w)
            log(f"  conformal 80% interval:           mean coverage "
                f"{cov_emp.mean():.3f}  mean width {w.mean():.2f}")

    log(f"  {model_key} done [{elapsed(t0)}]")
    return {"summary": rows, "reliability": reliability, "model": model_key}


def load_existing_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def checkpoint_model_result(result: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "coverage_summary.csv"
    rel_path = OUTPUT_DIR / "reliability_curve.csv"

    summary = pd.DataFrame(result["summary"])
    rel = pd.DataFrame([
        {
            "model": result["model"],
            "method": method,
            "target_coverage": target,
            "empirical_coverage": emp,
            "version": CODE_VERSION,
        }
        for target, emp, method in result["reliability"]
    ])

    old_summary = load_existing_csv(summary_path)
    if not old_summary.empty:
        old_summary = old_summary[old_summary["model"] != result["model"]]
        summary = pd.concat([old_summary, summary], ignore_index=True)
    old_rel = load_existing_csv(rel_path)
    if not old_rel.empty:
        old_rel = old_rel[old_rel["model"] != result["model"]]
        rel = pd.concat([old_rel, rel], ignore_index=True)

    summary.to_csv(summary_path, index=False)
    rel.to_csv(rel_path, index=False)
    log(f"  checkpointed CSVs -> {OUTPUT_DIR}")


def validate_outputs(models: list[str]) -> None:
    summary_path = OUTPUT_DIR / "coverage_summary.csv"
    rel_path = OUTPUT_DIR / "reliability_curve.csv"
    if not summary_path.exists() or not rel_path.exists():
        raise RuntimeError("missing Weekend 9 summary CSV outputs")
    summary = pd.read_csv(summary_path)
    rel = pd.read_csv(rel_path)
    for model in models:
        if model not in set(summary["model"]):
            raise RuntimeError(f"{model} missing from coverage_summary.csv")
        if model not in set(rel["model"]):
            raise RuntimeError(f"{model} missing from reliability_curve.csv")
        expected_methods = set(summary.loc[summary["model"] == model, "method"])
        raw_for_method = {
            "conformal": "conformal",
            "quantile_regression": "quantile",
        }
        for method_name, raw_suffix in raw_for_method.items():
            if method_name not in expected_methods:
                continue
            p = RAW_DIR / f"{model}_{raw_suffix}.npz"
            if not p.exists():
                raise RuntimeError(f"missing raw artifact {p}")
            data = np.load(p)
            shapes = {k: data[k].shape for k in ("lower", "upper", "truth")}
            if len(set(shapes.values())) != 1:
                raise RuntimeError(f"shape mismatch in {p}: {shapes}")
            for key in ("lower", "upper", "truth", "point"):
                if not np.isfinite(data[key]).all():
                    raise RuntimeError(f"non-finite values found in {p}:{key}")
            if (data["upper"] < data["lower"]).any():
                raise RuntimeError(f"interval crossing found in {p}")
    log("Validation passed: CSVs + raw interval arrays are internally consistent.")


def main():
    global QUICK, OUTPUT_DIR, RAW_DIR

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["xgboost"],
        choices=list(ADAPTERS),
        help="Default is xgboost. Add nbeats explicitly for the neural run.",
    )
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test run")
    ap.add_argument(
        "--full-neural",
        action="store_true",
        help="N-BEATS stride-1, 20 epochs, 300 samples. Can take many hours.",
    )
    ap.add_argument(
        "--nbeats-origin-stride",
        type=int,
        default=24,
        help="N-BEATS origin stride. 24 is the laptop-friendly default.",
    )
    ap.add_argument("--nbeats-samples", type=int, default=100)
    ap.add_argument("--nbeats-epochs", type=int, default=10)
    ap.add_argument(
        "--skip-nbeats-quantiles",
        action="store_true",
        help="Run N-BEATS conformal only; much faster than probabilistic sampling.",
    )
    ap.add_argument("--xgb-trees", type=int, default=300)
    args = ap.parse_args()
    QUICK = args.quick
    if QUICK:
        OUTPUT_DIR = SCRIPT_DIR / "probabilistic_weekend_9_quick"
        RAW_DIR = OUTPUT_DIR / "raw"

    cfg = RunConfig(
        neural_origin_stride=1 if args.full_neural else args.nbeats_origin_stride,
        neural_samples=300 if args.full_neural else args.nbeats_samples,
        neural_quantiles=not args.skip_nbeats_quantiles,
        neural_epochs=20 if args.full_neural else args.nbeats_epochs,
        xgb_trees=args.xgb_trees,
    )
    if QUICK:
        cfg = RunConfig(
            neural_origin_stride=24,
            neural_samples=25,
            neural_quantiles=not args.skip_nbeats_quantiles,
            neural_epochs=2,
            xgb_trees=60,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 64)
    log("Weekend 9: Probabilistic Forecasting — Conformal vs Quantile")
    log("=" * 64)
    log(f"TRAIN {TRAIN_START[:4]}-{TRAIN_END[:4]}  "
        f"CALIB {CALIB_START[:4]}  TEST {TEST_START[:4]}")
    log(f"Models: {args.models}   device(neural): {NEURAL_DEVICE}"
        + ("   [QUICK]" if QUICK else ""))
    if "nbeats" in args.models:
        log(f"N-BEATS config: stride={cfg.neural_origin_stride}, "
            f"epochs={cfg.neural_epochs}, samples={cfg.neural_samples}, "
            f"quantiles={cfg.neural_quantiles}")
        if not args.full_neural and not QUICK:
            log("NOTE: N-BEATS default is sampled-origin for runtime. "
                "Use --full-neural for exact stride-1.")

    df = load_full()
    log(f"Loaded {len(df):,} hourly rows")

    completed = []
    for key in args.models:
        result = evaluate_model(key, df, cfg)
        checkpoint_model_result(result)
        completed.append(key)

    validate_outputs(completed)
    log(f"\nDone. Outputs saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
