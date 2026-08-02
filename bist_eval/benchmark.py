from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from .adjustments import (
    classify_exposure,
    rebase_context,
    transform_target_after_prediction,
)
from .baselines import forecast_baselines
from .metrics import compute_benchmark_window_metrics
from .model_adapter import CohortPredictionResult, PredictionFailure
from .reporting import BENCHMARK_PREDICTION_COLUMNS, SKIP_COLUMNS
from .windows import build_benchmark_windows


@dataclass(frozen=True, slots=True)
class BenchmarkShardResult:
    predictions: pd.DataFrame
    window_metrics: pd.DataFrame
    skips: pd.DataFrame
    manifest: dict


def _raw_context(raw):
    output = raw.loc[:, ["timestamps", "open", "high", "low", "close", "volume"]].copy()
    output["amount"] = ((output.high + output.low + output.close) / 3) * output.volume
    return output


def _target_fp(target):
    payload = [
        (pd.Timestamp(timestamp).isoformat(), float(close))
        for timestamp, close in zip(target.timestamps, target.close)
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()


def _safe_values(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def _is_valid_positive_prediction(values):
    array = np.asarray(values, dtype=float)
    return bool(np.isfinite(array).all() and (array > 0).all())


def _prediction_rows(
    arm,
    method,
    window,
    prediction,
    actual,
    history_last_close,
    exposure,
    target_fingerprint,
    context_view,
):
    rows = []
    for step, (timestamp, predicted, realized) in enumerate(
        zip(window.target_timestamps, prediction, actual), 1
    ):
        rows.append(
            {
                "experiment_arm": arm,
                "symbol": window.symbol,
                "candidate_month": window.candidate_month,
                "forecast_origin": window.forecast_origin,
                "target_timestamp": timestamp,
                "horizon_step": step,
                "method": method,
                "predicted_close": float(predicted),
                "actual_close": float(realized),
                "history_last_close": float(history_last_close),
                "context_view": context_view,
                "scoring_target_view": "origin_rebased",
                "exposure_bucket": exposure.exposure_bucket,
                "context_factor_changed": exposure.context_factor_changed,
                "target_factor_changed": exposure.target_factor_changed,
                "context_max_abs_log_step": exposure.context_max_abs_log_step,
                "target_max_abs_log_from_origin": exposure.target_max_abs_log_from_origin,
                "common_target_fingerprint": target_fingerprint,
            }
        )
    return rows


def _prepare(raw_frames, cohorts, symbols, lookback, horizon, tolerance):
    groups = defaultdict(list)
    skip_rows = []
    for symbol in symbols:
        bundles, skips = build_benchmark_windows(
            symbol,
            raw_frames[symbol],
            cohorts,
            lookback=lookback,
            horizon=horizon,
        )
        skip_rows.extend(asdict(skip) for skip in skips)
        for bundle in bundles:
            origin_factor = float(bundle.context_provider_factors[-1])
            rebased, repairs = rebase_context(
                bundle.raw_context,
                bundle.context_provider_factors,
                origin_factor,
            )
            raw = _raw_context(bundle.raw_context)
            exposure = classify_exposure(
                bundle.context_provider_factors,
                bundle.scoring_record.target_provider_factors,
                origin_factor,
                tolerance,
            )
            groups[(bundle.forecast_origin, bundle.target_timestamps)].append(
                (bundle, raw, rebased, origin_factor, exposure, repairs)
            )
    return groups, skip_rows


def _predict_with_failures(adapter, windows) -> CohortPredictionResult:
    detailed = getattr(adapter, "predict_cohort_with_failures", None)
    if detailed is not None:
        return detailed(windows)
    return CohortPredictionResult(adapter.predict_cohort(windows), ())


def _failure_skip(failure: PredictionFailure, arm: str, lookback: int, horizon: int):
    return {
        "symbol": failure.symbol,
        "candidate_month": failure.candidate_month,
        "reason_code": failure.reason_code,
        "reason_detail": f"arm={arm}; {failure.reason_detail}",
        "available_history_rows": lookback,
        "available_target_rows": horizon,
    }


def _baseline_failure_skip(bundle, method, prediction, lookback, horizon):
    return {
        "symbol": bundle.symbol,
        "candidate_month": bundle.candidate_month,
        "reason_code": "invalid_baseline_output",
        "reason_detail": (
            f"arm=adjusted-baselines; method={method}; "
            "nonpositive_or_nonfinite_close="
            + json.dumps(_safe_values(np.asarray(prediction, dtype=float)), separators=(",", ":"))
        ),
        "available_history_rows": lookback,
        "available_target_rows": horizon,
    }


def run_mini_pair_shard(
    *, raw_frames, cohorts, symbols, config, adapter, manifest_base=None
):
    groups, skip_rows = _prepare(
        raw_frames,
        cohorts,
        symbols,
        config.lookback,
        config.horizon,
        config.material_factor_tolerance,
    )
    rows = []
    invalid_counts = {"raw-mini": 0, "adjusted-mini": 0}
    invalid_baselines = defaultdict(int)

    for _, items in sorted(groups.items(), key=lambda item: item[0][0]):
        raw_windows = [
            bundle.prediction_window(raw)
            for bundle, raw, rebased, origin, exposure, repairs in items
        ]
        adjusted_windows = [
            bundle.prediction_window(rebased)
            for bundle, raw, rebased, origin, exposure, repairs in items
        ]
        raw_result = _predict_with_failures(adapter, raw_windows)
        adjusted_result = _predict_with_failures(adapter, adjusted_windows)

        for failure in raw_result.failures:
            skip_rows.append(
                _failure_skip(failure, "raw-mini", config.lookback, config.horizon)
            )
            if failure.reason_code == "invalid_model_output":
                invalid_counts["raw-mini"] += 1
        for failure in adjusted_result.failures:
            skip_rows.append(
                _failure_skip(
                    failure, "adjusted-mini", config.lookback, config.horizon
                )
            )
            if failure.reason_code == "invalid_model_output":
                invalid_counts["adjusted-mini"] += 1

        for bundle, raw, rebased, origin_factor, exposure, _ in items:
            actual_frame = transform_target_after_prediction(
                bundle.scoring_record.raw_target,
                bundle.scoring_record.target_provider_factors,
                origin_factor,
            )
            actual = actual_frame.close.to_numpy(float)
            history_last_close = float(raw.close.iloc[-1])
            target_fingerprint = _target_fp(actual_frame)

            if bundle.symbol in raw_result.predictions:
                rows.extend(
                    _prediction_rows(
                        "raw-mini",
                        "kronos",
                        bundle,
                        raw_result.predictions[bundle.symbol],
                        actual,
                        history_last_close,
                        exposure,
                        target_fingerprint,
                        "raw",
                    )
                )
            if bundle.symbol in adjusted_result.predictions:
                rows.extend(
                    _prediction_rows(
                        "adjusted-mini",
                        "kronos",
                        bundle,
                        adjusted_result.predictions[bundle.symbol],
                        actual,
                        history_last_close,
                        exposure,
                        target_fingerprint,
                        "origin_rebased",
                    )
                )
            for method, prediction in forecast_baselines(
                rebased, config.horizon
            ).items():
                if not _is_valid_positive_prediction(prediction):
                    skip_rows.append(
                        _baseline_failure_skip(
                            bundle,
                            method,
                            prediction,
                            config.lookback,
                            config.horizon,
                        )
                    )
                    invalid_baselines[method] += 1
                    continue
                rows.extend(
                    _prediction_rows(
                        "adjusted-baselines",
                        method,
                        bundle,
                        prediction,
                        actual,
                        history_last_close,
                        exposure,
                        target_fingerprint,
                        "origin_rebased",
                    )
                )

    predictions = pd.DataFrame(rows, columns=BENCHMARK_PREDICTION_COLUMNS)
    window_metrics = (
        compute_benchmark_window_metrics(predictions)
        if len(predictions)
        else pd.DataFrame()
    )
    skips = pd.DataFrame(skip_rows, columns=SKIP_COLUMNS)
    manifest = {
        **(manifest_base or {}),
        "mode": "mini-pair",
        "experiment_arms": ["raw-mini", "adjusted-mini", "adjusted-baselines"],
        "symbols": list(symbols),
        "prediction_rows": len(predictions),
        "eligible_windows": int(
            (
                (window_metrics.experiment_arm == "raw-mini")
                & (window_metrics.method == "kronos")
            ).sum()
        )
        if len(window_metrics)
        else 0,
        "invalid_model_outputs": invalid_counts,
        "invalid_baseline_outputs": dict(sorted(invalid_baselines.items())),
    }
    return BenchmarkShardResult(predictions, window_metrics, skips, manifest)


def run_small_shard(
    *, raw_frames, cohorts, symbols, config, adapter, manifest_base=None
):
    groups, skip_rows = _prepare(
        raw_frames,
        cohorts,
        symbols,
        config.lookback,
        config.horizon,
        config.material_factor_tolerance,
    )
    rows = []
    invalid_count = 0

    for _, items in sorted(groups.items(), key=lambda item: item[0][0]):
        windows = [
            bundle.prediction_window(rebased)
            for bundle, raw, rebased, origin, exposure, repairs in items
        ]
        result = _predict_with_failures(adapter, windows)
        for failure in result.failures:
            skip_rows.append(
                _failure_skip(
                    failure, "adjusted-small", config.lookback, config.horizon
                )
            )
            if failure.reason_code == "invalid_model_output":
                invalid_count += 1

        for bundle, raw, rebased, origin_factor, exposure, _ in items:
            if bundle.symbol not in result.predictions:
                continue
            actual_frame = transform_target_after_prediction(
                bundle.scoring_record.raw_target,
                bundle.scoring_record.target_provider_factors,
                origin_factor,
            )
            actual = actual_frame.close.to_numpy(float)
            history_last_close = float(raw.close.iloc[-1])
            target_fingerprint = _target_fp(actual_frame)
            rows.extend(
                _prediction_rows(
                    "adjusted-small",
                    "kronos",
                    bundle,
                    result.predictions[bundle.symbol],
                    actual,
                    history_last_close,
                    exposure,
                    target_fingerprint,
                    "origin_rebased",
                )
            )

    predictions = pd.DataFrame(rows, columns=BENCHMARK_PREDICTION_COLUMNS)
    window_metrics = (
        compute_benchmark_window_metrics(predictions)
        if len(predictions)
        else pd.DataFrame()
    )
    skips = pd.DataFrame(skip_rows, columns=SKIP_COLUMNS)
    manifest = {
        **(manifest_base or {}),
        "mode": "small",
        "experiment_arms": ["adjusted-small"],
        "symbols": list(symbols),
        "prediction_rows": len(predictions),
        "eligible_windows": int((window_metrics.method == "kronos").sum())
        if len(window_metrics)
        else 0,
        "invalid_model_outputs": {"adjusted-small": invalid_count},
    }
    return BenchmarkShardResult(predictions, window_metrics, skips, manifest)
