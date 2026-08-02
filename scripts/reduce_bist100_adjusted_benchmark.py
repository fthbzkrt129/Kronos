#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bist_eval.comparison import (
    DEFAULT_COMPARISONS,
    build_paired_comparison,
    summarize_comparison_intervals,
)
from bist_eval.metrics import (
    aggregate_benchmark_period_metrics,
    aggregate_benchmark_symbol_metrics,
    compute_benchmark_ranking_metrics,
)
from bist_eval.reporting import (
    validate_benchmark_prediction_keys,
    write_benchmark_reduced_output,
)
from bist_eval.sharding import validate_benchmark_shard_manifests


def _read_shards(root):
    directories = sorted([path for path in Path(root).iterdir() if path.is_dir()])
    records = []
    for directory in directories:
        if not (directory / "COMPLETED").is_file():
            raise ValueError(f"shard missing COMPLETED: {directory}")
        if not (directory / "shard_manifest.json").is_file():
            continue
        manifest = json.loads((directory / "shard_manifest.json").read_text())
        records.append((manifest, directory))
    records.sort(key=lambda item: item[0]["shard_index"])
    return records


def _reason_counts(skips: pd.DataFrame, reason_code: str, pattern: str) -> dict[str, int]:
    if skips.empty or "reason_code" not in skips.columns:
        return {}
    invalid = skips.loc[skips.reason_code == reason_code]
    if invalid.empty:
        return {}
    labels = invalid.reason_detail.astype(str).str.extract(pattern, expand=False)
    labels = labels.dropna()
    return {str(label): int(count) for label, count in labels.value_counts().items()}


def _invalid_model_output_counts(skips: pd.DataFrame) -> dict[str, int]:
    return _reason_counts(skips, "invalid_model_output", r"arm=([^;]+)")


def _invalid_baseline_output_counts(skips: pd.DataFrame) -> dict[str, int]:
    return _reason_counts(skips, "invalid_baseline_output", r"method=([^;]+)")


def run_adjusted_reducer(
    *,
    mini_shards_dir,
    small_shards_dir,
    expected_shards,
    factor_manifest,
    output_dir,
    minimum_ranking_cohort=20,
    bootstrap_draws=10000,
    bootstrap_confidence=0.95,
    bootstrap_seed=20260802,
):
    mini = _read_shards(mini_shards_dir)
    small = _read_shards(small_shards_dir)
    validate_benchmark_shard_manifests(
        [manifest for manifest, directory in mini],
        [manifest for manifest, directory in small],
        expected_shards,
    )

    predictions = []
    window_metrics = []
    skips = []
    for manifest, directory in [*mini, *small]:
        predictions.append(pd.read_csv(directory / "predictions.csv"))
        window_metrics.append(pd.read_csv(directory / "window_metrics.csv"))
        skips.append(pd.read_csv(directory / "skipped_windows.csv"))

    prediction_frame = pd.concat(predictions, ignore_index=True)
    metric_frame = pd.concat(window_metrics, ignore_index=True)
    skip_frame = pd.concat(skips, ignore_index=True)
    validate_benchmark_prediction_keys(prediction_frame)

    for column in ("forecast_origin", "target_timestamp"):
        prediction_frame[column] = pd.to_datetime(prediction_frame[column])
    metric_frame.forecast_origin = pd.to_datetime(metric_frame.forecast_origin)

    symbol_metrics = aggregate_benchmark_symbol_metrics(metric_frame)
    period_metrics = aggregate_benchmark_period_metrics(metric_frame)
    ranking_metrics = compute_benchmark_ranking_metrics(
        metric_frame, minimum_ranking_cohort
    )

    paired_parts = []
    interval_parts = []
    decisions = {}
    for specification in DEFAULT_COMPARISONS:
        paired = build_paired_comparison(metric_frame, specification)
        if len(paired) == 0:
            decisions[specification.comparison_id] = "unavailable"
            continue
        paired_parts.append(paired)
        intervals = summarize_comparison_intervals(
            paired,
            draws=bootstrap_draws,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
        )
        interval_parts.append(intervals)
        decisions[specification.comparison_id] = intervals.decision.iloc[0]

    paired_all = (
        pd.concat(paired_parts, ignore_index=True) if paired_parts else pd.DataFrame()
    )
    interval_all = (
        pd.concat(interval_parts, ignore_index=True)
        if interval_parts
        else pd.DataFrame()
    )

    arm_summary = (
        metric_frame.groupby(["experiment_arm", "method"], as_index=False)
        .agg(
            windows=("symbol", "size"),
            mean_log_return_abs_error=("log_return_abs_error", "mean"),
            median_log_return_abs_error=("log_return_abs_error", "median"),
            direction_accuracy=("direction_correct", "mean"),
        )
    )
    invalid_model_counts = _invalid_model_output_counts(skip_frame)
    invalid_baseline_counts = _invalid_baseline_output_counts(skip_frame)
    requested_symbols = [
        symbol
        for manifest, directory in mini
        for symbol in manifest.get("symbols", [])
    ]
    evaluated_symbols = sorted(set(metric_frame.symbol))
    questions = {
        "Does adjusted data improve Kronos-mini?": decisions.get(
            "adjusted-mini_vs_raw-mini", "unavailable"
        ),
        "Does Kronos-small outperform adjusted Kronos-mini?": decisions.get(
            "adjusted-small_vs_adjusted-mini", "unavailable"
        ),
        "Does adjusted Kronos-mini beat last-close?": decisions.get(
            "adjusted-mini_vs_last-close", "unavailable"
        ),
        "Does adjusted Kronos-small beat last-close?": decisions.get(
            "adjusted-small_vs_last-close", "unavailable"
        ),
        "Is cross-sectional ranking reliable?": "review ranking_metrics.csv",
    }
    summary = {
        "prediction_rows": len(prediction_frame),
        "eligible_model_windows": int(
            (
                (metric_frame.method == "kronos")
                & metric_frame.experiment_arm.isin(
                    ["raw-mini", "adjusted-mini", "adjusted-small"]
                )
            ).sum()
        ),
        "symbols_requested": len(requested_symbols),
        "symbols_evaluated": len(evaluated_symbols),
        "questions": questions,
        "arm_metrics": arm_summary.to_dict("records"),
        "comparison_decisions": decisions,
        "exposure_counts": metric_frame.groupby("exposure_bucket").size().to_dict(),
        "invalid_model_output_counts": invalid_model_counts,
        "invalid_baseline_output_counts": invalid_baseline_counts,
        "invalid_output_policy": (
            "excluded per arm, method, and window without clipping, imputation, "
            "retry, or seed changes"
        ),
    }

    first = mini[0][0]
    manifest = {
        "schema_version": 1,
        "expected_shards": expected_shards,
        "source_data_fingerprint": first["source_data_fingerprint"],
        "factor_fingerprint": first["factor_fingerprint"],
        "universe_fingerprint": first["universe_fingerprint"],
        "cohort_fingerprint": first["cohort_fingerprint"],
        "common_target_fingerprint": first["common_target_fingerprint"],
        "common_protocol_fingerprint": first["common_protocol_fingerprint"],
        "mini_model_revision": mini[0][0].get("model_revision"),
        "mini_tokenizer_revision": mini[0][0].get("tokenizer_revision"),
        "small_model_revision": small[0][0].get("model_revision"),
        "small_tokenizer_revision": small[0][0].get("tokenizer_revision"),
        "symbols": requested_symbols,
        "evaluated_symbols": evaluated_symbols,
        "bootstrap": {
            "draws": bootstrap_draws,
            "confidence": bootstrap_confidence,
            "seed": bootstrap_seed,
        },
        "invalid_model_output_counts": invalid_model_counts,
        "invalid_baseline_output_counts": invalid_baseline_counts,
    }
    factor = json.loads(Path(factor_manifest).read_text())
    write_benchmark_reduced_output(
        output_dir,
        factor_manifest=factor,
        predictions=prediction_frame,
        window_metrics=metric_frame,
        skips=skip_frame,
        symbol_metrics=symbol_metrics,
        period_metrics=period_metrics,
        ranking_metrics=ranking_metrics,
        paired_comparisons=paired_all,
        bootstrap_intervals=interval_all,
        summary=summary,
        manifest=manifest,
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-shards-dir", type=Path, required=True)
    parser.add_argument("--small-shards-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--factor-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-ranking-cohort", type=int, default=20)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            run_adjusted_reducer(
                mini_shards_dir=arguments.mini_shards_dir,
                small_shards_dir=arguments.small_shards_dir,
                expected_shards=arguments.expected_shards,
                factor_manifest=arguments.factor_manifest,
                output_dir=arguments.output,
                minimum_ranking_cohort=arguments.minimum_ranking_cohort,
                bootstrap_draws=arguments.bootstrap_draws,
                bootstrap_confidence=arguments.bootstrap_confidence,
                bootstrap_seed=arguments.bootstrap_seed,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
