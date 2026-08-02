from __future__ import annotations

from datetime import date, datetime
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PREDICTION_COLUMNS = [
    "symbol",
    "candidate_month",
    "forecast_origin",
    "target_timestamp",
    "horizon_step",
    "method",
    "predicted_close",
    "actual_close",
    "history_last_close",
]
WINDOW_METRIC_COLUMNS = [
    "symbol",
    "candidate_month",
    "forecast_origin",
    "method",
    "mae",
    "rmse",
    "final_ape",
    "final_abs_error",
    "predicted_return_5d",
    "actual_return_5d",
    "direction_correct",
]
SKIP_COLUMNS = [
    "symbol",
    "candidate_month",
    "reason_code",
    "reason_detail",
    "available_history_rows",
    "available_target_rows",
]
BENCHMARK_PREDICTION_COLUMNS = [
    "experiment_arm",
    "symbol",
    "candidate_month",
    "forecast_origin",
    "target_timestamp",
    "horizon_step",
    "method",
    "predicted_close",
    "actual_close",
    "history_last_close",
    "context_view",
    "scoring_target_view",
    "exposure_bucket",
    "context_factor_changed",
    "target_factor_changed",
    "context_max_abs_log_step",
    "target_max_abs_log_from_origin",
    "common_target_fingerprint",
]
BENCHMARK_WINDOW_METRIC_COLUMNS = [
    "experiment_arm",
    "symbol",
    "candidate_month",
    "forecast_origin",
    "method",
    "mae",
    "rmse",
    "final_ape",
    "final_abs_error",
    "predicted_return_5d",
    "actual_return_5d",
    "predicted_log_return_5d",
    "actual_log_return_5d",
    "log_return_abs_error",
    "direction_correct",
    "final_target_timestamp",
    "history_last_close",
    "actual_final_close",
    "context_view",
    "scoring_target_view",
    "exposure_bucket",
    "context_factor_changed",
    "target_factor_changed",
    "context_max_abs_log_step",
    "target_max_abs_log_from_origin",
    "common_target_fingerprint",
]


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    raise TypeError(type(value).__name__)


def _atomic_text(text, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output)


def _atomic_csv(frame, path, columns=None):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = frame.copy()
    if columns is not None:
        missing = set(columns) - set(data.columns)
        if missing:
            raise ValueError("missing output columns: " + ", ".join(sorted(missing)))
        data = data.loc[:, columns]
    for column in data.columns:
        if "timestamp" in column or column == "forecast_origin":
            data[column] = data[column].map(
                lambda value: pd.Timestamp(value).isoformat()
                if not pd.isna(value)
                else value
            )
    temporary = output.with_suffix(output.suffix + ".tmp")
    data.to_csv(temporary, index=False)
    os.replace(temporary, output)


def _atomic_json(payload, path):
    _atomic_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        path,
    )


def validate_prediction_keys(predictions):
    if predictions.duplicated(
        ["symbol", "forecast_origin", "target_timestamp", "method"]
    ).any():
        raise ValueError("duplicate prediction key")


def validate_benchmark_prediction_keys(predictions):
    if predictions.duplicated(
        [
            "experiment_arm",
            "symbol",
            "forecast_origin",
            "target_timestamp",
            "method",
        ]
    ).any():
        raise ValueError("duplicate benchmark prediction key")


def write_shard_output(output_dir, predictions, window_metrics, skips, manifest):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "COMPLETED").unlink(missing_ok=True)
    validate_prediction_keys(predictions)
    _atomic_csv(predictions, output / "predictions.csv", PREDICTION_COLUMNS)
    _atomic_csv(window_metrics, output / "window_metrics.csv", WINDOW_METRIC_COLUMNS)
    _atomic_csv(skips, output / "skipped_windows.csv", SKIP_COLUMNS)
    _atomic_json(manifest, output / "shard_manifest.json")
    _atomic_text("ok\n", output / "COMPLETED")


def write_benchmark_shard_output(
    output_dir, predictions, window_metrics, skips, manifest
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "COMPLETED").unlink(missing_ok=True)
    validate_benchmark_prediction_keys(predictions)
    _atomic_csv(
        predictions,
        output / "predictions.csv",
        BENCHMARK_PREDICTION_COLUMNS,
    )
    _atomic_csv(
        window_metrics,
        output / "window_metrics.csv",
        BENCHMARK_WINDOW_METRIC_COLUMNS,
    )
    _atomic_csv(skips, output / "skipped_windows.csv", SKIP_COLUMNS)
    _atomic_json(manifest, output / "shard_manifest.json")
    _atomic_text("ok\n", output / "COMPLETED")


def _report(summary):
    return (
        "# Zero-shot historical evaluation of the 2026 Q3 BIST 100 "
        "constituent snapshot over 2023-2026\n\n"
        f"- Eligible windows: {summary.get('eligible_windows', 0)}\n\n"
        "This is research-only output and not investment advice. The current "
        "constituent snapshot creates survivorship bias. Yahoo is not an official "
        "licensed Borsa Istanbul feed.\n"
    )


def write_reduced_output(
    output_dir,
    *,
    predictions,
    window_metrics,
    skips,
    symbol_metrics,
    period_metrics,
    ranking_metrics,
    summary,
    manifest,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "COMPLETED").unlink(missing_ok=True)
    validate_prediction_keys(predictions)
    _atomic_csv(predictions, output / "predictions.csv", PREDICTION_COLUMNS)
    _atomic_csv(window_metrics, output / "window_metrics.csv", WINDOW_METRIC_COLUMNS)
    _atomic_csv(skips, output / "skipped_windows.csv", SKIP_COLUMNS)
    _atomic_csv(symbol_metrics, output / "symbol_metrics.csv")
    _atomic_csv(period_metrics, output / "period_metrics.csv")
    _atomic_csv(ranking_metrics, output / "ranking_metrics.csv")
    _atomic_json(summary, output / "summary.json")
    _atomic_json(manifest, output / "run_manifest.json")
    _atomic_text(_report(summary), output / "report.md")
    _atomic_text("ok\n", output / "COMPLETED")


def _benchmark_report(summary):
    questions = summary.get("questions", {})
    invalid_counts = summary.get("invalid_model_output_counts", {})
    lines = [
        "# Paired adjusted-price zero-shot benchmark of Kronos-mini and "
        "Kronos-small on the 2026 Q3 BIST 100 constituent snapshot over 2023-2026",
        "",
        "## Technical completion",
        f"- Prediction rows: {summary.get('prediction_rows', 0)}",
        f"- Eligible model windows: {summary.get('eligible_model_windows', 0)}",
        "",
        "## Benchmark questions",
    ]
    for question, answer in questions.items():
        lines.append(f"- {question}: {answer}")
    lines.extend(["", "## Rejected model outputs"])
    if invalid_counts:
        for arm, count in sorted(invalid_counts.items()):
            lines.append(f"- {arm}: {count} window(s)")
    else:
        lines.append("- None")
    lines.append(
        "Invalid sampled outputs were excluded per arm and window without "
        "clipping, imputation, retrying, or changing the seed. Paired conclusions "
        "use only windows present in both compared arms."
    )
    lines.extend(
        [
            "",
            "## Limitations",
            "This is research-only output, not investment advice or a trading "
            "strategy. The 2026 Q3 constituent snapshot is projected backward and "
            "introduces survivorship and selection bias. Yahoo is not an official "
            "licensed Borsa Istanbul source. The provider factor-ratio method is a "
            "research assumption. Volume remains raw, amount is estimated, and no "
            "transaction costs, liquidity constraints, portfolio construction, "
            "broker connection, or order placement are included.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_benchmark_reduced_output(
    output_dir,
    *,
    factor_manifest,
    predictions,
    window_metrics,
    skips,
    symbol_metrics,
    period_metrics,
    ranking_metrics,
    paired_comparisons,
    bootstrap_intervals,
    summary,
    manifest,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "COMPLETED").unlink(missing_ok=True)
    validate_benchmark_prediction_keys(predictions)
    _atomic_json(factor_manifest, output / "adjusted_data_manifest.json")
    _atomic_csv(
        predictions,
        output / "predictions.csv",
        BENCHMARK_PREDICTION_COLUMNS,
    )
    _atomic_csv(
        window_metrics,
        output / "window_metrics.csv",
        BENCHMARK_WINDOW_METRIC_COLUMNS,
    )
    _atomic_csv(skips, output / "skipped_windows.csv", SKIP_COLUMNS)
    _atomic_csv(symbol_metrics, output / "symbol_metrics.csv")
    _atomic_csv(period_metrics, output / "period_metrics.csv")
    _atomic_csv(ranking_metrics, output / "ranking_metrics.csv")
    _atomic_csv(paired_comparisons, output / "paired_comparisons.csv")
    _atomic_csv(bootstrap_intervals, output / "bootstrap_intervals.csv")
    _atomic_json(summary, output / "summary.json")
    _atomic_json(manifest, output / "run_manifest.json")
    _atomic_text(_benchmark_report(summary), output / "report.md")
    _atomic_text("ok\n", output / "COMPLETED")
