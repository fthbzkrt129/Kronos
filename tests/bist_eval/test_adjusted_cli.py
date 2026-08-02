import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bist_eval.config import AdjustedBenchmarkConfig
from scripts.evaluate_bist100_adjusted_benchmark import run_adjusted_evaluation
from scripts.reduce_bist100_adjusted_benchmark import (
    _invalid_model_output_counts,
    run_adjusted_reducer,
)
from scripts.validate_bist_adjustment_factors import run_factor_validation


class Adapter:
    def predict_cohort(self, windows):
        return {
            window.symbol: np.repeat(
                float(window.context.close.iloc[-1]), len(window.target_timestamps)
            )
            for window in windows
        }


def setup(tmp_path, symbols=("AAA", "BBB")):
    raw = tmp_path / "raw"
    raw.mkdir()
    universe = tmp_path / "universe.csv"
    universe.write_text(
        "symbol,name,valid_from,valid_to\n"
        + "".join(
            f"{symbol},{symbol},2026-01-01,2026-12-31\n" for symbol in symbols
        )
    )
    timestamps = pd.bdate_range("2022-01-03", periods=950)
    for index, symbol in enumerate(symbols):
        close = 100 + index + np.arange(len(timestamps)) * 0.1
        frame = pd.DataFrame(
            {
                "timestamps": timestamps,
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "adj_close": close,
                "volume": 1000 + np.arange(len(timestamps)),
                "symbol": symbol,
                "yahoo_symbol": symbol + ".IS",
            }
        )
        frame.to_csv(raw / f"{symbol}.csv", index=False)
    source = tmp_path / "manifest.json"
    source.write_text("{}")
    return raw, universe, source


def test_factor_command_writes_provenance_only(tmp_path):
    raw, universe, source = setup(tmp_path)
    output = tmp_path / "factors"
    manifest = run_factor_validation(
        raw_dir=raw,
        source_manifest=source,
        universe_path=universe,
        output_dir=output,
        strict=True,
    )
    assert manifest["summary"]["succeeded"] == 2
    assert {path.name for path in output.iterdir()} == {
        "factor_manifest.json",
        "factor_diagnostics.csv",
        "COMPLETED",
    }


def test_factor_source_digest_mismatch_fails_closed(tmp_path):
    raw, universe, source = setup(tmp_path)
    factor_dir = tmp_path / "factors"
    run_factor_validation(
        raw_dir=raw,
        source_manifest=source,
        universe_path=universe,
        output_dir=factor_dir,
        strict=True,
    )
    source.write_text('{"changed":true}')
    config = AdjustedBenchmarkConfig(
        start_date="2024-01-01",
        end_date="2025-01-31",
        calendar_coverage=1,
        minimum_ranking_cohort=1,
        shard_count=2,
    )
    with pytest.raises(ValueError, match="source manifest"):
        run_adjusted_evaluation(
            mode="mini-pair",
            raw_dir=raw,
            source_manifest=source,
            factor_manifest=factor_dir / "factor_manifest.json",
            universe_path=universe,
            output_dir=tmp_path / "out",
            config=config,
            symbols=["AAA"],
            strict=True,
            adapter=Adapter(),
        )


def test_two_mode_shards_reduce_to_completed_result(tmp_path):
    raw, universe, source = setup(tmp_path)
    factor_dir = tmp_path / "factors"
    run_factor_validation(
        raw_dir=raw,
        source_manifest=source,
        universe_path=universe,
        output_dir=factor_dir,
        strict=True,
    )
    mini = tmp_path / "mini"
    small = tmp_path / "small"
    mini.mkdir()
    small.mkdir()
    for index in range(2):
        mini_config = AdjustedBenchmarkConfig(
            start_date="2024-01-01",
            end_date="2025-01-31",
            calendar_coverage=1,
            minimum_ranking_cohort=1,
            shard_count=2,
            model_revision="m1",
            tokenizer_revision="t1",
        )
        small_config = AdjustedBenchmarkConfig(
            experiment_arm="adjusted-small",
            model_id="NeoQuasar/Kronos-small",
            tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
            start_date="2024-01-01",
            end_date="2025-01-31",
            calendar_coverage=1,
            minimum_ranking_cohort=1,
            shard_count=2,
            model_revision="m2",
            tokenizer_revision="t2",
        )
        run_adjusted_evaluation(
            mode="mini-pair",
            raw_dir=raw,
            source_manifest=source,
            factor_manifest=factor_dir / "factor_manifest.json",
            universe_path=universe,
            output_dir=mini / f"shard-{index}",
            config=mini_config,
            shard_index=index,
            strict=True,
            adapter=Adapter(),
        )
        run_adjusted_evaluation(
            mode="small",
            raw_dir=raw,
            source_manifest=source,
            factor_manifest=factor_dir / "factor_manifest.json",
            universe_path=universe,
            output_dir=small / f"shard-{index}",
            config=small_config,
            shard_index=index,
            strict=True,
            adapter=Adapter(),
        )
    output = tmp_path / "final"
    summary = run_adjusted_reducer(
        mini_shards_dir=mini,
        small_shards_dir=small,
        expected_shards=2,
        factor_manifest=factor_dir / "factor_manifest.json",
        output_dir=output,
        minimum_ranking_cohort=1,
        bootstrap_draws=100,
        bootstrap_confidence=0.9,
        bootstrap_seed=1,
    )
    assert summary["symbols_requested"] == 2
    assert summary["symbols_evaluated"] == 2
    assert summary["invalid_model_output_counts"] == {}
    run_manifest = json.loads((output / "run_manifest.json").read_text())
    assert run_manifest["symbols"] == ["AAA", "BBB"]
    assert run_manifest["evaluated_symbols"] == ["AAA", "BBB"]
    assert (output / "COMPLETED").is_file()
    assert (output / "bootstrap_intervals.csv").is_file()
    assert "Rejected model outputs" in (output / "report.md").read_text()


def test_invalid_model_output_counts_are_grouped_by_arm():
    skips = pd.DataFrame(
        {
            "reason_code": [
                "invalid_model_output",
                "invalid_model_output",
                "invalid_model_output",
                "insufficient_history",
            ],
            "reason_detail": [
                "arm=adjusted-small; negative close",
                "arm=adjusted-small; negative close",
                "arm=raw-mini; negative close",
                "need 400 rows",
            ],
        }
    )
    assert _invalid_model_output_counts(skips) == {
        "adjusted-small": 2,
        "raw-mini": 1,
    }
