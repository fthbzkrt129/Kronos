import dataclasses
from pathlib import Path

import pytest

from bist_eval.config import AdjustedBenchmarkConfig, MODEL_TOKENIZER_PAIRS
from bist_eval.model_adapter import KronosModelAdapter
from bist_eval.windows import PredictionWindow


def test_adjusted_config_and_model_pairs():
    config = AdjustedBenchmarkConfig()
    assert config.adjustment_formula_version == "origin-rebased-v1"
    assert config.bootstrap_draws == 10000
    assert MODEL_TOKENIZER_PAIRS["NeoQuasar/Kronos-small"] == (
        "NeoQuasar/Kronos-Tokenizer-base",
        512,
    )


def test_arm_changes_full_but_not_common_fingerprint():
    raw = AdjustedBenchmarkConfig(experiment_arm="raw-mini", context_view="raw")
    adjusted = AdjustedBenchmarkConfig()
    assert raw.fingerprint != adjusted.fingerprint
    assert raw.common_protocol_fingerprint == adjusted.common_protocol_fingerprint


def test_prediction_window_has_no_target_values_or_factors():
    assert {field.name for field in dataclasses.fields(PredictionWindow)} == {
        "symbol",
        "candidate_month",
        "forecast_origin",
        "target_timestamps",
        "context",
    }


def test_invalid_small_tokenizer_is_rejected():
    with pytest.raises(ValueError, match="model-tokenizer"):
        KronosModelAdapter(
            model_id="NeoQuasar/Kronos-small",
            tokenizer_id="NeoQuasar/Kronos-Tokenizer-2k",
        )


def test_full_workflow_restores_data_artifact_under_expected_root():
    workflow = Path(
        ".github/workflows/bist100-adjusted-kronos-benchmark.yml"
    ).read_text(encoding="utf-8")
    marker = (
        "name: adjusted-benchmark-data-${{ github.run_id }}\n"
        "          path: .artifacts"
    )
    assert workflow.count(marker) == 3
