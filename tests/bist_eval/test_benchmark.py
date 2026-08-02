import numpy as np
import pandas as pd

from bist_eval.benchmark import run_mini_pair_shard, run_small_shard
from bist_eval.calendar import MonthlyCohort
from bist_eval.config import AdjustedBenchmarkConfig
from bist_eval.model_adapter import CohortPredictionResult, PredictionFailure


class Adapter:
    def __init__(self):
        self.contexts = []

    def predict_cohort(self, windows):
        self.contexts.extend([window.context.copy() for window in windows])
        return {
            window.symbol: np.repeat(
                float(window.context.close.iloc[-1]), len(window.target_timestamps)
            )
            for window in windows
        }


class PartialFailureAdapter:
    def __init__(self, fail_call):
        self.calls = 0
        self.fail_call = fail_call

    def predict_cohort_with_failures(self, windows):
        self.calls += 1
        if self.calls == self.fail_call:
            window = windows[0]
            return CohortPredictionResult(
                {},
                (
                    PredictionFailure(
                        window.symbol,
                        window.candidate_month,
                        window.forecast_origin,
                        "invalid_model_output",
                        "nonpositive_or_nonfinite_close=[1.0,-3.0]",
                    ),
                ),
            )
        return CohortPredictionResult(
            {
                window.symbol: np.repeat(
                    float(window.context.close.iloc[-1]),
                    len(window.target_timestamps),
                )
                for window in windows
            },
            (),
        )


def fixture():
    rows = 410
    timestamps = pd.bdate_range("2022-01-03", periods=rows)
    close = 100 + np.arange(rows) * 0.1
    factors = np.ones(rows)
    factors[:200] = 0.5
    raw = pd.DataFrame(
        {
            "timestamps": timestamps,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "adj_close": close * factors,
            "volume": 1000 + np.arange(rows),
            "symbol": "AAA",
            "yahoo_symbol": "AAA.IS",
        }
    )
    origin = timestamps[404]
    return {"AAA": raw}, [
        MonthlyCohort("2023-07", origin, tuple(timestamps[405:410]))
    ]


def test_mini_pair_uses_common_target_and_separate_contexts():
    frames, cohorts = fixture()
    adapter = Adapter()
    result = run_mini_pair_shard(
        raw_frames=frames,
        cohorts=cohorts,
        symbols=["AAA"],
        config=AdjustedBenchmarkConfig(),
        adapter=adapter,
    )
    assert set(result.predictions.experiment_arm) == {
        "raw-mini",
        "adjusted-mini",
        "adjusted-baselines",
    }
    assert (
        result.predictions.groupby(["symbol", "target_timestamp"])
        .actual_close.nunique()
        .max()
        == 1
    )
    assert len(adapter.contexts) == 2
    assert not adapter.contexts[0].equals(adapter.contexts[1])


def test_small_arm_does_not_duplicate_baselines():
    frames, cohorts = fixture()
    config = AdjustedBenchmarkConfig(
        experiment_arm="adjusted-small",
        model_id="NeoQuasar/Kronos-small",
        tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
    )
    result = run_small_shard(
        raw_frames=frames,
        cohorts=cohorts,
        symbols=["AAA"],
        config=config,
        adapter=Adapter(),
    )
    assert set(result.predictions.experiment_arm) == {"adjusted-small"}
    assert set(result.predictions.method) == {"kronos"}


def test_invalid_adjusted_mini_window_is_excluded_without_imputation():
    frames, cohorts = fixture()
    result = run_mini_pair_shard(
        raw_frames=frames,
        cohorts=cohorts,
        symbols=["AAA"],
        config=AdjustedBenchmarkConfig(),
        adapter=PartialFailureAdapter(fail_call=2),
    )

    assert set(result.predictions.experiment_arm) == {
        "raw-mini",
        "adjusted-baselines",
    }
    assert "adjusted-mini" not in set(result.predictions.experiment_arm)
    assert (result.predictions.predicted_close > 0).all()
    assert len(result.skips) == 1
    assert result.skips.iloc[0].reason_code == "invalid_model_output"
    assert "arm=adjusted-mini" in result.skips.iloc[0].reason_detail
    assert result.manifest["invalid_model_outputs"]["adjusted-mini"] == 1


def test_invalid_small_window_becomes_explicit_skip():
    frames, cohorts = fixture()
    config = AdjustedBenchmarkConfig(
        experiment_arm="adjusted-small",
        model_id="NeoQuasar/Kronos-small",
        tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
    )
    result = run_small_shard(
        raw_frames=frames,
        cohorts=cohorts,
        symbols=["AAA"],
        config=config,
        adapter=PartialFailureAdapter(fail_call=1),
    )

    assert result.predictions.empty
    assert len(result.skips) == 1
    assert result.skips.iloc[0].reason_code == "invalid_model_output"
    assert "arm=adjusted-small" in result.skips.iloc[0].reason_detail
    assert result.manifest["invalid_model_outputs"]["adjusted-small"] == 1


def test_invalid_baseline_window_is_excluded_without_clipping(monkeypatch):
    def fake_baselines(context, horizon):
        last = np.repeat(float(context.close.iloc[-1]), horizon)
        return {
            "last_close": last,
            "momentum_20": last + 1.0,
            "linear_trend_20": np.array([4.0, 3.0, 2.0, 1.0, -0.5]),
        }

    monkeypatch.setattr("bist_eval.benchmark.forecast_baselines", fake_baselines)
    frames, cohorts = fixture()
    result = run_mini_pair_shard(
        raw_frames=frames,
        cohorts=cohorts,
        symbols=["AAA"],
        config=AdjustedBenchmarkConfig(),
        adapter=Adapter(),
    )

    baseline_methods = set(
        result.predictions.loc[
            result.predictions.experiment_arm == "adjusted-baselines", "method"
        ]
    )
    assert baseline_methods == {"last_close", "momentum_20"}
    assert (result.predictions.predicted_close > 0).all()
    rejected = result.skips[
        result.skips.reason_code == "invalid_baseline_output"
    ]
    assert len(rejected) == 1
    assert "method=linear_trend_20" in rejected.iloc[0].reason_detail
    assert "-0.5" in rejected.iloc[0].reason_detail
    assert result.manifest["invalid_baseline_outputs"] == {
        "linear_trend_20": 1
    }
