import pandas as pd
import pytest

from bist_eval.model_adapter import KronosModelAdapter, derive_cohort_seed
from bist_eval.windows import ForecastWindow


class FakePredictor:
    def __init__(self, mode="valid"):
        self.calls = []
        self.mode = mode

    def predict_batch(self, frames, history_timestamps, target_timestamps, **kwargs):
        self.calls.append((frames, history_timestamps, target_timestamps, kwargs))
        outputs = []
        for index, timestamps in enumerate(target_timestamps):
            count = len(timestamps) - (1 if self.mode == "schema" else 0)
            values = [10.0] * count
            if self.mode == "negative" and index == 1:
                values[-1] = -3.0
            outputs.append(
                pd.DataFrame(
                    {"close": values},
                    index=pd.DatetimeIndex(timestamps.iloc[:count]),
                )
            )
        return outputs


def make_window(frame_factory, symbol="X"):
    frame = frame_factory(405)
    return ForecastWindow(
        symbol,
        "2024-07",
        frame.timestamps.iloc[399],
        tuple(frame.timestamps.iloc[400:405]),
        frame.iloc[:400].copy(),
        frame.iloc[400:405].copy(),
    )


def test_predict_cohort_contract(frame_factory):
    fake = FakePredictor()
    adapter = KronosModelAdapter(predictor=fake, seed=7)
    window = make_window(frame_factory)
    output = adapter.predict_cohort([window])
    assert list(output) == ["X"]
    assert len(output["X"]) == 5
    frames, history_timestamps, target_timestamps, kwargs = fake.calls[0]
    assert list(frames[0].columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert len(history_timestamps[0]) == 400
    assert kwargs["verbose"] is False


def test_invalid_output_schema_still_fails_legacy_api(frame_factory):
    with pytest.raises(ValueError, match="schema"):
        KronosModelAdapter(predictor=FakePredictor("schema")).predict_cohort(
            [make_window(frame_factory)]
        )


def test_detailed_api_preserves_valid_batch_members(frame_factory):
    windows = [
        make_window(frame_factory, "VALID"),
        make_window(frame_factory, "INVALID"),
    ]
    adapter = KronosModelAdapter(predictor=FakePredictor("negative"))
    result = adapter.predict_cohort_with_failures(windows)

    assert list(result.predictions) == ["VALID"]
    assert len(result.predictions["VALID"]) == 5
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert failure.symbol == "INVALID"
    assert failure.reason_code == "invalid_model_output"
    assert "-3.0" in failure.reason_detail


def test_invalid_close_still_fails_legacy_api(frame_factory):
    windows = [
        make_window(frame_factory, "VALID"),
        make_window(frame_factory, "INVALID"),
    ]
    with pytest.raises(ValueError, match="invalid close"):
        KronosModelAdapter(predictor=FakePredictor("negative")).predict_cohort(
            windows
        )


def test_seed_is_stable():
    assert derive_cohort_seed(1, pd.Timestamp("2026-01-01")) == derive_cohort_seed(
        1, pd.Timestamp("2026-01-01")
    )
