import pandas as pd
import pytest

from bist_data.quality import CandleQualityError, to_kronos_frame, validate_candles


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamps": pd.to_datetime(["2026-07-30", "2026-07-31"]),
            "open": [10.0, 11.0],
            "high": [12.0, 13.0],
            "low": [9.0, 10.0],
            "close": [11.0, 12.0],
            "adj_close": [10.5, 11.5],
            "volume": [100.0, 200.0],
            "symbol": ["THYAO", "THYAO"],
            "yahoo_symbol": ["THYAO.IS", "THYAO.IS"],
        }
    )


def test_validate_candles_rejects_impossible_high_low_relationship():
    frame = valid_frame()
    frame.loc[0, "high"] = 8.0

    with pytest.raises(CandleQualityError, match="high"):
        validate_candles(frame)


def test_validate_candles_rejects_duplicate_timestamps():
    frame = valid_frame()
    frame.loc[1, "timestamps"] = frame.loc[0, "timestamps"]

    with pytest.raises(CandleQualityError, match="duplicate"):
        validate_candles(frame)


def test_validate_candles_rejects_negative_volume():
    frame = valid_frame()
    frame.loc[0, "volume"] = -1

    with pytest.raises(CandleQualityError, match="volume"):
        validate_candles(frame)


def test_to_kronos_frame_generates_amount_from_typical_price():
    result = to_kronos_frame(valid_frame())

    assert list(result.columns) == [
        "timestamps",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert result.loc[0, "amount"] == pytest.approx(((12.0 + 9.0 + 11.0) / 3) * 100)
