"""Quality gates and Kronos conversion for daily candles."""

from __future__ import annotations

import pandas as pd


_REQUIRED_COLUMNS = {"timestamps", "open", "high", "low", "close", "volume"}
_PRICE_COLUMNS = ["open", "high", "low", "close"]


class CandleQualityError(ValueError):
    """Raised when candle data violates the research data contract."""


def validate_candles(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate required fields and basic OHLCV invariants.

    A normalized copy sorted by timestamp is returned. Invalid observations are
    rejected rather than silently repaired.
    """

    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise CandleQualityError(
            "missing candle columns: " + ", ".join(sorted(missing))
        )
    if frame.empty:
        raise CandleQualityError("candle frame is empty")

    validated = frame.copy()
    try:
        validated["timestamps"] = pd.to_datetime(
            validated["timestamps"], errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise CandleQualityError("timestamps contain invalid values") from exc

    if validated["timestamps"].duplicated().any():
        raise CandleQualityError("duplicate timestamps are not allowed")

    numeric_columns = _PRICE_COLUMNS + ["volume"]
    for column in numeric_columns:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[numeric_columns].isna().any().any():
        raise CandleQualityError("OHLCV columns contain missing or non-numeric values")

    if (validated[_PRICE_COLUMNS] < 0).any().any():
        raise CandleQualityError("price columns must not be negative")
    if (validated["volume"] < 0).any():
        raise CandleQualityError("volume must not be negative")

    max_body = validated[["open", "close", "low"]].max(axis=1)
    min_body = validated[["open", "close", "high"]].min(axis=1)
    if (validated["high"] < max_body).any():
        raise CandleQualityError("high must be at least open, close, and low")
    if (validated["low"] > min_body).any():
        raise CandleQualityError("low must be at most open, close, and high")

    return validated.sort_values("timestamps").reset_index(drop=True)


def to_kronos_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Create Kronos OHLCVA columns from validated Yahoo daily candles."""

    validated = validate_candles(frame)
    typical_price = (
        validated["high"] + validated["low"] + validated["close"]
    ) / 3.0
    result = validated.loc[
        :, ["timestamps", "open", "high", "low", "close", "volume"]
    ].copy()
    result["amount"] = typical_price * validated["volume"]
    return result
