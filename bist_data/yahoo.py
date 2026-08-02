"""Yahoo Finance adapter for daily Borsa Istanbul candles."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pandas as pd


class YahooDownloadError(RuntimeError):
    """Raised when Yahoo data cannot be downloaded or normalized."""


_EXPECTED_NAMES = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "adj_close",
    "adj_close": "adj_close",
    "volume": "volume",
    "date": "timestamps",
    "datetime": "timestamps",
    "timestamps": "timestamps",
}
_REQUIRED_PRICE_COLUMNS = {"open", "high", "low", "close", "volume"}


def _normalize_label(value: object) -> str:
    return str(value).strip().lower().replace("_", " ")


def _flatten_column(column: object) -> str:
    if isinstance(column, tuple):
        for level in column:
            normalized = _normalize_label(level)
            if normalized in _EXPECTED_NAMES:
                return _EXPECTED_NAMES[normalized]
        return "_".join(str(level) for level in column if str(level))

    normalized = _normalize_label(column)
    return _EXPECTED_NAMES.get(normalized, normalized.replace(" ", "_"))


def _to_yahoo_symbol(symbol: str) -> str:
    clean = symbol.strip().upper()
    return clean if clean.endswith(".IS") else f"{clean}.IS"


def normalize_yahoo_frame(frame: pd.DataFrame, yahoo_symbol: str) -> pd.DataFrame:
    """Normalize flat or MultiIndex yfinance output into a stable schema.

    Incomplete OHLCV responses are rejected here so ``download_daily_candles``
    can retry them as transient provider failures instead of passing them into
    downstream quality checks.
    """

    if frame is None or frame.empty:
        raise YahooDownloadError(f"empty Yahoo response for {yahoo_symbol}")

    normalized = frame.copy()
    normalized.columns = [_flatten_column(column) for column in normalized.columns]

    if "timestamps" not in normalized.columns:
        normalized = normalized.reset_index()
        normalized.columns = [_flatten_column(column) for column in normalized.columns]

    missing = _REQUIRED_PRICE_COLUMNS - set(normalized.columns)
    if missing:
        raise YahooDownloadError(
            f"Yahoo response for {yahoo_symbol} is missing columns: "
            + ", ".join(sorted(missing))
        )
    if "timestamps" not in normalized.columns:
        raise YahooDownloadError(
            f"Yahoo response for {yahoo_symbol} does not include timestamps"
        )

    normalized["timestamps"] = pd.to_datetime(
        normalized["timestamps"], errors="raise"
    )
    if normalized["timestamps"].dt.tz is not None:
        normalized["timestamps"] = normalized["timestamps"].dt.tz_localize(None)

    for column in sorted(_REQUIRED_PRICE_COLUMNS):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    incomplete_rows = normalized[list(sorted(_REQUIRED_PRICE_COLUMNS))].isna().any(axis=1)
    if incomplete_rows.any():
        raise YahooDownloadError(
            f"Yahoo response for {yahoo_symbol} contains "
            f"{int(incomplete_rows.sum())} row(s) with missing OHLCV values"
        )

    if "adj_close" not in normalized.columns:
        normalized["adj_close"] = normalized["close"]
    else:
        normalized["adj_close"] = pd.to_numeric(
            normalized["adj_close"], errors="coerce"
        ).fillna(normalized["close"])

    base_symbol = yahoo_symbol.removesuffix(".IS")
    normalized["symbol"] = base_symbol
    normalized["yahoo_symbol"] = yahoo_symbol

    output_columns = [
        "timestamps",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "symbol",
        "yahoo_symbol",
    ]
    return normalized.loc[:, output_columns].reset_index(drop=True)


def _default_downloader(symbol: str, **kwargs: Any) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise YahooDownloadError(
            "yfinance is required; install dependencies from requirements.txt"
        ) from exc
    return yf.download(symbol, **kwargs)


def download_daily_candles(
    symbol: str,
    *,
    start: str,
    end: str,
    retries: int = 2,
    downloader: Callable[..., pd.DataFrame] | None = None,
    sleep_seconds: float = 1.0,
) -> pd.DataFrame:
    """Download one daily BIST series with bounded retry behavior."""

    if retries < 0:
        raise ValueError("retries must be zero or greater")

    yahoo_symbol = _to_yahoo_symbol(symbol)
    fetch = downloader or _default_downloader
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            frame = fetch(
                yahoo_symbol,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                group_by="column",
                threads=False,
            )
            return normalize_yahoo_frame(frame, yahoo_symbol)
        except Exception as exc:
            last_error = exc
            if attempt < retries and sleep_seconds > 0:
                time.sleep(sleep_seconds)

    assert last_error is not None
    if isinstance(last_error, YahooDownloadError):
        raise last_error
    raise YahooDownloadError(
        f"failed to download {yahoo_symbol} after {retries + 1} attempt(s): "
        f"{last_error}"
    ) from last_error
