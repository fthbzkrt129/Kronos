from collections import defaultdict

import pandas as pd
import pytest

from bist_data.yahoo import YahooDownloadError, download_daily_candles, normalize_yahoo_frame


def sample_index():
    return pd.DatetimeIndex(["2026-07-30", "2026-07-31"], name="Date")


def valid_response() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [10.0],
            "High": [12.0],
            "Low": [9.0],
            "Close": [11.0],
            "Adj Close": [10.5],
            "Volume": [100.0],
        },
        index=pd.DatetimeIndex(["2026-07-30"], name="Date"),
    )


def test_normalize_yahoo_frame_handles_flat_columns():
    frame = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [12.0, 13.0],
            "Low": [9.0, 10.0],
            "Close": [11.0, 12.0],
            "Adj Close": [10.5, 11.5],
            "Volume": [100, 200],
        },
        index=sample_index(),
    )

    normalized = normalize_yahoo_frame(frame, "THYAO.IS")

    assert list(normalized.columns) == [
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
    assert normalized["symbol"].unique().tolist() == ["THYAO"]
    assert normalized["yahoo_symbol"].unique().tolist() == ["THYAO.IS"]


def test_normalize_yahoo_frame_handles_multiindex_columns():
    columns = pd.MultiIndex.from_tuples(
        [
            ("Open", "THYAO.IS"),
            ("High", "THYAO.IS"),
            ("Low", "THYAO.IS"),
            ("Close", "THYAO.IS"),
            ("Adj Close", "THYAO.IS"),
            ("Volume", "THYAO.IS"),
        ]
    )
    frame = pd.DataFrame(
        [[10.0, 12.0, 9.0, 11.0, 10.5, 100]],
        index=pd.DatetimeIndex(["2026-07-30"], name="Date"),
        columns=columns,
    )

    normalized = normalize_yahoo_frame(frame, "THYAO.IS")

    assert normalized.loc[0, "close"] == 11.0
    assert normalized.loc[0, "volume"] == 100


def test_download_daily_candles_retries_transient_failure():
    calls = defaultdict(int)

    def fake_download(symbol, **kwargs):
        calls[symbol] += 1
        if calls[symbol] == 1:
            raise RuntimeError("temporary")
        return valid_response()

    result = download_daily_candles(
        "THYAO",
        start="2026-01-01",
        end="2026-08-01",
        retries=1,
        downloader=fake_download,
        sleep_seconds=0,
    )

    assert calls["THYAO.IS"] == 2
    assert len(result) == 1


def test_download_daily_candles_retries_incomplete_provider_frame():
    calls = defaultdict(int)

    def fake_download(symbol, **kwargs):
        calls[symbol] += 1
        if calls[symbol] == 1:
            incomplete = valid_response()
            incomplete.loc[:, "Volume"] = float("nan")
            return incomplete
        return valid_response()

    result = download_daily_candles(
        "THYAO",
        start="2026-01-01",
        end="2026-08-01",
        retries=1,
        downloader=fake_download,
        sleep_seconds=0,
    )

    assert calls["THYAO.IS"] == 2
    assert result.loc[0, "volume"] == 100


def test_normalize_yahoo_frame_rejects_missing_ohlcv_values():
    frame = valid_response()
    frame.loc[:, "Close"] = float("nan")

    with pytest.raises(YahooDownloadError, match="missing OHLCV"):
        normalize_yahoo_frame(frame, "THYAO.IS")


def test_download_daily_candles_rejects_empty_response():
    def fake_download(symbol, **kwargs):
        return pd.DataFrame()

    with pytest.raises(YahooDownloadError, match="empty"):
        download_daily_candles(
            "THYAO",
            start="2026-01-01",
            end="2026-08-01",
            retries=0,
            downloader=fake_download,
        )
