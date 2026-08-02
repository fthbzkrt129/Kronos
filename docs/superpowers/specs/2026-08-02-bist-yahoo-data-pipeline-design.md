# BIST Yahoo Data Pipeline Design

## Purpose

Add a research-only BIST market-data ingestion layer to the Kronos fork. The first release downloads daily Yahoo Finance candles for a versioned BIST 100 universe, validates the data, converts it to Kronos-compatible OHLCVA columns, and records a reproducible manifest. It does not place orders, provide investment advice, or claim production-grade market data.

## Scope

### Included

- A versioned XU100 constituent snapshot for 2026 Q3.
- Yahoo ticker conversion using the `.IS` suffix.
- Daily OHLCV download with bounded retry behavior.
- Deterministic column normalization and quality validation.
- Per-symbol raw and Kronos-ready CSV output.
- A JSON manifest containing successes, failures, dates, and source metadata.
- Unit tests that run without network access.

### Excluded

- Intraday/tick data.
- Automatic scraping of Borsa Istanbul membership history.
- Corporate-action reconstruction beyond Yahoo's raw and adjusted fields.
- Fine-tuning, backtesting, portfolio construction, or broker integration.
- Production or commercial redistribution of Yahoo data.

## Architecture

The feature is isolated in a new `bist_data` Python package. `universe.py` owns constituent-file parsing and date filtering. `yahoo.py` owns the external adapter and normalizes yfinance output. `quality.py` validates candles and creates the columns consumed by Kronos. A small CLI composes those modules and writes data plus a manifest.

The existing China/Qlib fine-tuning flow remains unchanged. BIST data is introduced as an additional adapter, not as a rewrite of `finetune/qlib_data_preprocess.py`.

## Data Contract

The universe CSV contains:

- `symbol`: current Borsa Istanbul code.
- `name`: company display name.
- `valid_from`: inclusive membership date.
- `valid_to`: inclusive membership end date or blank.

Raw output columns:

- `timestamps`
- `open`
- `high`
- `low`
- `close`
- `adj_close`
- `volume`
- `symbol`
- `yahoo_symbol`

Kronos-ready output columns:

- `timestamps`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `amount`

`amount` is an estimate calculated as typical price times volume because Yahoo daily data does not expose official turnover.

## Failure Handling

- Empty responses are recorded as symbol failures.
- Duplicate timestamps are rejected.
- Missing required OHLCV fields are rejected.
- Negative volume and impossible high/low relationships are rejected.
- Transient download failures are retried a bounded number of times.
- Partial output is allowed by default, while `--fail-on-error` provides strict CI behavior.

## Testing

Tests inject fake downloader functions, so unit tests never depend on Yahoo availability. Coverage includes universe filtering, ticker generation, flat and MultiIndex normalization, retry behavior, invalid-candle rejection, and Kronos amount generation.

## Data and Legal Notes

The XU100 snapshot is a research seed compiled from the current constituent list and checked against Borsa Istanbul's 2026 Q3 periodic review. Yahoo/yfinance is intended for personal research and educational use. The generated data directory remains untracked.
