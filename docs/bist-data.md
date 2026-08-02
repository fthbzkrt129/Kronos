# BIST Yahoo Research Data Pipeline

This fork includes a research-only adapter that downloads daily Borsa Istanbul candles from Yahoo Finance and converts them into the columns consumed by Kronos.

## Important limitations

- Yahoo/yfinance data is suitable for personal research and prototyping, not as an official exchange feed.
- The included `XU100` file is a versioned 2026 Q3 snapshot. It is not a complete historical membership database.
- Yahoo does not provide official daily turnover in this download. The Kronos `amount` value is estimated as `((high + low + close) / 3) * volume`.
- Do not connect this pipeline directly to a live brokerage account. Validate with licensed data, walk-forward backtests, and paper trading first.

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

## Download the full 2026 Q3 BIST 100 snapshot

The Yahoo `end` date is exclusive. The command below writes generated files under `data/bist/`, which is ignored by Git.

```bash
python scripts/download_bist_yahoo.py \
  --universe data/universes/xu100_2026_q3.csv \
  --as-of 2026-08-02 \
  --start 2015-01-01 \
  --end 2026-08-03 \
  --output data/bist/yahoo \
  --fail-on-error
```

## Download selected symbols

```bash
python scripts/download_bist_yahoo.py \
  --symbols THYAO ASELS TUPRS \
  --start 2020-01-01 \
  --output data/bist/yahoo-sample
```

Comma-separated values are also accepted:

```bash
python scripts/download_bist_yahoo.py --symbols THYAO,ASELS,TUPRS
```

## Output structure

```text
data/bist/yahoo/
├── raw/
│   ├── ASELS.csv
│   └── THYAO.csv
├── kronos/
│   ├── ASELS.csv
│   └── THYAO.csv
└── manifest.json
```

Raw CSV files contain:

```text
timestamps,open,high,low,close,adj_close,volume,symbol,yahoo_symbol
```

Kronos CSV files contain:

```text
timestamps,open,high,low,close,volume,amount
```

The manifest records the requested date range, successful symbols, failed symbols, row counts, file paths, and data limitations. Without `--fail-on-error`, failures are recorded but successful symbols are still written.

## Run tests

Tests inject fake Yahoo responses and require no network connection.

```bash
python -m pytest tests/bist_data -v
python -m compileall bist_data scripts/download_bist_yahoo.py
```

## Next research milestone

The next milestone is a zero-shot Kronos evaluation against simple BIST baselines. Fine-tuning should only begin after the raw data, corporate actions, point-in-time index membership, and transaction-cost assumptions have been upgraded to licensed or independently verified sources.
