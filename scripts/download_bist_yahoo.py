#!/usr/bin/env python3
"""Download research-only daily BIST candles from Yahoo Finance."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bist_data.quality import (
    repair_ohlc_envelope,
    to_kronos_frame,
    validate_candles,
)
from bist_data.universe import UniverseEntry, load_universe
from bist_data.yahoo import download_daily_candles


DEFAULT_UNIVERSE = Path("data/universes/xu100_2026_q3.csv")
DEFAULT_OUTPUT = Path("data/bist/yahoo")


def _parse_symbols(values: Sequence[str] | None) -> list[str] | None:
    if not values:
        return None
    symbols: list[str] = []
    for value in values:
        symbols.extend(part.strip().upper() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(symbols))


def _select_entries(
    entries: list[UniverseEntry], symbols: Sequence[str] | None
) -> list[UniverseEntry]:
    if not symbols:
        return entries

    by_symbol = {entry.symbol: entry for entry in entries}
    unknown = sorted(set(symbols) - set(by_symbol))
    if unknown:
        raise ValueError("symbols are not present in the universe: " + ", ".join(unknown))
    return [by_symbol[symbol] for symbol in symbols]


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, date_format="%Y-%m-%d")
    os.replace(temporary, path)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_pipeline(
    *,
    universe_path: str | Path,
    output_dir: str | Path,
    start: str,
    end: str,
    symbols: Sequence[str] | None,
    retries: int,
    sleep_seconds: float,
    fail_on_error: bool,
    as_of: date | str | None = None,
    downloader: Callable[..., pd.DataFrame] | None = None,
) -> tuple[dict[str, Any], int]:
    """Run the BIST Yahoo pipeline and return its manifest and exit code."""

    universe = Path(universe_path)
    output = Path(output_dir)
    parsed_symbols = _parse_symbols(symbols)
    entries = _select_entries(load_universe(universe, as_of=as_of), parsed_symbols)

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for entry in entries:
        try:
            raw = download_daily_candles(
                entry.symbol,
                start=start,
                end=end,
                retries=retries,
                downloader=downloader,
                sleep_seconds=sleep_seconds,
            )
            model_frame, ohlc_repairs = repair_ohlc_envelope(raw)
            validated = validate_candles(model_frame)
            kronos = to_kronos_frame(validated)

            raw_path = output / "raw" / f"{entry.symbol}.csv"
            kronos_path = output / "kronos" / f"{entry.symbol}.csv"
            _atomic_write_csv(raw, raw_path)
            _atomic_write_csv(kronos, kronos_path)

            successes.append(
                {
                    "symbol": entry.symbol,
                    "yahoo_symbol": entry.yahoo_symbol,
                    "rows": len(validated),
                    "first_timestamp": validated["timestamps"].min().date().isoformat(),
                    "last_timestamp": validated["timestamps"].max().date().isoformat(),
                    "raw_file": str(raw_path),
                    "kronos_file": str(kronos_path),
                    "ohlc_repair_count": len(ohlc_repairs),
                    "ohlc_repairs": ohlc_repairs,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "symbol": entry.symbol,
                    "yahoo_symbol": entry.yahoo_symbol,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    repair_count = sum(item["ohlc_repair_count"] for item in successes)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "provider": "Yahoo Finance",
            "client": "yfinance",
            "interval": "1d",
            "auto_adjust": False,
            "research_only": True,
        },
        "request": {
            "universe_file": str(universe),
            "as_of": as_of.isoformat() if isinstance(as_of, date) else as_of,
            "start": start,
            "end_exclusive": end,
            "symbols": [entry.symbol for entry in entries],
            "retries": retries,
        },
        "summary": {
            "requested": len(entries),
            "succeeded": len(successes),
            "failed": len(failures),
        },
        "quality": {
            "ohlc_repairs": repair_count,
        },
        "successes": successes,
        "failures": failures,
        "limitations": [
            "Yahoo data is for personal research and is not an official Borsa Istanbul feed.",
            "The amount column is estimated as typical price multiplied by volume.",
            "Raw CSV files preserve normalized Yahoo values; Kronos CSV files may minimally expand high or low to contain open and close, with every repair audited in the manifest.",
            "Corporate-action and historical-index-membership accuracy require licensed data before live use.",
        ],
    }
    _atomic_write_json(manifest, output / "manifest.json")

    exit_code = 1 if fail_on_error and failures else 0
    return manifest, exit_code


def build_parser() -> argparse.ArgumentParser:
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    parser = argparse.ArgumentParser(
        description=(
            "Download daily BIST candles from Yahoo Finance for personal research. "
            "The --end date is exclusive."
        )
    )
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=tomorrow)
    parser.add_argument("--as-of", default=None)
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Optional symbols separated by spaces or commas, for example THYAO ASELS.",
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return a non-zero exit code when any symbol fails.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest, exit_code = run_pipeline(
            universe_path=args.universe,
            output_dir=args.output,
            start=args.start,
            end=args.end,
            symbols=args.symbols,
            retries=args.retries,
            sleep_seconds=args.sleep_seconds,
            fail_on_error=args.fail_on_error,
            as_of=args.as_of,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    summary = manifest["summary"]
    print(
        "BIST Yahoo download complete: "
        f"{summary['succeeded']} succeeded, {summary['failed']} failed, "
        f"{manifest['quality']['ohlc_repairs']} audited OHLC repair(s)."
    )
    print(f"Manifest: {Path(args.output) / 'manifest.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
