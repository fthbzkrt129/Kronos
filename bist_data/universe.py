"""Versioned BIST universe parsing and validation."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]+$")
_REQUIRED_COLUMNS = {"symbol", "name", "valid_from", "valid_to"}


class UniverseError(ValueError):
    """Raised when a universe snapshot is malformed."""


@dataclass(frozen=True, slots=True)
class UniverseEntry:
    """One index constituent and its inclusive membership dates."""

    symbol: str
    name: str
    valid_from: date
    valid_to: date | None

    @property
    def yahoo_symbol(self) -> str:
        """Return Yahoo Finance's Borsa Istanbul ticker format."""

        return f"{self.symbol}.IS"

    def is_active_on(self, as_of: date) -> bool:
        """Return whether this constituent is active on ``as_of``."""

        return self.valid_from <= as_of and (
            self.valid_to is None or as_of <= self.valid_to
        )


def _parse_date(value: str, *, field: str, row_number: int) -> date | None:
    value = value.strip()
    if not value and field == "valid_to":
        return None
    if not value:
        raise UniverseError(f"row {row_number}: {field} is required")

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise UniverseError(
            f"row {row_number}: {field} must use YYYY-MM-DD format"
        ) from exc


def _coerce_as_of(value: date | str | None) -> date | None:
    if value is None or isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise UniverseError("as_of must use YYYY-MM-DD format") from exc


def _validate_unique(entries: Iterable[UniverseEntry]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for entry in entries:
        if entry.symbol in seen:
            duplicates.add(entry.symbol)
        seen.add(entry.symbol)
    if duplicates:
        symbols = ", ".join(sorted(duplicates))
        raise UniverseError(f"duplicate symbols found: {symbols}")


def load_universe(
    path: str | Path,
    *,
    as_of: date | str | None = None,
) -> list[UniverseEntry]:
    """Load and validate a versioned universe CSV.

    The expected columns are ``symbol``, ``name``, ``valid_from``, and
    ``valid_to``. Dates are inclusive. When ``as_of`` is supplied, only active
    constituents are returned.
    """

    csv_path = Path(path)
    if not csv_path.is_file():
        raise UniverseError(f"universe file not found: {csv_path}")

    active_date = _coerce_as_of(as_of)
    entries: list[UniverseEntry] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = _REQUIRED_COLUMNS - columns
        if missing:
            raise UniverseError(
                "missing required universe columns: " + ", ".join(sorted(missing))
            )

        for row_number, row in enumerate(reader, start=2):
            symbol = (row.get("symbol") or "").strip().upper()
            name = (row.get("name") or "").strip()
            if not symbol or not _SYMBOL_PATTERN.fullmatch(symbol):
                raise UniverseError(f"row {row_number}: invalid symbol {symbol!r}")
            if not name:
                raise UniverseError(f"row {row_number}: name is required")

            valid_from = _parse_date(
                row.get("valid_from") or "",
                field="valid_from",
                row_number=row_number,
            )
            valid_to = _parse_date(
                row.get("valid_to") or "",
                field="valid_to",
                row_number=row_number,
            )
            assert valid_from is not None
            if valid_to is not None and valid_from > valid_to:
                raise UniverseError(
                    f"row {row_number}: valid_from must not be after valid_to"
                )

            entries.append(
                UniverseEntry(
                    symbol=symbol,
                    name=name,
                    valid_from=valid_from,
                    valid_to=valid_to,
                )
            )

    _validate_unique(entries)
    if active_date is not None:
        entries = [entry for entry in entries if entry.is_active_on(active_date)]

    return entries
