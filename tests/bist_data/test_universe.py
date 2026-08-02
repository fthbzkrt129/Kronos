from datetime import date
from pathlib import Path

import pytest

from bist_data.universe import UniverseError, load_universe


def write_universe(tmp_path: Path, rows: str) -> Path:
    path = tmp_path / "universe.csv"
    path.write_text(
        "symbol,name,valid_from,valid_to\n" + rows,
        encoding="utf-8",
    )
    return path


def test_load_universe_filters_membership_dates_and_builds_yahoo_symbol(tmp_path: Path):
    path = write_universe(
        tmp_path,
        "THYAO,Turk Hava Yollari,2026-07-01,2026-09-30\n"
        "ASELS,Aselsan,2026-10-01,\n",
    )

    entries = load_universe(path, as_of=date(2026, 8, 2))

    assert [entry.symbol for entry in entries] == ["THYAO"]
    assert entries[0].yahoo_symbol == "THYAO.IS"


def test_load_universe_rejects_duplicate_symbols(tmp_path: Path):
    path = write_universe(
        tmp_path,
        "THYAO,Turk Hava Yollari,2026-07-01,2026-09-30\n"
        "THYAO,Duplicate,2026-07-01,2026-09-30\n",
    )

    with pytest.raises(UniverseError, match="duplicate"):
        load_universe(path)


def test_load_universe_rejects_invalid_date_range(tmp_path: Path):
    path = write_universe(
        tmp_path,
        "THYAO,Turk Hava Yollari,2026-10-01,2026-09-30\n",
    )

    with pytest.raises(UniverseError, match="valid_from"):
        load_universe(path)


def test_load_universe_accepts_iso_date_string(tmp_path: Path):
    path = write_universe(
        tmp_path,
        "THYAO,Turk Hava Yollari,2026-07-01,2026-09-30\n",
    )

    entries = load_universe(path, as_of="2026-08-02")

    assert len(entries) == 1
