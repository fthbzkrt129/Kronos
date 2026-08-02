import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.download_bist_yahoo import run_pipeline


def write_universe(tmp_path: Path) -> Path:
    path = tmp_path / "universe.csv"
    path.write_text(
        "symbol,name,valid_from,valid_to\n"
        "THYAO,Turk Hava Yollari,2026-07-01,2026-09-30\n"
        "ASELS,Aselsan,2026-07-01,2026-09-30\n",
        encoding="utf-8",
    )
    return path


def good_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [10.0],
            "High": [12.0],
            "Low": [9.0],
            "Close": [11.0],
            "Adj Close": [10.5],
            "Volume": [100],
        },
        index=pd.DatetimeIndex(["2026-07-30"], name="Date"),
    )


def invalid_envelope_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [2.506153],
            "High": [2.506153],
            "Low": [2.506153],
            "Close": [2.504615],
            "Adj Close": [2.504615],
            "Volume": [71201],
        },
        index=pd.DatetimeIndex(["2022-06-27"], name="Date"),
    )


def test_run_pipeline_writes_raw_kronos_and_manifest(tmp_path: Path):
    universe = write_universe(tmp_path)
    output = tmp_path / "output"

    manifest, exit_code = run_pipeline(
        universe_path=universe,
        output_dir=output,
        start="2026-01-01",
        end="2026-08-01",
        symbols=["THYAO"],
        retries=0,
        sleep_seconds=0,
        fail_on_error=True,
        downloader=lambda symbol, **kwargs: good_frame(),
    )

    assert exit_code == 0
    assert (output / "raw" / "THYAO.csv").is_file()
    assert (output / "kronos" / "THYAO.csv").is_file()
    assert manifest["summary"] == {"requested": 1, "succeeded": 1, "failed": 0}
    assert manifest["quality"] == {"ohlc_repairs": 0}
    on_disk = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["successes"][0]["symbol"] == "THYAO"
    assert on_disk["successes"][0]["ohlc_repairs"] == []


def test_run_pipeline_preserves_raw_and_audits_kronos_ohlc_repair(tmp_path: Path):
    universe = write_universe(tmp_path)
    output = tmp_path / "output"

    manifest, exit_code = run_pipeline(
        universe_path=universe,
        output_dir=output,
        start="2022-01-01",
        end="2023-01-01",
        symbols=["THYAO"],
        retries=0,
        sleep_seconds=0,
        fail_on_error=True,
        downloader=lambda symbol, **kwargs: invalid_envelope_frame(),
    )

    raw = pd.read_csv(output / "raw" / "THYAO.csv")
    kronos = pd.read_csv(output / "kronos" / "THYAO.csv")

    assert exit_code == 0
    assert raw.loc[0, "low"] == pytest.approx(2.506153)
    assert kronos.loc[0, "low"] == pytest.approx(2.504615)
    assert manifest["quality"] == {"ohlc_repairs": 1}
    assert manifest["successes"][0]["ohlc_repair_count"] == 1
    assert manifest["successes"][0]["ohlc_repairs"][0] == {
        "timestamp": "2022-06-27",
        "column": "low",
        "original_value": pytest.approx(2.506153),
        "repaired_value": pytest.approx(2.504615),
        "reason": "expand_ohlc_envelope",
    }


def test_run_pipeline_records_partial_failure_and_strict_exit_code(tmp_path: Path):
    universe = write_universe(tmp_path)

    def mixed_download(symbol, **kwargs):
        if symbol == "ASELS.IS":
            raise RuntimeError("provider unavailable")
        return good_frame()

    manifest, exit_code = run_pipeline(
        universe_path=universe,
        output_dir=tmp_path / "output",
        start="2026-01-01",
        end="2026-08-01",
        symbols=None,
        retries=0,
        sleep_seconds=0,
        fail_on_error=True,
        downloader=mixed_download,
    )

    assert exit_code == 1
    assert manifest["summary"] == {"requested": 2, "succeeded": 1, "failed": 1}
    assert manifest["quality"] == {"ohlc_repairs": 0}
    assert manifest["failures"][0]["symbol"] == "ASELS"
