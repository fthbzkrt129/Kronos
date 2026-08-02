import json
from pathlib import Path

import pandas as pd

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
    on_disk = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["successes"][0]["symbol"] == "THYAO"


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
    assert manifest["failures"][0]["symbol"] == "ASELS"
