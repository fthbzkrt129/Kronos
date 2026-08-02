"""Kronos-ready CSV discovery, validation, and source fingerprints."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from pathlib import Path
import hashlib, math
import pandas as pd
from bist_data.quality import validate_candles
COLUMNS=["timestamps","open","high","low","close","volume","amount"]
def discover_symbol_files(data_dir: Path, symbols: Sequence[str]) -> dict[str, Path]:
    root=Path(data_dir); return {s: root/f"{s}.csv" for s in symbols if (root/f"{s}.csv").is_file()}
def load_symbol_frame(path: Path) -> pd.DataFrame:
    frame=pd.read_csv(path)
    if "amount" not in frame.columns: raise ValueError("amount column is required")
    frame["timestamps"] = pd.to_datetime(frame["timestamps"], errors="raise")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    if not frame["amount"].map(math.isfinite).all(): raise ValueError("amount contains non-finite values")
    validated=validate_candles(frame)
    amount_by_timestamp = frame.set_index("timestamps")["amount"]
    validated["amount"] = validated["timestamps"].map(amount_by_timestamp)
    return validated.loc[:,COLUMNS].sort_values("timestamps").reset_index(drop=True)
def load_timestamp_coverage(files: Mapping[str, Path]) -> dict[str,pd.DatetimeIndex]:
    out={}
    for symbol,path in sorted(files.items()):
        ts=pd.to_datetime(pd.read_csv(path,usecols=["timestamps"])["timestamps"], errors="raise")
        if ts.duplicated().any(): raise ValueError(f"duplicate timestamps for {symbol}")
        out[symbol]=pd.DatetimeIndex(ts.sort_values().unique())
    return out
def source_data_fingerprint(files: Mapping[str, Path], manifest_path: Path | None=None) -> str:
    digest=hashlib.sha256()
    if manifest_path and Path(manifest_path).is_file(): digest.update(Path(manifest_path).read_bytes())
    for symbol,path in sorted(files.items()):
        digest.update(symbol.encode()); digest.update(hashlib.sha256(Path(path).read_bytes()).digest())
    return digest.hexdigest()
