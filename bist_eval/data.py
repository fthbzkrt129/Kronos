from __future__ import annotations
from pathlib import Path
import hashlib,math
import numpy as np,pandas as pd
from bist_data.quality import validate_candles
COLUMNS=["timestamps","open","high","low","close","volume","amount"]
RAW_YAHOO_COLUMNS=["timestamps","open","high","low","close","adj_close","volume","symbol","yahoo_symbol"]
def discover_symbol_files(data_dir,symbols):
    root=Path(data_dir); return {s:root/f"{s}.csv" for s in symbols if (root/f"{s}.csv").is_file()}
def discover_raw_symbol_files(data_dir,symbols): return discover_symbol_files(data_dir,symbols)
def load_symbol_frame(path):
    frame=pd.read_csv(path)
    if "amount" not in frame.columns: raise ValueError("amount column is required")
    frame.timestamps=pd.to_datetime(frame.timestamps,errors="raise"); frame.amount=pd.to_numeric(frame.amount,errors="coerce")
    if not frame.amount.map(math.isfinite).all(): raise ValueError("amount contains non-finite values")
    validated=validate_candles(frame); amount=frame.set_index("timestamps").amount; validated["amount"]=validated.timestamps.map(amount)
    return validated.loc[:,COLUMNS].sort_values("timestamps").reset_index(drop=True)
def load_raw_symbol_frame(path_or_frame):
    frame=path_or_frame.copy() if isinstance(path_or_frame,pd.DataFrame) else pd.read_csv(path_or_frame)
    missing=set(RAW_YAHOO_COLUMNS)-set(frame.columns)
    if missing: raise ValueError("missing raw Yahoo columns: "+", ".join(sorted(missing)))
    out=frame.loc[:,RAW_YAHOO_COLUMNS].copy(); out.timestamps=pd.to_datetime(out.timestamps,errors="raise")
    if out.timestamps.duplicated().any(): raise ValueError("duplicate timestamps are not allowed")
    nums=["open","high","low","close","adj_close","volume"]
    for c in nums: out[c]=pd.to_numeric(out[c],errors="coerce")
    if not np.isfinite(out[nums].to_numpy(float)).all(): raise ValueError("raw Yahoo numeric columns contain non-finite values")
    if (out[["open","high","low","close","adj_close"]]<=0).any().any(): raise ValueError("raw Yahoo prices must be strictly positive")
    if (out.volume<0).any(): raise ValueError("volume must not be negative")
    out=out.sort_values("timestamps").reset_index(drop=True)
    if not out.timestamps.is_monotonic_increasing: raise ValueError("timestamps must be increasing")
    return out
def load_timestamp_coverage(files):
    out={}
    for symbol,path in sorted(files.items()):
        ts=pd.to_datetime(pd.read_csv(path,usecols=["timestamps"]).timestamps,errors="raise")
        if ts.duplicated().any(): raise ValueError(f"duplicate timestamps for {symbol}")
        out[symbol]=pd.DatetimeIndex(ts.sort_values().unique())
    return out
def source_data_fingerprint(files,manifest_path=None):
    d=hashlib.sha256()
    if manifest_path and Path(manifest_path).is_file():d.update(Path(manifest_path).read_bytes())
    for s,p in sorted(files.items()):d.update(s.encode());d.update(hashlib.sha256(Path(p).read_bytes()).digest())
    return d.hexdigest()
