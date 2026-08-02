from pathlib import Path
import numpy as np, pandas as pd, pytest

def make_frame(rows=430,start="2023-01-02",missing=None,close_start=100.0):
    ts=pd.bdate_range(start,periods=rows)
    if missing: ts=ts.delete(missing)
    close=close_start+np.arange(len(ts))*0.1
    return pd.DataFrame({"timestamps":ts,"open":close-.2,"high":close+.5,"low":close-.5,"close":close,"volume":1000+np.arange(len(ts)),"amount":close*(1000+np.arange(len(ts)))})
@pytest.fixture
def frame_factory(): return make_frame
@pytest.fixture
def universe_writer(tmp_path):
    def write(symbols):
        p=tmp_path/"universe.csv"; p.write_text("symbol,name,valid_from,valid_to\n"+"".join(f"{s},{s},2026-07-01,2026-09-30\n" for s in symbols)); return p
    return write
