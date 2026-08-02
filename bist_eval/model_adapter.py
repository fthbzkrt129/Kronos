"""Lazy Kronos model loading and validated cohort prediction."""
from __future__ import annotations
import hashlib, random
from pathlib import Path
import numpy as np, pandas as pd
from .windows import ForecastWindow
MODEL_COLUMNS=["open","high","low","close","volume","amount"]
def derive_cohort_seed(base_seed: int, forecast_origin: pd.Timestamp) -> int:
    raw=f"{base_seed}:{pd.Timestamp(forecast_origin).date().isoformat()}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4],"big")
def _seed_everything(seed:int):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except ImportError: pass
class KronosModelAdapter:
    def __init__(self, *, model_id="NeoQuasar/Kronos-mini", tokenizer_id="NeoQuasar/Kronos-Tokenizer-2k", model_revision=None, tokenizer_revision=None, model_path=None, tokenizer_path=None, device=None, temperature=1.0, top_p=0.9, sample_count=1, seed=20260802, predictor=None):
        self.model_id=model_id; self.tokenizer_id=tokenizer_id; self.model_revision=model_revision; self.tokenizer_revision=tokenizer_revision
        self.model_path=Path(model_path) if model_path else None; self.tokenizer_path=Path(tokenizer_path) if tokenizer_path else None
        self.device=device; self.temperature=temperature; self.top_p=top_p; self.sample_count=sample_count; self.seed=seed; self.predictor=predictor
    def load(self):
        if self.predictor is not None: return self
        from model import Kronos,KronosPredictor,KronosTokenizer
        t_source=str(self.tokenizer_path) if self.tokenizer_path else self.tokenizer_id
        m_source=str(self.model_path) if self.model_path else self.model_id
        t_kwargs={} if self.tokenizer_path else ({"revision":self.tokenizer_revision} if self.tokenizer_revision else {})
        m_kwargs={} if self.model_path else ({"revision":self.model_revision} if self.model_revision else {})
        tokenizer=KronosTokenizer.from_pretrained(t_source,**t_kwargs); model=Kronos.from_pretrained(m_source,**m_kwargs)
        tokenizer.eval(); model.eval(); self.predictor=KronosPredictor(model,tokenizer,device=self.device,max_context=512); return self
    def predict_cohort(self, windows: list[ForecastWindow]):
        if not windows: return {}
        self.load(); origin=windows[0].forecast_origin
        if any(w.forecast_origin!=origin or w.target_timestamps!=windows[0].target_timestamps for w in windows): raise ValueError("windows must share one common cohort")
        _seed_everything(derive_cohort_seed(self.seed,origin))
        dfs=[w.context.loc[:,MODEL_COLUMNS] for w in windows]; xs=[w.context["timestamps"] for w in windows]; ys=[pd.Series(w.target_timestamps) for w in windows]
        if not hasattr(self.predictor,"predict_batch"): raise TypeError("production predictor must implement predict_batch")
        outputs=self.predictor.predict_batch(dfs,xs,ys,pred_len=len(windows[0].target_timestamps),T=self.temperature,top_p=self.top_p,sample_count=self.sample_count,verbose=False)
        if len(outputs)!=len(windows): raise ValueError("predictor returned wrong batch length")
        result={}
        for w,out in zip(windows,outputs):
            if not isinstance(out,pd.DataFrame) or "close" not in out.columns or len(out)!=len(w.target_timestamps): raise ValueError("invalid predictor output schema")
            values=pd.to_numeric(out["close"],errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all(): raise ValueError("predictor returned non-finite close values")
            out_index=pd.DatetimeIndex(out.index)
            if tuple(out_index)!=tuple(w.target_timestamps): raise ValueError("predictor timestamps do not match target timestamps")
            result[w.symbol]=values
        return result
