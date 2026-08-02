from __future__ import annotations
import hashlib,random
from pathlib import Path
import numpy as np,pandas as pd
from .config import MODEL_TOKENIZER_PAIRS
MODEL_COLUMNS=["open","high","low","close","volume","amount"]
def validate_model_tokenizer_pair(model_id,tokenizer_id):
    pair=MODEL_TOKENIZER_PAIRS.get(model_id)
    if pair is None or pair[0]!=tokenizer_id: raise ValueError("unsupported model-tokenizer pair")
    return pair[1]
def derive_cohort_seed(base_seed,forecast_origin):
    raw=f"{base_seed}:{pd.Timestamp(forecast_origin).date().isoformat()}".encode();return int.from_bytes(hashlib.sha256(raw).digest()[:4],"big")
def _seed_everything(seed):
    random.seed(seed);np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    except ImportError:pass
class KronosModelAdapter:
    def __init__(self,*,model_id="NeoQuasar/Kronos-mini",tokenizer_id="NeoQuasar/Kronos-Tokenizer-2k",model_revision=None,tokenizer_revision=None,model_path=None,tokenizer_path=None,device=None,temperature=1.,top_p=.9,sample_count=1,seed=20260802,predictor=None,max_context=None):
        allowed=validate_model_tokenizer_pair(model_id,tokenizer_id)
        if max_context is not None and max_context>allowed: raise ValueError("max_context exceeds model contract")
        self.model_id=model_id;self.tokenizer_id=tokenizer_id;self.model_revision=model_revision;self.tokenizer_revision=tokenizer_revision
        self.model_path=Path(model_path) if model_path else None;self.tokenizer_path=Path(tokenizer_path) if tokenizer_path else None;self.device=device
        self.temperature=temperature;self.top_p=top_p;self.sample_count=sample_count;self.seed=seed;self.predictor=predictor;self.max_context=max_context or allowed
    def load(self):
        if self.predictor is not None:return self
        from model import Kronos,KronosPredictor,KronosTokenizer
        t_source=str(self.tokenizer_path) if self.tokenizer_path else self.tokenizer_id;m_source=str(self.model_path) if self.model_path else self.model_id
        t_kwargs={} if self.tokenizer_path else ({"revision":self.tokenizer_revision} if self.tokenizer_revision else {});m_kwargs={} if self.model_path else ({"revision":self.model_revision} if self.model_revision else {})
        tok=KronosTokenizer.from_pretrained(t_source,**t_kwargs);model=Kronos.from_pretrained(m_source,**m_kwargs);tok.eval();model.eval();self.predictor=KronosPredictor(model,tok,device=self.device,max_context=self.max_context);return self
    def predict_cohort(self,windows):
        if not windows:return {}
        self.load();origin=windows[0].forecast_origin
        if any(w.forecast_origin!=origin or w.target_timestamps!=windows[0].target_timestamps for w in windows):raise ValueError("windows must share one common cohort")
        if any(len(w.context)>self.max_context for w in windows):raise ValueError("context exceeds model maximum")
        _seed_everything(derive_cohort_seed(self.seed,origin))
        dfs=[w.context.loc[:,MODEL_COLUMNS] for w in windows];xs=[w.context.timestamps for w in windows];ys=[pd.Series(w.target_timestamps) for w in windows]
        if not hasattr(self.predictor,"predict_batch"):raise TypeError("production predictor must implement predict_batch")
        outs=self.predictor.predict_batch(dfs,xs,ys,pred_len=len(windows[0].target_timestamps),T=self.temperature,top_p=self.top_p,sample_count=self.sample_count,verbose=False)
        if len(outs)!=len(windows):raise ValueError("predictor returned wrong batch length")
        result={}
        for w,out in zip(windows,outs):
            if not isinstance(out,pd.DataFrame) or "close" not in out.columns or len(out)!=len(w.target_timestamps):raise ValueError("invalid predictor output schema")
            vals=pd.to_numeric(out.close,errors="coerce").to_numpy(float)
            if not np.isfinite(vals).all() or (vals<=0).any():raise ValueError("predictor returned invalid close values")
            if tuple(pd.DatetimeIndex(out.index))!=tuple(w.target_timestamps):raise ValueError("predictor timestamps do not match target timestamps")
            result[w.symbol]=vals
        return result
