from __future__ import annotations
from dataclasses import asdict,dataclass
from datetime import date
import hashlib,json
MODEL_TOKENIZER_PAIRS={
 "NeoQuasar/Kronos-mini":("NeoQuasar/Kronos-Tokenizer-2k",512),
 "NeoQuasar/Kronos-small":("NeoQuasar/Kronos-Tokenizer-base",512),
}
def _fingerprint(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
@dataclass(frozen=True,slots=True)
class EvaluationConfig:
    schema_version:int=1; start_date:str="2023-01-01"; end_date:str="2026-08-02"; lookback:int=400; horizon:int=5; calendar_coverage:float=.8; minimum_ranking_cohort:int=20
    model_id:str="NeoQuasar/Kronos-mini"; tokenizer_id:str="NeoQuasar/Kronos-Tokenizer-2k"; model_revision:str|None=None; tokenizer_revision:str|None=None
    temperature:float=1.; top_p:float=.9; sample_count:int=1; seed:int=20260802; shard_count:int=10
    def __post_init__(self):
        try:s=date.fromisoformat(self.start_date); e=date.fromisoformat(self.end_date)
        except ValueError as exc: raise ValueError("start_date and end_date must use YYYY-MM-DD") from exc
        if e<s: raise ValueError("end_date must not precede start_date")
        for f in ("lookback","horizon","sample_count","shard_count","minimum_ranking_cohort"):
            if getattr(self,f)<=0: raise ValueError(f"{f} must be positive")
        if not 0<self.calendar_coverage<=1: raise ValueError("calendar_coverage must be in (0, 1]")
        if not 0<self.top_p<=1: raise ValueError("top_p must be in (0, 1]")
        if self.temperature<=0: raise ValueError("temperature must be positive")
    def to_canonical_dict(self):return asdict(self)
    @property
    def fingerprint(self):return _fingerprint(self.to_canonical_dict())
@dataclass(frozen=True,slots=True)
class AdjustedBenchmarkConfig:
    schema_version:int=1; experiment_arm:str="adjusted-mini"; context_view:str="origin_rebased"; scoring_target_view:str="origin_rebased"
    adjustment_formula_version:str="origin-rebased-v1"; material_factor_tolerance:float=1e-8; bootstrap_draws:int=10000; bootstrap_confidence:float=.95; bootstrap_seed:int=20260802
    start_date:str="2023-01-01"; end_date:str="2026-08-02"; lookback:int=400; horizon:int=5; calendar_coverage:float=.8; minimum_ranking_cohort:int=20
    model_id:str="NeoQuasar/Kronos-mini"; tokenizer_id:str="NeoQuasar/Kronos-Tokenizer-2k"; model_revision:str|None=None; tokenizer_revision:str|None=None
    temperature:float=1.; top_p:float=.9; sample_count:int=1; seed:int=20260802; shard_count:int=10
    def __post_init__(self):
        EvaluationConfig(start_date=self.start_date,end_date=self.end_date,lookback=self.lookback,horizon=self.horizon,calendar_coverage=self.calendar_coverage,minimum_ranking_cohort=self.minimum_ranking_cohort,model_id=self.model_id,tokenizer_id=self.tokenizer_id,model_revision=self.model_revision,tokenizer_revision=self.tokenizer_revision,temperature=self.temperature,top_p=self.top_p,sample_count=self.sample_count,seed=self.seed,shard_count=self.shard_count)
        allowed={"raw-mini":("raw","NeoQuasar/Kronos-mini"),"adjusted-mini":("origin_rebased","NeoQuasar/Kronos-mini"),"adjusted-small":("origin_rebased","NeoQuasar/Kronos-small")}
        if self.experiment_arm not in allowed: raise ValueError("unknown experiment_arm")
        view,model=allowed[self.experiment_arm]
        if self.context_view!=view or self.model_id!=model: raise ValueError("invalid arm context/model combination")
        pair=MODEL_TOKENIZER_PAIRS.get(self.model_id)
        if pair is None or pair[0]!=self.tokenizer_id: raise ValueError("unsupported model-tokenizer pair")
        if self.lookback>pair[1]: raise ValueError("lookback exceeds model maximum context")
        if self.scoring_target_view!="origin_rebased": raise ValueError("scoring_target_view must be origin_rebased")
        if self.adjustment_formula_version!="origin-rebased-v1": raise ValueError("unsupported adjustment formula")
        if self.material_factor_tolerance<=0: raise ValueError("material_factor_tolerance must be positive")
        if self.bootstrap_draws<=0: raise ValueError("bootstrap_draws must be positive")
        if not 0<self.bootstrap_confidence<1: raise ValueError("bootstrap_confidence must be in (0, 1)")
    def to_canonical_dict(self):return asdict(self)
    @property
    def fingerprint(self):return _fingerprint(self.to_canonical_dict())
    @property
    def common_protocol_fingerprint(self):
        d=self.to_canonical_dict().copy()
        for k in ("experiment_arm","context_view","model_id","tokenizer_id","model_revision","tokenizer_revision","bootstrap_draws","bootstrap_confidence","bootstrap_seed"): d.pop(k,None)
        return _fingerprint(d)
