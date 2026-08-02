"""Immutable evaluation configuration and reproducibility fingerprints."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import date
import hashlib, json
@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    schema_version: int = 1
    start_date: str = "2023-01-01"
    end_date: str = "2026-08-02"
    lookback: int = 400
    horizon: int = 5
    calendar_coverage: float = 0.80
    minimum_ranking_cohort: int = 20
    model_id: str = "NeoQuasar/Kronos-mini"
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-2k"
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    temperature: float = 1.0
    top_p: float = 0.9
    sample_count: int = 1
    seed: int = 20260802
    shard_count: int = 10
    def __post_init__(self):
        try: start=date.fromisoformat(self.start_date); end=date.fromisoformat(self.end_date)
        except ValueError as exc: raise ValueError("start_date and end_date must use YYYY-MM-DD") from exc
        if end < start: raise ValueError("end_date must not precede start_date")
        for field in ("lookback","horizon","sample_count","shard_count","minimum_ranking_cohort"):
            if getattr(self,field) <= 0: raise ValueError(f"{field} must be positive")
        if not 0 < self.calendar_coverage <= 1: raise ValueError("calendar_coverage must be in (0, 1]")
        if not 0 < self.top_p <= 1: raise ValueError("top_p must be in (0, 1]")
        if self.temperature <= 0: raise ValueError("temperature must be positive")
    def to_canonical_dict(self): return asdict(self)
    @property
    def fingerprint(self):
        raw=json.dumps(self.to_canonical_dict(), sort_keys=True, separators=(",",":"), ensure_ascii=True)
        return hashlib.sha256(raw.encode()).hexdigest()
