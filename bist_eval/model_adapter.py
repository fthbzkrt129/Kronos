from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MODEL_TOKENIZER_PAIRS

MODEL_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


@dataclass(frozen=True, slots=True)
class PredictionFailure:
    symbol: str
    candidate_month: str
    forecast_origin: pd.Timestamp
    reason_code: str
    reason_detail: str


@dataclass(frozen=True, slots=True)
class CohortPredictionResult:
    predictions: dict[str, np.ndarray]
    failures: tuple[PredictionFailure, ...]


def validate_model_tokenizer_pair(model_id, tokenizer_id):
    pair = MODEL_TOKENIZER_PAIRS.get(model_id)
    if pair is None or pair[0] != tokenizer_id:
        raise ValueError("unsupported model-tokenizer pair")
    return pair[1]


def derive_cohort_seed(base_seed, forecast_origin):
    raw = f"{base_seed}:{pd.Timestamp(forecast_origin).date().isoformat()}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _safe_values(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


class KronosModelAdapter:
    def __init__(
        self,
        *,
        model_id="NeoQuasar/Kronos-mini",
        tokenizer_id="NeoQuasar/Kronos-Tokenizer-2k",
        model_revision=None,
        tokenizer_revision=None,
        model_path=None,
        tokenizer_path=None,
        device=None,
        temperature=1.0,
        top_p=0.9,
        sample_count=1,
        seed=20260802,
        predictor=None,
        max_context=None,
    ):
        allowed = validate_model_tokenizer_pair(model_id, tokenizer_id)
        if max_context is not None and max_context > allowed:
            raise ValueError("max_context exceeds model contract")
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.model_path = Path(model_path) if model_path else None
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else None
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.sample_count = sample_count
        self.seed = seed
        self.predictor = predictor
        self.max_context = max_context or allowed

    def load(self):
        if self.predictor is not None:
            return self
        from model import Kronos, KronosPredictor, KronosTokenizer

        tokenizer_source = str(self.tokenizer_path) if self.tokenizer_path else self.tokenizer_id
        model_source = str(self.model_path) if self.model_path else self.model_id
        tokenizer_kwargs = (
            {}
            if self.tokenizer_path
            else ({"revision": self.tokenizer_revision} if self.tokenizer_revision else {})
        )
        model_kwargs = (
            {}
            if self.model_path
            else ({"revision": self.model_revision} if self.model_revision else {})
        )
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
        model = Kronos.from_pretrained(model_source, **model_kwargs)
        tokenizer.eval()
        model.eval()
        self.predictor = KronosPredictor(
            model,
            tokenizer,
            device=self.device,
            max_context=self.max_context,
        )
        return self

    def _prepare_batch(self, windows):
        if not windows:
            return [], [], []
        origin = windows[0].forecast_origin
        if any(
            window.forecast_origin != origin
            or window.target_timestamps != windows[0].target_timestamps
            for window in windows
        ):
            raise ValueError("windows must share one common cohort")
        if any(len(window.context) > self.max_context for window in windows):
            raise ValueError("context exceeds model maximum")
        _seed_everything(derive_cohort_seed(self.seed, origin))
        frames = [window.context.loc[:, MODEL_COLUMNS] for window in windows]
        history_timestamps = [window.context.timestamps for window in windows]
        target_timestamps = [pd.Series(window.target_timestamps) for window in windows]
        return frames, history_timestamps, target_timestamps

    def predict_cohort_with_failures(self, windows) -> CohortPredictionResult:
        if not windows:
            return CohortPredictionResult({}, ())
        self.load()
        frames, history_timestamps, target_timestamps = self._prepare_batch(windows)
        if not hasattr(self.predictor, "predict_batch"):
            raise TypeError("production predictor must implement predict_batch")
        outputs = self.predictor.predict_batch(
            frames,
            history_timestamps,
            target_timestamps,
            pred_len=len(windows[0].target_timestamps),
            T=self.temperature,
            top_p=self.top_p,
            sample_count=self.sample_count,
            verbose=False,
        )
        if len(outputs) != len(windows):
            raise ValueError("predictor returned wrong batch length")

        predictions: dict[str, np.ndarray] = {}
        failures: list[PredictionFailure] = []
        for window, output in zip(windows, outputs):
            if (
                not isinstance(output, pd.DataFrame)
                or "close" not in output.columns
                or len(output) != len(window.target_timestamps)
            ):
                failures.append(
                    PredictionFailure(
                        window.symbol,
                        window.candidate_month,
                        window.forecast_origin,
                        "invalid_predictor_schema",
                        "predictor output must contain one close value per target timestamp",
                    )
                )
                continue

            values = pd.to_numeric(output.close, errors="coerce").to_numpy(float)
            if not np.isfinite(values).all() or (values <= 0).any():
                failures.append(
                    PredictionFailure(
                        window.symbol,
                        window.candidate_month,
                        window.forecast_origin,
                        "invalid_model_output",
                        "nonpositive_or_nonfinite_close="
                        + json.dumps(_safe_values(values), separators=(",", ":")),
                    )
                )
                continue

            if tuple(pd.DatetimeIndex(output.index)) != tuple(window.target_timestamps):
                failures.append(
                    PredictionFailure(
                        window.symbol,
                        window.candidate_month,
                        window.forecast_origin,
                        "invalid_predictor_timestamps",
                        "predictor timestamps do not match target timestamps",
                    )
                )
                continue
            predictions[window.symbol] = values

        return CohortPredictionResult(predictions, tuple(failures))

    def predict_cohort(self, windows):
        result = self.predict_cohort_with_failures(windows)
        if result.failures:
            first = result.failures[0]
            if first.reason_code == "invalid_predictor_schema":
                raise ValueError("invalid predictor output schema")
            if first.reason_code == "invalid_predictor_timestamps":
                raise ValueError("predictor timestamps do not match target timestamps")
            raise ValueError("predictor returned invalid close values")
        return result.predictions
