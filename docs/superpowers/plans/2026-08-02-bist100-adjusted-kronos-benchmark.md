# BIST 100 Adjusted-Price Kronos Benchmark Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-safe paired benchmark that compares raw-context Kronos-mini, origin-rebased-context Kronos-mini, and origin-rebased-context Kronos-small on identical BIST 100 monthly windows and a common origin-rebased scoring target.

**Architecture:** Preserve the existing zero-shot evaluator and add benchmark-specific adjustment, orchestration, comparison, and reporting modules. Raw Yahoo rows remain immutable; each adjusted context is generated in memory relative to its own forecast origin, target adjustment occurs only after prediction, and the reducer performs matched-window comparisons with symbol- and origin-clustered bootstrap intervals.

**Tech Stack:** Python 3.11, pandas, NumPy, PyTorch, Hugging Face Hub, Kronos-mini, Kronos-small, pytest, GitHub Actions, CSV/JSON/Markdown artifacts.

---

## Source of Truth

- Approved design: `docs/superpowers/specs/2026-08-02-bist100-adjusted-kronos-benchmark-design.md`
- Existing evaluator: `bist_eval/`
- Existing evaluator CLI: `scripts/evaluate_bist100_zero_shot.py`
- Existing reducer: `scripts/reduce_bist100_zero_shot.py`
- Existing asset resolver: `scripts/resolve_kronos_assets.py`
- Existing raw Yahoo schema: `timestamps,open,high,low,close,adj_close,volume,symbol,yahoo_symbol`
- Existing universe: `data/universes/xu100_2026_q3.csv`
- Verified source artifact from the first BIST data run: artifact `8826613095`, SHA-256 `51578854a6fc341f34bff8bea4cf8f57d900006f2a35e2a34e983d6389933f2c`

The implementation must not modify Kronos model internals, fine-tune models, create portfolios, place orders, or describe the result as a historical point-in-time BIST 100 backtest.

## Locked File Structure

### New benchmark modules

- Create `bist_eval/adjustments.py` — raw Yahoo factor validation, origin rebasing, post-prediction target transformation, diagnostics, and fingerprints.
- Create `bist_eval/benchmark.py` — benchmark-window construction and per-shard arm orchestration.
- Create `bist_eval/comparison.py` — arm compatibility, paired joins, winner labels, clustered bootstrap, and decisions.

### Existing modules to extend without breaking old APIs

- Modify `bist_eval/config.py` — add immutable benchmark arm configuration while preserving `EvaluationConfig`.
- Modify `bist_eval/data.py` — add strict raw Yahoo loading while preserving Kronos-ready loading.
- Modify `bist_eval/windows.py` — add model-facing windows that contain no target values and internal scoring records.
- Modify `bist_eval/model_adapter.py` — validate explicit model/tokenizer pairs and configurable maximum context.
- Modify `bist_eval/metrics.py` — add common-target log-return metrics and arm-aware aggregates.
- Modify `bist_eval/sharding.py` — add benchmark arm-set manifest validation.
- Modify `bist_eval/reporting.py` — add benchmark shard/final schemas and report writer while preserving old writers.

### Commands

- Create `scripts/validate_bist_adjustment_factors.py` — validate raw files and write factor provenance only; never write static adjusted model-input CSVs.
- Create `scripts/evaluate_bist100_adjusted_benchmark.py` — run either the paired mini arms or the small arm for one deterministic shard/subset.
- Create `scripts/reduce_bist100_adjusted_benchmark.py` — validate 20 arm shards, combine outputs, compare arms, bootstrap, and report.

### Workflows and docs

- Modify `.github/workflows/bist-eval-tests.yml` — include new modules/scripts in network-free PR checks.
- Create `.github/workflows/bist-adjusted-benchmark-smoke.yml` — manual two-symbol/two-origin three-arm real-model smoke.
- Create `.github/workflows/bist100-adjusted-kronos-benchmark.yml` — manual immutable-data, 20-shard full benchmark.
- Create `docs/bist-adjusted-kronos-benchmark.md` — protocol, commands, interpretation, and limitations.
- Modify `.gitignore` — ignore generated factor manifests, benchmark artifacts, results, and model snapshots.

### Tests

- Modify `tests/bist_eval/conftest.py`.
- Modify `tests/bist_eval/test_config.py`.
- Modify `tests/bist_eval/test_windows.py`.
- Modify `tests/bist_eval/test_model_adapter.py`.
- Modify `tests/bist_eval/test_metrics.py`.
- Modify `tests/bist_eval/test_sharding.py`.
- Modify `tests/bist_eval/test_reporting.py`.
- Modify `tests/bist_eval/test_cli.py`.
- Create `tests/bist_eval/test_adjustments.py`.
- Create `tests/bist_eval/test_benchmark.py`.
- Create `tests/bist_eval/test_comparison.py`.

Do not add SciPy. Spearman remains average-rank plus Pearson. Bootstrap uses NumPy only.

---

## Chunk 1: Benchmark Contracts and Raw Factor Provenance

### Task 1: Add immutable arm configuration and explicit model-tokenizer registry

**Files:**
- Modify: `bist_eval/config.py`
- Modify: `tests/bist_eval/test_config.py`

- [ ] **Step 1: Write failing configuration tests**

Add tests that preserve all existing `EvaluationConfig` behavior and lock the new contract:

```python
from bist_eval.config import AdjustedBenchmarkConfig, MODEL_TOKENIZER_PAIRS


def test_adjusted_benchmark_defaults():
    config = AdjustedBenchmarkConfig()
    assert config.lookback == 400
    assert config.horizon == 5
    assert config.adjustment_formula_version == "origin-rebased-v1"
    assert config.material_factor_tolerance == 1e-8
    assert config.bootstrap_draws == 10_000
    assert config.bootstrap_confidence == 0.95


def test_supported_model_pairs_are_explicit():
    assert MODEL_TOKENIZER_PAIRS["NeoQuasar/Kronos-mini"] == (
        "NeoQuasar/Kronos-Tokenizer-2k",
        512,
    )
    assert MODEL_TOKENIZER_PAIRS["NeoQuasar/Kronos-small"] == (
        "NeoQuasar/Kronos-Tokenizer-base",
        512,
    )


def test_arm_changes_fingerprint():
    raw = AdjustedBenchmarkConfig(experiment_arm="raw-mini")
    adjusted = AdjustedBenchmarkConfig(experiment_arm="adjusted-mini")
    assert raw.fingerprint != adjusted.fingerprint
    assert raw.common_protocol_fingerprint == adjusted.common_protocol_fingerprint
```

Reject unknown arms, unsupported model-tokenizer pairs, nonpositive tolerance/draws, invalid confidence, lookback above the selected model maximum context, and target/context view combinations not allowed by the spec.

- [ ] **Step 2: Run tests and verify failure**

```bash
python -m pytest tests/bist_eval/test_config.py -v
```

Expected: missing `AdjustedBenchmarkConfig`/registry failures.

- [ ] **Step 3: Implement the minimal contract**

Add without changing old defaults:

```python
MODEL_TOKENIZER_PAIRS = {
    "NeoQuasar/Kronos-mini": ("NeoQuasar/Kronos-Tokenizer-2k", 512),
    "NeoQuasar/Kronos-small": ("NeoQuasar/Kronos-Tokenizer-base", 512),
}

@dataclass(frozen=True, slots=True)
class AdjustedBenchmarkConfig:
    schema_version: int = 1
    experiment_arm: str = "adjusted-mini"
    context_view: str = "origin_rebased"
    scoring_target_view: str = "origin_rebased"
    adjustment_formula_version: str = "origin-rebased-v1"
    material_factor_tolerance: float = 1e-8
    bootstrap_draws: int = 10_000
    bootstrap_confidence: float = 0.95
    bootstrap_seed: int = 20260802
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
```

Provide `to_canonical_dict()`, full `fingerprint`, and a `common_protocol_fingerprint` excluding arm/model identity but including universe protocol, date range, lookback, horizon, calendar rule, scoring-target view, formula version, tolerance, sampling parameters, and seed.

- [ ] **Step 4: Run tests and verify pass**

```bash
python -m pytest tests/bist_eval/test_config.py -v
```

- [ ] **Step 5: Commit**

```bash
git add bist_eval/config.py tests/bist_eval/test_config.py
git commit -m "feat(eval): add adjusted benchmark contracts"
```

### Task 2: Load immutable raw Yahoo frames and validate provider factors

**Files:**
- Modify: `bist_eval/data.py`
- Create: `bist_eval/adjustments.py`
- Modify: `tests/bist_eval/conftest.py`
- Create: `tests/bist_eval/test_adjustments.py`

- [ ] **Step 1: Extend synthetic fixtures**

Add `raw_frame_factory` with columns:

```text
timestamps,open,high,low,close,adj_close,volume,symbol,yahoo_symbol
```

Allow deterministic split-like/dividend-like factor changes, missing `adj_close`, zero/negative prices, duplicate timestamps, and non-finite values.

- [ ] **Step 2: Write failing raw-loader/factor tests**

Test:

```python
def test_provider_factor_is_adj_close_over_close(raw_frame_factory):
    raw = raw_frame_factory(rows=3, factors=[0.5, 0.5, 1.0])
    validated = load_raw_symbol_frame(raw)
    factors = validate_provider_factors(validated)
    np.testing.assert_allclose(factors.to_numpy(), [0.5, 0.5, 1.0])
```

Also require:

- exact stable column order,
- finite OHLC/adj-close/volume,
- strictly positive OHLC and `adj_close`,
- nonnegative volume,
- unique monotonic timestamps,
- no fill/interpolation/clipping,
- deterministic raw-file and factor-series fingerprints.

- [ ] **Step 3: Run tests and verify failure**

```bash
python -m pytest tests/bist_eval/test_adjustments.py -k "factor or raw" -v
```

- [ ] **Step 4: Implement strict loading and factor validation**

In `bist_eval/data.py` add:

```python
RAW_YAHOO_COLUMNS = [
    "timestamps", "open", "high", "low", "close", "adj_close",
    "volume", "symbol", "yahoo_symbol",
]

def discover_raw_symbol_files(data_dir: Path, symbols: Sequence[str]) -> dict[str, Path]: ...
def load_raw_symbol_frame(path: Path) -> pd.DataFrame: ...
```

In `bist_eval/adjustments.py` add:

```python
@dataclass(frozen=True, slots=True)
class FactorDiagnostics:
    symbol: str
    rows: int
    first_timestamp: pd.Timestamp
    last_timestamp: pd.Timestamp
    minimum_factor: float
    maximum_factor: float
    materially_changed_rows: int
    factor_fingerprint: str


def provider_factor(frame: pd.DataFrame) -> pd.Series: ...
def validate_provider_factors(frame: pd.DataFrame) -> pd.Series: ...
def build_factor_diagnostics(symbol: str, frame: pd.DataFrame, tolerance: float) -> FactorDiagnostics: ...
```

Fingerprint serialized timestamps and IEEE-754 factor values in stable row order. Do not round before hashing.

- [ ] **Step 5: Run tests and verify pass**

```bash
python -m pytest tests/bist_eval/test_adjustments.py -k "factor or raw" -v
```

- [ ] **Step 6: Commit**

```bash
git add bist_eval/data.py bist_eval/adjustments.py tests/bist_eval/conftest.py tests/bist_eval/test_adjustments.py
git commit -m "feat(eval): add raw factor provenance"
```

### Task 3: Add the factor-provenance preparation CLI

**Files:**
- Create: `scripts/validate_bist_adjustment_factors.py`
- Modify: `tests/bist_eval/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Using temporary raw CSV files, assert the command writes only:

```text
factor_manifest.json
factor_diagnostics.csv
COMPLETED
```

It must not create `adjusted/`, `kronos/`, or per-symbol model-input files. Strict mode fails on one invalid symbol and leaves no `COMPLETED` marker.

- [ ] **Step 2: Run tests and verify failure**

```bash
python -m pytest tests/bist_eval/test_cli.py -k "factor_manifest" -v
```

- [ ] **Step 3: Implement CLI and atomic manifest output**

Arguments:

```text
--raw-dir
--source-manifest
--universe
--output
--material-factor-tolerance
--symbols
--strict
```

Manifest fields must include source artifact ID/digest when supplied, source manifest digest, universe digest, formula version, tolerance, symbol counts, failures, per-symbol factor fingerprints, aggregate factor fingerprint, and explicit statement that no static adjusted model-input data was written.

- [ ] **Step 4: Run tests and verify pass**

```bash
python -m pytest tests/bist_eval/test_cli.py -k "factor_manifest" -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_bist_adjustment_factors.py tests/bist_eval/test_cli.py
git commit -m "feat(eval): add factor provenance command"
```

---

## Chunk 2: Leakage-Safe Window and Adjustment Boundaries

### Task 4: Separate model-facing context from held-out scoring data

**Files:**
- Modify: `bist_eval/windows.py`
- Modify: `tests/bist_eval/test_windows.py`

- [ ] **Step 1: Write failing structural leakage tests**

Lock separate immutable types:

```python
@dataclass(frozen=True, slots=True)
class PredictionWindow:
    symbol: str
    candidate_month: str
    forecast_origin: pd.Timestamp
    target_timestamps: tuple[pd.Timestamp, ...]
    context: pd.DataFrame

@dataclass(frozen=True, slots=True)
class ScoringRecord:
    symbol: str
    candidate_month: str
    forecast_origin: pd.Timestamp
    target_timestamps: tuple[pd.Timestamp, ...]
    raw_target: pd.DataFrame
    target_provider_factors: np.ndarray
```

Tests must prove `PredictionWindow` has no target frame, raw target values, or target factors. Existing `ForecastWindow` and `build_symbol_windows` tests must continue passing.

- [ ] **Step 2: Run tests and verify failure**

```bash
python -m pytest tests/bist_eval/test_windows.py -v
```

- [ ] **Step 3: Implement benchmark window bundles**

Add an internal `BenchmarkWindowBundle` containing raw context, provider context factors, a `PredictionWindow` factory, and a private scoring record. The public function returns bundles only to benchmark orchestration; model/baseline functions receive only `PredictionWindow.context`.

```python
def build_benchmark_windows(
    symbol: str,
    raw_frame: pd.DataFrame,
    cohorts: list[MonthlyCohort],
    *,
    lookback: int,
    horizon: int,
) -> tuple[list[BenchmarkWindowBundle], list[SkipRecord]]: ...
```

Use identical eligibility and skip codes as the existing evaluator. Add factor-specific skip/error details only when the raw frame is invalid.

- [ ] **Step 4: Run tests and verify pass**

```bash
python -m pytest tests/bist_eval/test_windows.py -v
```

- [ ] **Step 5: Commit**

```bash
git add bist_eval/windows.py tests/bist_eval/test_windows.py
git commit -m "feat(eval): isolate benchmark targets from predictors"
```

### Task 5: Implement origin-rebased contexts and post-prediction targets

**Files:**
- Modify: `bist_eval/adjustments.py`
- Modify: `tests/bist_eval/test_adjustments.py`

- [ ] **Step 1: Write failing rebasing tests**

Lock formulas:

```python
relative_factor_t = provider_factor_t / provider_factor_origin
rebased_price_t = raw_price_t * relative_factor_t
rebased_amount_t = ((rebased_high_t + rebased_low_t + rebased_close_t) / 3) * raw_volume_t
```

Required assertions:

- origin relative factor is exactly/approximately `1.0`,
- rebased origin OHLC equals raw origin OHLC,
- volume is unchanged,
- target transformation is a separate function called after prediction,
- multiplying every factor through a date after the final target by a common constant leaves all relative values unchanged,
- changing target factors changes scored targets but not raw/rebased context or predictions from fake adapters,
- no static adjusted CSV is needed.

- [ ] **Step 2: Run tests and verify failure**

```bash
python -m pytest tests/bist_eval/test_adjustments.py -k "rebase or leakage or exposure" -v
```

- [ ] **Step 3: Implement transformation and diagnostics**

```python
@dataclass(frozen=True, slots=True)
class ExposureDiagnostics:
    context_factor_changed: bool
    target_factor_changed: bool
    context_max_abs_log_step: float
    target_max_abs_log_from_origin: float
    exposure_bucket: str


def rebase_context(raw_context: pd.DataFrame, factors: pd.Series, origin_factor: float) -> tuple[pd.DataFrame, list[dict]]: ...
def transform_target_after_prediction(raw_target: pd.DataFrame, target_factors: np.ndarray, origin_factor: float) -> pd.DataFrame: ...
def classify_exposure(context_factors, target_factors, origin_factor, tolerance) -> ExposureDiagnostics: ...
```

Definitions:

```python
context_max_abs_log_step = max(abs(diff(log(context_factors))))
target_max_abs_log_from_origin = max(abs(log(target_factors / origin_factor)))
```

Bucket values: `no_material_change` or `material_factor_change`.

Pass rebased context through `repair_ohlc_envelope` then `validate_candles`; record every envelope repair. Model input price fields must be strictly positive.

- [ ] **Step 4: Run tests and verify pass**

```bash
python -m pytest tests/bist_eval/test_adjustments.py -v
```

- [ ] **Step 5: Commit**

```bash
git add bist_eval/adjustments.py tests/bist_eval/test_adjustments.py
git commit -m "feat(eval): add origin-rebased window transforms"
```

---

## Chunk 3: Model Pairing, Metrics, and Arm Orchestration

### Task 6: Enforce model-tokenizer pairing in the lazy adapter

**Files:**
- Modify: `bist_eval/model_adapter.py`
- Modify: `tests/bist_eval/test_model_adapter.py`

- [ ] **Step 1: Write failing pairing tests**

Cover valid mini/2k and small/base pairs, invalid cross-pairs, lookback above max context, local asset paths with recorded IDs, and preservation of lazy imports.

```python
def test_small_requires_base_tokenizer():
    with pytest.raises(ValueError, match="model-tokenizer"):
        KronosModelAdapter(
            model_id="NeoQuasar/Kronos-small",
            tokenizer_id="NeoQuasar/Kronos-Tokenizer-2k",
        )
```

Confirm the adapter accepts `PredictionWindow` objects only and cannot read target data.

- [ ] **Step 2: Run tests and verify failure**

```bash
python -m pytest tests/bist_eval/test_model_adapter.py -v
```

- [ ] **Step 3: Implement explicit validation**

Add `validate_model_tokenizer_pair()` using the registry in `config.py`. Set predictor `max_context` from the registry rather than a silent hard-coded assumption. Preserve existing mini behavior.

For paired raw-mini/adjusted-mini calls, reset the same cohort seed before each arm so stochastic sampling is aligned.

- [ ] **Step 4: Run tests and verify pass**

```bash
python -m pytest tests/bist_eval/test_model_adapter.py -v
```

- [ ] **Step 5: Commit**

```bash
git add bist_eval/model_adapter.py tests/bist_eval/test_model_adapter.py
git commit -m "feat(eval): validate Kronos model-tokenizer pairs"
```

### Task 7: Add common-target log-return metrics and arm-aware aggregates

**Files:**
- Modify: `bist_eval/metrics.py`
- Modify: `tests/bist_eval/test_metrics.py`

- [ ] **Step 1: Write failing metric tests**

Add prediction rows with `experiment_arm` and exposure fields. Lock:

```python
predicted_log_return = np.log(predicted_final_close / history_last_close)
actual_log_return = np.log(actual_final_close / history_last_close)
log_return_abs_error = abs(predicted_log_return - actual_log_return)
```

Reject/noncompute nonpositive prices; do not use epsilon. Test arm-aware symbol/period/ranking aggregation and exposure-bucket summaries. Existing metric tests must remain green.

- [ ] **Step 2: Run tests and verify failure**

```bash
python -m pytest tests/bist_eval/test_metrics.py -v
```

- [ ] **Step 3: Implement backward-compatible benchmark metrics**

Add new functions rather than changing old output unexpectedly:

```python
def compute_benchmark_window_metrics(predictions: pd.DataFrame) -> pd.DataFrame: ...
def aggregate_benchmark_symbol_metrics(window_metrics: pd.DataFrame) -> pd.DataFrame: ...
def aggregate_benchmark_period_metrics(window_metrics: pd.DataFrame) -> pd.DataFrame: ...
def compute_benchmark_ranking_metrics(window_metrics: pd.DataFrame, minimum_cohort: int) -> pd.DataFrame: ...
```

Group keys include `experiment_arm` and `method`. Ranking uses predicted/realized log returns. Keep simple returns for readable diagnostics.

- [ ] **Step 4: Run tests and verify pass**

```bash
python -m pytest tests/bist_eval/test_metrics.py -v
```

- [ ] **Step 5: Commit**

```bash
git add bist_eval/metrics.py tests/bist_eval/test_metrics.py
git commit -m "feat(eval): add benchmark log-return metrics"
```

### Task 8: Build the paired-arm benchmark engine

**Files:**
- Create: `bist_eval/benchmark.py`
- Create: `tests/bist_eval/test_benchmark.py`

- [ ] **Step 1: Write failing orchestration tests with fake adapters**

Test mini mode:

- one raw load creates both `raw-mini` and `adjusted-mini`,
- same symbols/cohorts/target timestamps,
- same common rebased actual target,
- raw context and rebased context differ when factors change,
- identical cohort seed is applied to both arms,
- adjusted baselines are computed once,
- target transformation occurs after both model predictions,
- no adapter receives target values/factors.

Test small mode emits only `adjusted-small` model rows and no duplicate baselines.

- [ ] **Step 2: Run tests and verify failure**

```bash
python -m pytest tests/bist_eval/test_benchmark.py -v
```

- [ ] **Step 3: Implement engine entry points**

```python
@dataclass(frozen=True, slots=True)
class BenchmarkShardResult:
    predictions: pd.DataFrame
    window_metrics: pd.DataFrame
    skips: pd.DataFrame
    manifest: dict


def run_mini_pair_shard(..., adapter: KronosModelAdapter) -> BenchmarkShardResult: ...
def run_small_shard(..., adapter: KronosModelAdapter) -> BenchmarkShardResult: ...
```

For every bundle and cohort:

1. build raw `PredictionWindow`,
2. build rebased `PredictionWindow`,
3. predict raw-mini,
4. reset seed and predict adjusted-mini,
5. compute adjusted baselines from rebased context once,
6. for small mode predict adjusted-small,
7. only then transform target and create scored rows.

Prediction key:

```text
(experiment_arm, symbol, forecast_origin, target_timestamp, method)
```

Baseline arm value: `adjusted-baselines`; methods remain `last_close`, `momentum_20`, `linear_trend_20`.

- [ ] **Step 4: Run tests and verify pass**

```bash
python -m pytest tests/bist_eval/test_benchmark.py -v
```

- [ ] **Step 5: Commit**

```bash
git add bist_eval/benchmark.py tests/bist_eval/test_benchmark.py
git commit -m "feat(eval): add three-arm benchmark engine"
```

### Task 9: Add the benchmark shard CLI

**Files:**
- Create: `scripts/evaluate_bist100_adjusted_benchmark.py`
- Modify: `tests/bist_eval/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Test modes:

```text
--mode mini-pair
--mode small
```

Also test subset mode, shard mode, mutually exclusive flags, missing factor manifest, source/factor digest mismatch, unsupported model pair, strict output, and no model load when no eligible windows exist.

- [ ] **Step 2: Run tests and verify failure**

```bash
python -m pytest tests/bist_eval/test_cli.py -k "adjusted_benchmark" -v
```

- [ ] **Step 3: Implement thin CLI composition**

Required arguments:

```text
--mode
--raw-dir
--source-manifest
--factor-manifest
--universe
--output
--start / --end
--lookback / --horizon
--calendar-coverage
--minimum-ranking-cohort
--model-id / --tokenizer-id
--model-path / --tokenizer-path
--model-revision / --tokenizer-revision
--temperature / --top-p / --sample-count / --seed / --device
--material-factor-tolerance
--symbols
--shard-index / --shard-count
--strict
```

The common calendar must be built from all valid raw symbol timestamps before selecting one shard. Manifests must record source, factor, universe, cohort, common protocol, arm config, model/tokenizer, target formula, target data, and selected-symbol fingerprints.

- [ ] **Step 4: Run tests and verify pass**

```bash
python -m pytest tests/bist_eval/test_cli.py -k "adjusted_benchmark" -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate_bist100_adjusted_benchmark.py tests/bist_eval/test_cli.py
git commit -m "feat(eval): add adjusted benchmark shard CLI"
```

---

## Chunk 4: Arm Artifacts, Paired Comparisons, and Bootstrap

### Task 10: Add benchmark output schemas and arm-set manifest validation

**Files:**
- Modify: `bist_eval/reporting.py`
- Modify: `bist_eval/sharding.py`
- Modify: `tests/bist_eval/test_reporting.py`
- Modify: `tests/bist_eval/test_sharding.py`

- [ ] **Step 1: Write failing schema and manifest tests**

Require benchmark prediction columns including:

```text
experiment_arm,symbol,candidate_month,forecast_origin,target_timestamp,
horizon_step,method,predicted_close,actual_close,history_last_close,
context_view,scoring_target_view,exposure_bucket,
context_factor_changed,target_factor_changed,
context_max_abs_log_step,target_max_abs_log_from_origin
```

Duplicate-key validation includes `experiment_arm`. Validate ten mini-pair shards plus ten small shards, no overlap within a mode, identical source/factor/cohort/common-target protocol across modes, expected arm sets, and exact model/tokenizer revisions per mode.

- [ ] **Step 2: Run tests and verify failure**

```bash
python -m pytest tests/bist_eval/test_reporting.py tests/bist_eval/test_sharding.py -v
```

- [ ] **Step 3: Implement benchmark writers and validators**

Add:

```python
def write_benchmark_shard_output(...): ...
def validate_benchmark_shard_manifests(mini_manifests, small_manifests, expected_count): ...
```

Shard files:

```text
predictions.csv
window_metrics.csv
skipped_windows.csv
shard_manifest.json
COMPLETED
```

Completion marker remains last. Old zero-shot writers/validators stay unchanged.

- [ ] **Step 4: Run tests and verify pass**

```bash
python -m pytest tests/bist_eval/test_reporting.py tests/bist_eval/test_sharding.py -v
```

- [ ] **Step 5: Commit**

```bash
git add bist_eval/reporting.py bist_eval/sharding.py tests/bist_eval/test_reporting.py tests/bist_eval/test_sharding.py
git commit -m "feat(eval): add benchmark artifact contracts"
```

### Task 11: Implement exact paired comparisons

**Files:**
- Create: `bist_eval/comparison.py`
- Create: `tests/bist_eval/test_comparison.py`

- [ ] **Step 1: Write failing paired-join tests**

Required comparisons:

```text
adjusted-mini vs raw-mini
adjusted-small vs adjusted-mini
adjusted-mini vs adjusted-baselines:last_close
adjusted-mini vs adjusted-baselines:momentum_20
adjusted-mini vs adjusted-baselines:linear_trend_20
adjusted-small vs each adjusted baseline
```

Join key:

```text
symbol,candidate_month,forecast_origin
```

Before joining, validate identical final target timestamp, history last close, actual final close, scoring target view, exposure bucket, and common target fingerprint. Unmatched windows are excluded and counted. Difference is challenger error minus reference error; negative favors challenger. Ties remain ties.

- [ ] **Step 2: Run tests and verify failure**

```bash
python -m pytest tests/bist_eval/test_comparison.py -k "pair" -v
```

- [ ] **Step 3: Implement paired records**

```python
@dataclass(frozen=True, slots=True)
class ComparisonSpec:
    comparison_id: str
    challenger_arm: str
    challenger_method: str
    reference_arm: str
    reference_method: str


def build_paired_comparison(window_metrics, spec: ComparisonSpec) -> pd.DataFrame: ...
```

Output includes challenger/reference values, signed difference, winner, target/protocol fingerprints, and exposure bucket.

- [ ] **Step 4: Run tests and verify pass**

```bash
python -m pytest tests/bist_eval/test_comparison.py -k "pair" -v
```

- [ ] **Step 5: Commit**

```bash
git add bist_eval/comparison.py tests/bist_eval/test_comparison.py
git commit -m "feat(eval): add paired arm comparisons"
```

### Task 12: Add deterministic symbol- and origin-clustered bootstrap

**Files:**
- Modify: `bist_eval/comparison.py`
- Modify: `tests/bist_eval/test_comparison.py`

- [ ] **Step 1: Write failing deterministic bootstrap tests**

Use small synthetic paired data and fixed seeds. Test:

- repeated calls return identical intervals,
- symbol resampling includes all windows for each sampled symbol with multiplicity,
- origin resampling includes all symbols for each sampled origin with multiplicity,
- percentile bounds use `(1-confidence)/2` and `1-(1-confidence)/2`,
- one/zero cluster returns unavailable, not a fabricated interval,
- all-negative differences produce `robustly_better`,
- all-positive produce `robustly_worse`,
- disagreement or zero-crossing produces `mixed_or_inconclusive`.

- [ ] **Step 2: Run tests and verify failure**

```bash
python -m pytest tests/bist_eval/test_comparison.py -k "bootstrap" -v
```

- [ ] **Step 3: Implement bootstrap helpers**

```python
def clustered_bootstrap_mean_difference(
    paired: pd.DataFrame,
    *,
    cluster_column: str,
    draws: int,
    confidence: float,
    seed: int,
) -> dict: ...


def summarize_comparison_intervals(...): ...
```

Use `np.random.default_rng(seed)`. Do not silently drop non-finite differences; validate before resampling. Record number of rows and unique clusters.

- [ ] **Step 4: Run tests and verify pass**

```bash
python -m pytest tests/bist_eval/test_comparison.py -v
```

- [ ] **Step 5: Commit**

```bash
git add bist_eval/comparison.py tests/bist_eval/test_comparison.py
git commit -m "feat(eval): add clustered benchmark intervals"
```

### Task 13: Implement the benchmark reducer and final report

**Files:**
- Create: `scripts/reduce_bist100_adjusted_benchmark.py`
- Modify: `bist_eval/reporting.py`
- Modify: `tests/bist_eval/test_cli.py`
- Modify: `tests/bist_eval/test_reporting.py`

- [ ] **Step 1: Write failing reducer tests**

Create synthetic directories for ten mini-pair and ten small shards. Cover success and failures for missing shard, absent completion marker, wrong arm set, source/factor/cohort/target mismatch, model revision mismatch, duplicate prediction key, mismatched actual target, malformed schema, and no paired windows.

- [ ] **Step 2: Run tests and verify failure**

```bash
python -m pytest tests/bist_eval/test_cli.py -k "adjusted_reduce" -v
```

- [ ] **Step 3: Implement reducer**

Arguments:

```text
--mini-shards-dir
--small-shards-dir
--expected-shards
--factor-manifest
--output
--minimum-ranking-cohort
--bootstrap-draws
--bootstrap-confidence
--bootstrap-seed
```

Validate manifests before reading/concatenating predictions. Produce:

```text
adjusted_data_manifest.json
predictions.csv
window_metrics.csv
skipped_windows.csv
symbol_metrics.csv
period_metrics.csv
ranking_metrics.csv
paired_comparisons.csv
bootstrap_intervals.csv
summary.json
run_manifest.json
report.md
COMPLETED
```

`summary.json` must answer all five questions from the spec and include results for all windows plus both exposure buckets. `report.md` must distinguish technical completion from model performance and include survivorship, Yahoo, factor-ratio, raw-volume, amount, and non-investment warnings.

- [ ] **Step 4: Run tests and verify pass**

```bash
python -m pytest tests/bist_eval/test_cli.py -k "adjusted_reduce" -v
python -m pytest tests/bist_eval/test_reporting.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/reduce_bist100_adjusted_benchmark.py bist_eval/reporting.py tests/bist_eval/test_cli.py tests/bist_eval/test_reporting.py
git commit -m "feat(eval): add adjusted benchmark reducer"
```

---

## Chunk 5: CI, Real-Model Workflows, and Documentation

### Task 14: Extend network-free PR CI

**Files:**
- Modify: `.github/workflows/bist-eval-tests.yml`

- [ ] **Step 1: Add trigger paths and compile targets**

Include:

```text
bist_eval/adjustments.py
bist_eval/benchmark.py
bist_eval/comparison.py
scripts/validate_bist_adjustment_factors.py
scripts/evaluate_bist100_adjusted_benchmark.py
scripts/reduce_bist100_adjusted_benchmark.py
tests/bist_eval/**
```

Keep dependencies `numpy pandas pytest` only. No Torch, Yahoo, Hugging Face, or SciPy download.

- [ ] **Step 2: Run equivalent commands locally**

```bash
python -m pytest tests/bist_eval -v
python -m compileall -q bist_eval \
  scripts/validate_bist_adjustment_factors.py \
  scripts/evaluate_bist100_adjusted_benchmark.py \
  scripts/reduce_bist100_adjusted_benchmark.py
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/bist-eval-tests.yml
git commit -m "ci(eval): cover adjusted benchmark tests"
```

### Task 15: Add the manual three-arm smoke workflow

**Files:**
- Create: `.github/workflows/bist-adjusted-benchmark-smoke.yml`

- [ ] **Step 1: Define manual-only inputs**

Defaults:

```text
source_artifact_id=8826613095
source_artifact_sha256=51578854a6fc341f34bff8bea4cf8f57d900006f2a35e2a34e983d6389933f2c
symbols=THYAO,ASELS
start_date=2023-01-01
end_date=2023-02-28
seed=20260802
sample_count=1
```

No schedule or pull-request trigger.

- [ ] **Step 2: Implement one resource-bounded job**

The job must:

1. download and checksum the immutable source artifact,
2. validate factor provenance for THYAO/ASELS,
3. resolve exact mini/2k and small/base assets,
4. run mini-pair subset,
5. run small subset,
6. reduce with minimum ranking cohort 2 and test-sized bootstrap draws such as 1,000,
7. validate exactly two symbols and two forecast origins,
8. upload raw manifests, model manifests, shards, and final result.

Use CPU explicitly for reproducibility of the smoke and a bounded timeout. The smoke is a technical gate only.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/bist-adjusted-benchmark-smoke.yml
git commit -m "ci(eval): add adjusted benchmark smoke workflow"
```

### Task 16: Add the full manual 20-shard benchmark workflow

**Files:**
- Create: `.github/workflows/bist100-adjusted-kronos-benchmark.yml`

- [ ] **Step 1: Define validated manual inputs**

Required/default inputs:

```text
source_artifact_id
source_artifact_sha256
start_date=2023-01-01
end_date=2026-08-02
seed=20260802
sample_count=1
shard_count=10
bootstrap_draws=10000
bootstrap_confidence=0.95
material_factor_tolerance=1e-8
```

Enforce exactly ten shards for the first release.

- [ ] **Step 2: Add immutable `prepare-data` job**

- download source artifact through `gh api`,
- verify supplied SHA-256 before unzip,
- assert manifest `100/100/0`, 100 raw CSVs, last candle `2026-07-31`,
- run factor validation strict mode,
- upload raw data + source/factor manifests once.

Do not call live Yahoo in this workflow.

- [ ] **Step 3: Add model preparation jobs**

`prepare-model-mini` resolves:

```text
NeoQuasar/Kronos-mini
NeoQuasar/Kronos-Tokenizer-2k
```

`prepare-model-small` resolves:

```text
NeoQuasar/Kronos-small
NeoQuasar/Kronos-Tokenizer-base
```

Each uploads exact asset manifest and files once. Validate 40-character revisions and exact IDs.

- [ ] **Step 4: Add `evaluate-mini` ten-shard matrix**

Run `--mode mini-pair`. Use contiguous deterministic shard mapping, `max-parallel: 5`, strict mode, a bounded timeout, and upload artifact even on failure. Require `COMPLETED`.

- [ ] **Step 5: Add `evaluate-small` ten-shard matrix**

Run `--mode small` on the same shard indexes and data. Start after data/small-model preparation. Use an appropriate parallel cap and timeout; require `COMPLETED`.

- [ ] **Step 6: Add reducer job**

Wait for all 20 shards. Download artifacts into separate mini/small roots, run reducer with 10,000 draws, validate required files and 100-symbol manifest, write key performance/interval conclusions to `$GITHUB_STEP_SUMMARY`, and upload a 30-day result artifact.

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/bist100-adjusted-kronos-benchmark.yml
git commit -m "ci(eval): add full adjusted Kronos benchmark"
```

### Task 17: Document operations and generated-path boundaries

**Files:**
- Create: `docs/bist-adjusted-kronos-benchmark.md`
- Modify: `.gitignore`

- [ ] **Step 1: Add ignore rules**

```text
results/bist100-adjusted-benchmark/
.artifacts/bist100-adjusted-benchmark/
.models/kronos-mini/
.models/kronos-small/
data/bist/adjustment-factors/
```

Do not ignore specs, plans, universe, tests, or source code.

- [ ] **Step 2: Write operator documentation**

Document:

- why static adjusted CSVs are forbidden,
- origin-relative factor formula and common target,
- three arms and exact model/tokenizer pairs,
- local network-free tests,
- factor validation command,
- local fake-adapter smoke,
- manual real smoke workflow,
- full workflow inputs,
- output files and bootstrap decisions,
- exposure buckets,
- previous raw zero-shot result as context only,
- research limitations and no trading capability.

- [ ] **Step 3: Commit**

```bash
git add .gitignore docs/bist-adjusted-kronos-benchmark.md
git commit -m "docs(eval): document adjusted Kronos benchmark"
```

---

## Chunk 6: Verification, PR, and Full Run Gates

### Task 18: Run complete local verification

**Files:**
- No planned source changes unless a test exposes a targeted defect.

- [ ] **Step 1: Run adjusted and legacy evaluation tests**

```bash
python -m pytest tests/bist_eval -v
```

Expected: all old and new tests pass.

- [ ] **Step 2: Run ingestion regression tests**

```bash
python -m pytest tests/bist_data -v
```

Expected: existing 20 ingestion tests remain green.

- [ ] **Step 3: Compile all changed modules/scripts**

```bash
python -m compileall -q bist_eval \
  scripts/validate_bist_adjustment_factors.py \
  scripts/evaluate_bist100_adjusted_benchmark.py \
  scripts/reduce_bist100_adjusted_benchmark.py
```

- [ ] **Step 4: Run synthetic 20-shard end-to-end benchmark**

Use fake adapters and generated raw frames to create ten mini-pair shards and ten small shards. Reduce them with a small deterministic bootstrap draw count. Confirm final `COMPLETED`, no duplicate keys, matching targets, and all expected comparisons.

- [ ] **Step 5: Check branch scope**

```bash
git diff --stat master...HEAD
git diff --check master...HEAD
```

Confirm no generated data, result CSVs, model weights, tokens, or temporary workflows are committed.

### Task 19: Run the real two-symbol/two-origin smoke

**Files:**
- No source changes unless the smoke exposes a reproducible defect.

- [ ] **Step 1: Trigger `BIST Adjusted Benchmark Smoke` manually**

Use the exact branch commit, immutable artifact ID/digest, and approved defaults.

- [ ] **Step 2: Verify preparation and model assets**

Confirm source checksum, factor manifest, mini revision, mini tokenizer revision, small revision, and base tokenizer revision.

- [ ] **Step 3: Verify three-arm smoke outputs**

Require:

- symbols exactly THYAO and ASELS,
- two common origins,
- raw-mini, adjusted-mini, adjusted-small, and adjusted baselines,
- identical actual common targets across arms,
- paired comparison rows,
- both cluster interval types,
- no target leakage contract violation,
- final `COMPLETED`.

- [ ] **Step 4: Record smoke evidence**

Add workflow run ID, artifact ID/digest, exact revisions, row/window counts, and any limitations to the branch/PR notes. Do not treat smoke performance as evidence.

### Task 20: Open and validate the Draft PR

**Files:**
- No new implementation files.

- [ ] **Step 1: Open a Draft PR only after smoke success**

Title:

```text
feat(eval): add adjusted Kronos benchmark
```

Body must include architecture, leakage boundary, model pairs, test results, smoke evidence, workflow cost/concurrency, Yahoo/survivorship limitations, and no trading/fine-tuning statement.

- [ ] **Step 2: Wait for network-free PR CI and fix only root causes**

Use `systematic-debugging` before any change caused by unexpected test/CI behavior.

- [ ] **Step 3: Review diff and discussion**

Confirm all CI green, no unresolved threads, no secret exposure, no generated assets, and branch is mergeable. Do not mark ready or merge without explicit user approval.

### Task 21: Run the full benchmark after code review approval

**Files:**
- No repository change unless a reproducible implementation defect is found.

- [ ] **Step 1: Trigger the full manual workflow**

Use exact PR head SHA and immutable source artifact digest.

- [ ] **Step 2: Verify all preparation gates**

Require 100 raw files, 100 valid factor series, zero strict failures, and exact model/tokenizer revisions.

- [ ] **Step 3: Verify all 20 shard artifacts**

Ten mini-pair plus ten small shards must be complete, non-overlapping within mode, identically partitioned across modes, and fingerprint-compatible.

- [ ] **Step 4: Verify reduced result**

Reconcile predictions, window metrics, skips, paired comparisons, cluster counts, intervals, exposure buckets, and summary. Check that conclusions use `robustly_better`, `robustly_worse`, or `mixed_or_inconclusive` exactly as specified.

- [ ] **Step 5: Record result in Draft PR**

Post run/artifact IDs and digests, revisions, eligible/skipped counts, primary log-return errors, direction/ranking metrics, baseline comparisons, clustered intervals, and limitations. No profitability claim.

- [ ] **Step 6: Request a separate merge decision**

Do not merge, schedule recurring runs, fine-tune, construct a portfolio, or connect a broker without a new explicit user decision.

---

## Final Acceptance Checklist

- [ ] Existing raw zero-shot evaluator remains backward compatible.
- [ ] Raw files stay immutable; no static adjusted model-input CSV is produced.
- [ ] Every adjusted context is rebased to its own origin factor.
- [ ] Model and baseline inputs contain no target values or target factors.
- [ ] Target transformation occurs only after prediction.
- [ ] Raw-mini, adjusted-mini, adjusted-small, and baselines use one common rebased target.
- [ ] Mini and small use only their approved tokenizers and exact revisions.
- [ ] All model contexts contain exactly 400 rows and targets exactly five dates.
- [ ] Primary metric is final five-day log-return absolute error.
- [ ] Paired joins reject incompatible or mismatched targets.
- [ ] Symbol- and origin-clustered intervals are deterministic and reported separately.
- [ ] Robust decision requires both intervals to agree completely above/below zero.
- [ ] Exposure diagnostics separate material corporate-action windows.
- [ ] Twenty shards are complete and reducer validated.
- [ ] PR tests require no network, Torch, Yahoo, Hugging Face, or SciPy.
- [ ] Real smoke succeeds before Draft PR completion claims.
- [ ] Full benchmark remains manual and resource bounded.
- [ ] Reports prominently state Yahoo, survivorship, factor-ratio, raw-volume, estimated-amount, and research-only limitations.
- [ ] No fine-tuning, portfolio construction, order placement, broker connection, or investment recommendation is introduced.
