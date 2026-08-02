# BIST 100 Zero-Shot Evaluation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, leakage-safe, zero-shot evaluation that compares Kronos-mini with three deterministic baselines across the 2026 Q3 BIST 100 constituent snapshot over common monthly forecast cohorts from 2023 through 2026.

**Architecture:** Add an isolated `bist_eval` package for calendar construction, window generation, baselines, model adaptation, metrics, sharding, and reporting. A lightweight PR workflow runs network-free tests, while a manually dispatched workflow regenerates Yahoo data, resolves exact public Hugging Face model revisions once, evaluates ten deterministic symbol shards, and reduces them into one validated report.

**Tech Stack:** Python 3.11, pandas, NumPy, PyTorch, Hugging Face Hub, Kronos-mini, pytest, GitHub Actions, CSV/JSON/Markdown artifacts.

---

## Source of Truth

- Approved design: `docs/superpowers/specs/2026-08-02-bist100-zero-shot-evaluation-design.md`
- Existing BIST data package: `bist_data/`
- Existing downloader: `scripts/download_bist_yahoo.py`
- Existing universe: `data/universes/xu100_2026_q3.csv`
- Existing Kronos API: `model.Kronos`, `model.KronosTokenizer`, `model.KronosPredictor`

The implementation must not change Kronos model internals, add order placement, or describe the output as a historical BIST 100 index backtest.

## Locked File Structure

### Evaluation package

- Create `bist_eval/__init__.py` — public evaluation types and helpers.
- Create `bist_eval/config.py` — immutable run configuration, validation, canonical JSON, and fingerprint.
- Create `bist_eval/data.py` — schema-valid Kronos CSV loading and symbol-file discovery.
- Create `bist_eval/calendar.py` — timestamp-only canonical calendar and common monthly cohorts.
- Create `bist_eval/windows.py` — leakage-free 400-row context / five-row target windows and structured skips.
- Create `bist_eval/baselines.py` — last-close, 20-row momentum, and 20-row linear-trend forecasts.
- Create `bist_eval/model_adapter.py` — lazy model loading, deterministic seeding, batched prediction, and output validation.
- Create `bist_eval/metrics.py` — per-window, per-symbol, per-period, ranking, and overall metrics.
- Create `bist_eval/sharding.py` — deterministic symbol partitioning and shard manifest validation.
- Create `bist_eval/reporting.py` — stable schemas, atomic output, JSON serialization, completion marker, and Markdown report.

### Commands

- Create `scripts/evaluate_bist100_zero_shot.py` — run one symbol shard or a local subset.
- Create `scripts/reduce_bist100_zero_shot.py` — validate and combine all shard artifacts.
- Create `scripts/resolve_kronos_assets.py` — resolve exact Hugging Face revisions and download model/tokenizer into local directories.

### Workflows and documentation

- Create `.github/workflows/bist-eval-tests.yml` — network-free PR tests.
- Create `.github/workflows/bist100-zero-shot-evaluation.yml` — manual preparation, ten shards, reduction, and artifacts.
- Create `docs/bist-zero-shot-evaluation.md` — local and Actions usage, interpretation, and limitations.
- Modify `.gitignore` — ignore generated evaluation results and local model snapshots.

### Tests

- Create `tests/bist_eval/conftest.py`.
- Create `tests/bist_eval/test_config.py`.
- Create `tests/bist_eval/test_calendar.py`.
- Create `tests/bist_eval/test_windows.py`.
- Create `tests/bist_eval/test_baselines.py`.
- Create `tests/bist_eval/test_model_adapter.py`.
- Create `tests/bist_eval/test_metrics.py`.
- Create `tests/bist_eval/test_sharding.py`.
- Create `tests/bist_eval/test_reporting.py`.
- Create `tests/bist_eval/test_cli.py`.

Do not add SciPy for correlation metrics; pandas rank/correlation operations are sufficient for this milestone.

---

## Chunk 1: Deterministic Experiment Contracts

### Task 1: Add immutable run configuration and fingerprint

**Files:**
- Create: `bist_eval/__init__.py`
- Create: `bist_eval/config.py`
- Test: `tests/bist_eval/test_config.py`

- [ ] **Step 1: Write failing configuration tests**

Cover:

```python
from bist_eval.config import EvaluationConfig


def test_config_has_stable_fingerprint():
    left = EvaluationConfig(seed=20260802)
    right = EvaluationConfig(seed=20260802)
    assert left.fingerprint == right.fingerprint


def test_config_rejects_invalid_horizon():
    with pytest.raises(ValueError, match="horizon"):
        EvaluationConfig(horizon=0)


def test_config_defaults_match_approved_spec():
    config = EvaluationConfig()
    assert config.lookback == 400
    assert config.horizon == 5
    assert config.model_id == "NeoQuasar/Kronos-mini"
    assert config.tokenizer_id == "NeoQuasar/Kronos-Tokenizer-2k"
    assert config.calendar_coverage == 0.80
    assert config.minimum_ranking_cohort == 20
```

- [ ] **Step 2: Run the targeted tests and confirm failure**

Run:

```bash
python -m pytest tests/bist_eval/test_config.py -v
```

Expected: collection fails because `bist_eval.config` does not exist.

- [ ] **Step 3: Implement `EvaluationConfig`**

Use a frozen dataclass. Include at least:

```python
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
```

Validation must reject invalid ISO dates, end before start, nonpositive lookback/horizon/sample count/shard count, coverage outside `(0, 1]`, top-p outside `(0, 1]`, and nonpositive minimum cohort.

Create `to_canonical_dict()` and calculate `fingerprint` as SHA-256 of sorted compact JSON. Resolved model/tokenizer revisions must participate in the fingerprint used by shard outputs.

- [ ] **Step 4: Run tests and confirm pass**

Run:

```bash
python -m pytest tests/bist_eval/test_config.py -v
```

Expected: all configuration tests pass.

- [ ] **Step 5: Commit**

```bash
git add bist_eval/__init__.py bist_eval/config.py tests/bist_eval/test_config.py
git commit -m "feat(eval): add zero-shot evaluation config"
```

### Task 2: Load and validate Kronos-ready symbol files

**Files:**
- Create: `bist_eval/data.py`
- Create: `tests/bist_eval/conftest.py`
- Test: `tests/bist_eval/test_windows.py`

- [ ] **Step 1: Add reusable synthetic OHLCVA fixtures**

Create fixture helpers that generate deterministic daily rows with columns:

```text
timestamps,open,high,low,close,volume,amount
```

Fixtures must support missing dates, duplicate timestamps, insufficient history, and non-finite values.

- [ ] **Step 2: Write failing loader tests**

Test that `load_symbol_frame(path)`:

- parses and sorts timestamps,
- delegates OHLCV invariants to `bist_data.quality.validate_candles`,
- requires `amount`,
- rejects duplicate dates and non-finite amount,
- returns a frame with stable column order.

- [ ] **Step 3: Run tests and confirm failure**

```bash
python -m pytest tests/bist_eval/test_windows.py -k "load_symbol" -v
```

Expected: import or symbol-not-found failure.

- [ ] **Step 4: Implement discovery and loading**

Public functions:

```python
def discover_symbol_files(data_dir: Path, symbols: Sequence[str]) -> dict[str, Path]: ...
def load_symbol_frame(path: Path) -> pd.DataFrame: ...
def load_timestamp_coverage(files: Mapping[str, Path]) -> dict[str, pd.DatetimeIndex]: ...
```

`load_timestamp_coverage` may read only timestamp columns for calendar construction. It must not inspect prices or returns.

- [ ] **Step 5: Run loader tests and confirm pass**

```bash
python -m pytest tests/bist_eval/test_windows.py -k "load_symbol" -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add bist_eval/data.py tests/bist_eval/conftest.py tests/bist_eval/test_windows.py
git commit -m "feat(eval): add validated symbol data loading"
```

---

## Chunk 2: Common Calendar and Leakage-Free Windows

### Task 3: Build timestamp-only canonical monthly cohorts

**Files:**
- Create: `bist_eval/calendar.py`
- Test: `tests/bist_eval/test_calendar.py`

- [ ] **Step 1: Write failing calendar tests**

Cover:

- coverage count uses `ceil(valid_file_count * coverage_threshold)`,
- price values do not affect calendar output,
- first canonical date in each calendar month is the origin,
- the next five canonical dates are the shared targets,
- incomplete final cohorts are omitted with a structured reason,
- output is deterministic regardless of mapping insertion order.

Expected structure:

```python
@dataclass(frozen=True, slots=True)
class MonthlyCohort:
    candidate_month: str
    forecast_origin: pd.Timestamp
    target_timestamps: tuple[pd.Timestamp, ...]
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
python -m pytest tests/bist_eval/test_calendar.py -v
```

Expected: module missing.

- [ ] **Step 3: Implement canonical calendar construction**

Public API:

```python
def build_canonical_calendar(
    timestamp_coverage: Mapping[str, pd.DatetimeIndex],
    *,
    coverage_threshold: float,
    start_date: str,
    end_date: str,
) -> pd.DatetimeIndex: ...


def build_monthly_cohorts(
    calendar: pd.DatetimeIndex,
    *,
    horizon: int,
) -> tuple[list[MonthlyCohort], list[CalendarSkip]]: ...
```

The implementation must count each symbol at most once per date and use timestamps only.

- [ ] **Step 4: Run tests and confirm pass**

```bash
python -m pytest tests/bist_eval/test_calendar.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bist_eval/calendar.py tests/bist_eval/test_calendar.py
git commit -m "feat(eval): add common monthly market calendar"
```

### Task 4: Generate exact, leakage-free forecast windows

**Files:**
- Create: `bist_eval/windows.py`
- Modify: `tests/bist_eval/test_windows.py`

- [ ] **Step 1: Write failing window tests**

Required assertions:

```python
assert len(window.context) == 400
assert len(window.target) == 5
assert window.context["timestamps"].max() == window.forecast_origin
assert window.context["timestamps"].max() < window.target["timestamps"].min()
assert tuple(window.target["timestamps"]) == cohort.target_timestamps
```

Also cover skip codes:

- `missing_origin_date`
- `missing_target_date`
- `insufficient_history`
- `invalid_input_values`
- `invalid_target_values`

A recent IPO must be skipped, never padded.

- [ ] **Step 2: Run tests and confirm failure**

```bash
python -m pytest tests/bist_eval/test_windows.py -k "window or skip" -v
```

Expected: missing implementation.

- [ ] **Step 3: Implement window and skip records**

Use immutable records:

```python
@dataclass(frozen=True, slots=True)
class ForecastWindow:
    symbol: str
    candidate_month: str
    forecast_origin: pd.Timestamp
    target_timestamps: tuple[pd.Timestamp, ...]
    context: pd.DataFrame
    target: pd.DataFrame

@dataclass(frozen=True, slots=True)
class SkipRecord:
    symbol: str
    candidate_month: str
    reason_code: str
    reason_detail: str
    available_history_rows: int
    available_target_rows: int
```

`build_symbol_windows` must use index lookups and positional context slicing. Never forward-fill dates or values.

- [ ] **Step 4: Run tests and confirm pass**

```bash
python -m pytest tests/bist_eval/test_windows.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bist_eval/windows.py tests/bist_eval/test_windows.py
git commit -m "feat(eval): add leakage-safe forecast windows"
```

---

## Chunk 3: Baselines, Model Adapter, and Metrics

### Task 5: Implement deterministic baseline forecasts

**Files:**
- Create: `bist_eval/baselines.py`
- Test: `tests/bist_eval/test_baselines.py`

- [ ] **Step 1: Write failing baseline tests**

Lock these formulas:

```python
# Last close
prediction = np.repeat(history_close[-1], horizon)

# 20-row momentum: 19 observed transitions
rate = (history_close[-1] / history_close[-20]) ** (1.0 / 19.0) - 1.0
prediction[h - 1] = history_close[-1] * (1.0 + rate) ** h

# Linear trend
slope, intercept = np.polyfit(np.arange(20), history_close[-20:], 1)
prediction = slope * np.arange(20, 20 + horizon) + intercept
```

Test constant prices, positive trend, negative trend, nonpositive momentum endpoints, and insufficient baseline history.

- [ ] **Step 2: Run tests and confirm failure**

```bash
python -m pytest tests/bist_eval/test_baselines.py -v
```

- [ ] **Step 3: Implement baseline registry**

Public API:

```python
BASELINE_METHODS = ("last_close", "momentum_20", "linear_trend_20")

def forecast_baselines(context: pd.DataFrame, horizon: int) -> dict[str, np.ndarray]: ...
```

Momentum must fail closed for zero or negative endpoint prices rather than create complex or infinite values.

- [ ] **Step 4: Run tests and confirm pass**

```bash
python -m pytest tests/bist_eval/test_baselines.py -v
```

- [ ] **Step 5: Commit**

```bash
git add bist_eval/baselines.py tests/bist_eval/test_baselines.py
git commit -m "feat(eval): add deterministic forecast baselines"
```

### Task 6: Add a lazy, testable Kronos model adapter

**Files:**
- Create: `bist_eval/model_adapter.py`
- Test: `tests/bist_eval/test_model_adapter.py`

- [ ] **Step 1: Write fake-predictor tests first**

Verify that the adapter:

- imports without loading Torch weights,
- sends exactly `open,high,low,close,volume,amount`,
- sends 400 context timestamps and five target timestamps,
- aligns returned rows to the five common target dates,
- passes `T`, `top_p`, `sample_count`, and `verbose=False`,
- rejects wrong row counts, missing close, non-finite output, and timestamp mismatch,
- batches windows sharing one forecast cohort,
- derives a stable cohort seed from base seed and forecast-origin ISO date.

- [ ] **Step 2: Run tests and confirm failure**

```bash
python -m pytest tests/bist_eval/test_model_adapter.py -v
```

- [ ] **Step 3: Implement model loading and prediction**

Use lazy imports inside `load()`:

```python
from model import Kronos, KronosPredictor, KronosTokenizer

tokenizer = KronosTokenizer.from_pretrained(tokenizer_path_or_id, revision=revision)
model = Kronos.from_pretrained(model_path_or_id, revision=revision)
tokenizer.eval()
model.eval()
predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
```

Before each cohort batch, seed `random`, NumPy, and Torch. Set CUDA seeds when available. Do not promise bit-for-bit equivalence across device types; record device and seed.

Prefer `KronosPredictor.predict_batch` for all eligible windows in the same shard and monthly cohort. Fall back to individual fake predictor calls only in tests, not silently in production.

- [ ] **Step 4: Run tests and confirm pass**

```bash
python -m pytest tests/bist_eval/test_model_adapter.py -v
```

- [ ] **Step 5: Commit**

```bash
git add bist_eval/model_adapter.py tests/bist_eval/test_model_adapter.py
git commit -m "feat(eval): add Kronos zero-shot model adapter"
```

### Task 7: Implement guarded window and aggregate metrics

**Files:**
- Create: `bist_eval/metrics.py`
- Test: `tests/bist_eval/test_metrics.py`

- [ ] **Step 1: Write failing metric tests**

Cover:

- five-step MAE and RMSE,
- final-horizon absolute percentage error with zero-denominator result `NaN`,
- predicted and realized return from history last close,
- direction correctness, including explicit behavior for zero returns,
- Pearson correlation only when at least two finite, nonconstant pairs exist,
- Spearman correlation using average ranks and Pearson correlation of ranks,
- top-five overlap,
- predicted-top-five mean realized return,
- minimum cohort threshold,
- Kronos win rate versus each baseline by final-horizon absolute error.

- [ ] **Step 2: Run tests and confirm failure**

```bash
python -m pytest tests/bist_eval/test_metrics.py -v
```

- [ ] **Step 3: Implement metric functions**

Use explicit functions with DataFrame outputs:

```python
def compute_window_metrics(predictions: pd.DataFrame) -> pd.DataFrame: ...
def aggregate_symbol_metrics(window_metrics: pd.DataFrame) -> pd.DataFrame: ...
def aggregate_period_metrics(window_metrics: pd.DataFrame) -> pd.DataFrame: ...
def compute_ranking_metrics(window_metrics: pd.DataFrame, minimum_cohort: int) -> pd.DataFrame: ...
def build_overall_summary(...) -> dict[str, Any]: ...
```

Never coerce undefined correlation or percentage errors to zero. Preserve them as missing values and report defined-count denominators.

- [ ] **Step 4: Run tests and confirm pass**

```bash
python -m pytest tests/bist_eval/test_metrics.py -v
```

- [ ] **Step 5: Commit**

```bash
git add bist_eval/metrics.py tests/bist_eval/test_metrics.py
git commit -m "feat(eval): add zero-shot evaluation metrics"
```

---

## Chunk 4: Shards, Outputs, and Commands

### Task 8: Add deterministic sharding and compatibility validation

**Files:**
- Create: `bist_eval/sharding.py`
- Test: `tests/bist_eval/test_sharding.py`

- [ ] **Step 1: Write failing sharding tests**

Test that ordered 100-symbol input with ten shards yields stable contiguous groups of ten, no omissions, no duplicates, and valid shard indexes `0..9`.

Test reducer validation rejects:

- missing shard index,
- duplicate shard index,
- different config fingerprint,
- different source-data fingerprint,
- different resolved model/tokenizer revision,
- overlapping symbols.

- [ ] **Step 2: Run tests and confirm failure**

```bash
python -m pytest tests/bist_eval/test_sharding.py -v
```

- [ ] **Step 3: Implement shard records**

Public API:

```python
def partition_symbols(symbols: Sequence[str], shard_count: int) -> list[tuple[str, ...]]: ...
def select_shard(symbols: Sequence[str], shard_count: int, shard_index: int) -> tuple[str, ...]: ...
def validate_shard_manifests(manifests: Sequence[dict[str, Any]], expected_count: int) -> None: ...
```

Partition from the universe-file order. Do not use Python hashes.

- [ ] **Step 4: Run tests and confirm pass**

```bash
python -m pytest tests/bist_eval/test_sharding.py -v
```

- [ ] **Step 5: Commit**

```bash
git add bist_eval/sharding.py tests/bist_eval/test_sharding.py
git commit -m "feat(eval): add deterministic evaluation shards"
```

### Task 9: Add atomic reporting and completion contracts

**Files:**
- Create: `bist_eval/reporting.py`
- Test: `tests/bist_eval/test_reporting.py`

- [ ] **Step 1: Write failing reporting tests**

Cover:

- fixed prediction and metric column order,
- ISO timestamps,
- NumPy/Pandas scalar JSON serialization,
- atomic CSV/JSON writes,
- report disclaimer text,
- completion marker written last,
- no completion marker after schema failure,
- duplicate prediction key rejection for `(symbol, forecast_origin, target_timestamp, method)`.

- [ ] **Step 2: Run tests and confirm failure**

```bash
python -m pytest tests/bist_eval/test_reporting.py -v
```

- [ ] **Step 3: Implement report writers**

Shard output:

```text
predictions.csv
window_metrics.csv
skipped_windows.csv
shard_manifest.json
COMPLETED
```

Reduced output:

```text
predictions.csv
window_metrics.csv
skipped_windows.csv
symbol_metrics.csv
period_metrics.csv
ranking_metrics.csv
summary.json
run_manifest.json
report.md
COMPLETED
```

The Markdown title must contain:

> Zero-shot historical evaluation of the 2026 Q3 BIST 100 constituent snapshot over 2023-2026

Include the survivorship-bias, Yahoo research-data, uncosted diagnostic, and non-investment-advice warnings.

- [ ] **Step 4: Run tests and confirm pass**

```bash
python -m pytest tests/bist_eval/test_reporting.py -v
```

- [ ] **Step 5: Commit**

```bash
git add bist_eval/reporting.py tests/bist_eval/test_reporting.py
git commit -m "feat(eval): add evaluation artifact reporting"
```

### Task 10: Implement the shard evaluation CLI

**Files:**
- Create: `scripts/evaluate_bist100_zero_shot.py`
- Test: `tests/bist_eval/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Using synthetic CSVs and a fake model adapter, test:

- symbol subset mode,
- shard index/count mode,
- mutually exclusive subset and shard flags,
- common calendar built from all supplied symbol files, not only shard files,
- strict missing-file behavior,
- valid shard artifact output,
- stable config and source fingerprints,
- no model load when no eligible windows exist.

- [ ] **Step 2: Run tests and confirm failure**

```bash
python -m pytest tests/bist_eval/test_cli.py -k "evaluate" -v
```

- [ ] **Step 3: Implement CLI composition**

Required arguments include:

```text
--data-dir
--universe
--output
--start
--end
--lookback
--horizon
--calendar-coverage
--minimum-ranking-cohort
--model-id / --model-path
--tokenizer-id / --tokenizer-path
--model-revision
--tokenizer-revision
--temperature
--top-p
--sample-count
--seed
--device
--symbols
--shard-index
--shard-count
--strict
```

Flow:

1. Load ordered universe.
2. Discover all source files and calculate source-data fingerprint from manifest plus file metadata/checksums.
3. Build the common timestamp calendar from all schema-valid files.
4. Select this shard's symbols.
5. Build windows/skips.
6. Group windows by common monthly cohort.
7. Run baselines and batched Kronos prediction.
8. Write predictions, metrics, skips, shard manifest, then `COMPLETED`.

- [ ] **Step 4: Run CLI tests and confirm pass**

```bash
python -m pytest tests/bist_eval/test_cli.py -k "evaluate" -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate_bist100_zero_shot.py tests/bist_eval/test_cli.py
git commit -m "feat(eval): add BIST 100 shard evaluator"
```

### Task 11: Implement the reducer CLI

**Files:**
- Create: `scripts/reduce_bist100_zero_shot.py`
- Modify: `tests/bist_eval/test_cli.py`

- [ ] **Step 1: Write failing reducer tests**

Create ten small fake shard directories. Test successful reduction and failure on missing shard, mismatched fingerprint, duplicate prediction key, overlapping symbols, absent completion marker, and malformed schema.

- [ ] **Step 2: Run tests and confirm failure**

```bash
python -m pytest tests/bist_eval/test_cli.py -k "reduce" -v
```

- [ ] **Step 3: Implement reducer**

Required arguments:

```text
--shards-dir
--expected-shards
--output
--minimum-ranking-cohort
```

The reducer validates all manifests before concatenating any data. It then calculates symbol, period, ranking, and overall metrics and writes final artifacts atomically.

- [ ] **Step 4: Run tests and confirm pass**

```bash
python -m pytest tests/bist_eval/test_cli.py -k "reduce" -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/reduce_bist100_zero_shot.py tests/bist_eval/test_cli.py
git commit -m "feat(eval): add BIST 100 evaluation reducer"
```

---

## Chunk 5: Reproducible Assets and CI Workflows

### Task 12: Resolve exact public Hugging Face assets once per run

**Files:**
- Create: `scripts/resolve_kronos_assets.py`
- Modify: `tests/bist_eval/test_cli.py`

- [ ] **Step 1: Write network-free tests with a fake Hub client**

Test that the command:

- resolves model and tokenizer commit SHAs,
- downloads each to a distinct local directory,
- writes `asset_manifest.json`,
- refuses mismatched requested/resolved identifiers,
- never writes an access token,
- can be consumed through local model/tokenizer paths by the evaluator.

- [ ] **Step 2: Run tests and confirm failure**

```bash
python -m pytest tests/bist_eval/test_cli.py -k "assets" -v
```

- [ ] **Step 3: Implement asset resolver**

Use `huggingface_hub.HfApi.model_info(...).sha` and `snapshot_download(..., revision=resolved_sha, local_dir=...)`. Record identifiers, exact revisions, generated time, and library version.

- [ ] **Step 4: Run tests and confirm pass**

```bash
python -m pytest tests/bist_eval/test_cli.py -k "assets" -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/resolve_kronos_assets.py tests/bist_eval/test_cli.py
git commit -m "feat(eval): add reproducible Kronos asset resolver"
```

### Task 13: Add lightweight network-free PR workflow

**Files:**
- Create: `.github/workflows/bist-eval-tests.yml`

- [ ] **Step 1: Add workflow with narrow triggers**

Trigger on pull requests affecting:

```text
bist_eval/**
scripts/evaluate_bist100_zero_shot.py
scripts/reduce_bist100_zero_shot.py
scripts/resolve_kronos_assets.py
tests/bist_eval/**
.github/workflows/bist-eval-tests.yml
```

Use Python 3.11 and install only:

```bash
python -m pip install numpy pandas pytest
```

Model-adapter imports must remain lazy so these tests do not require Torch or model downloads.

- [ ] **Step 2: Run the same command locally**

```bash
python -m pytest tests/bist_eval -v
```

Expected: all network-free tests pass.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/bist-eval-tests.yml
git commit -m "ci(eval): add network-free evaluation tests"
```

### Task 14: Add manually dispatched full sharded evaluation workflow

**Files:**
- Create: `.github/workflows/bist100-zero-shot-evaluation.yml`

- [ ] **Step 1: Define manual inputs**

At minimum:

```text
start_date (default 2023-01-01)
end_date (default 2026-08-02)
model_id (default NeoQuasar/Kronos-mini)
tokenizer_id (default NeoQuasar/Kronos-Tokenizer-2k)
seed (default 20260802)
sample_count (default 1)
shard_count (default 10; enforce exactly 10 in first release)
```

Do not add a schedule in this milestone.

- [ ] **Step 2: Add `prepare-data` job**

- checkout,
- install `pandas==2.2.2 yfinance==1.5.1`,
- run existing `scripts/download_bist_yahoo.py` for all 100 symbols with strict mode,
- upload the full data directory and manifest,
- retain artifact for 14 days.

- [ ] **Step 3: Add `prepare-model` job**

- install `huggingface_hub==0.33.1`,
- run `scripts/resolve_kronos_assets.py`,
- upload model/tokenizer directories and asset manifest once,
- retain artifact for 14 days.

- [ ] **Step 4: Add ten-shard matrix evaluation job**

Use:

```yaml
strategy:
  fail-fast: false
  max-parallel: 5
  matrix:
    shard_index: [0,1,2,3,4,5,6,7,8,9]
```

Each job:

- downloads the same data and model assets,
- installs pinned repository runtime dependencies,
- runs one strict shard with local model paths,
- uses a 120-minute timeout,
- uploads shard output even on failure for diagnosis,
- fails if `COMPLETED` is absent.

- [ ] **Step 5: Add reducer job**

The reducer runs only after all shards succeed. It downloads all shard artifacts, validates ten compatible manifests, generates final outputs, and uploads one `bist100-zero-shot-<run_id>` artifact retained for 30 days.

- [ ] **Step 6: Add workflow summary**

Write to `$GITHUB_STEP_SUMMARY`:

- model/tokenizer revisions,
- eligible/skipped windows,
- symbols evaluated,
- Kronos versus baseline summary,
- ranking cohort count,
- prominent research-only disclaimer.

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/bist100-zero-shot-evaluation.yml
git commit -m "ci(eval): add sharded BIST 100 zero-shot workflow"
```

---

## Chunk 6: Documentation and Verification

### Task 15: Document usage and interpretation

**Files:**
- Create: `docs/bist-zero-shot-evaluation.md`
- Modify: `.gitignore`

- [ ] **Step 1: Add generated-path ignores**

Ignore:

```text
results/bist100-zero-shot/
.artifacts/bist100-zero-shot/
.models/kronos/
```

Do not ignore source universe, tests, specs, or plans.

- [ ] **Step 2: Write operator documentation**

Document:

- local synthetic/unit test commands,
- local subset smoke command,
- full workflow dispatch inputs,
- output schemas,
- exact meaning of common monthly cohorts,
- survivorship/selection bias,
- Yahoo and estimated amount limitations,
- why top-five realized return is uncosted diagnostic only,
- why fine-tuning remains out of scope.

- [ ] **Step 3: Commit**

```bash
git add .gitignore docs/bist-zero-shot-evaluation.md
git commit -m "docs(eval): document BIST zero-shot evaluation"
```

### Task 16: Run repository verification before opening the PR

**Files:**
- No source changes unless a verification failure requires a targeted fix.

- [ ] **Step 1: Run the complete network-free evaluation suite**

```bash
python -m pytest tests/bist_eval -v
```

Expected: PASS.

- [ ] **Step 2: Run existing BIST ingestion tests**

```bash
python -m pytest tests/bist_data -v
```

Expected: PASS; the evaluation work must not regress ingestion.

- [ ] **Step 3: Compile new modules and commands**

```bash
python -m compileall bist_eval scripts/evaluate_bist100_zero_shot.py scripts/reduce_bist100_zero_shot.py scripts/resolve_kronos_assets.py
```

Expected: exit code 0.

- [ ] **Step 4: Run a synthetic end-to-end shard/reducer smoke**

Use generated test data and a fake predictor to create ten shard outputs and reduce them. Confirm final `COMPLETED`, stable schemas, and no duplicate keys.

- [ ] **Step 5: Run a real-model two-symbol/two-cohort smoke with explicit approval**

Use local or Actions-resolved public model assets. The smoke must use at least two symbols and two common cohorts and produce schema-valid output. Do not start the full 100-symbol run until this succeeds.

- [ ] **Step 6: Open a Draft PR**

PR description must include:

- scope and architecture,
- test commands/results,
- real-model smoke result,
- compute and artifact-retention behavior,
- survivorship-bias warning,
- statement that no trading/order capability exists,
- explicit gate that full 100-symbol workflow remains manually dispatched.

- [ ] **Step 7: Review PR diff and CI**

Confirm:

- only planned files changed,
- no generated data/model files committed,
- no tokens/secrets in manifests or logs,
- PR workflow green,
- no unresolved review threads.

### Task 17: Run the full evaluation after code review approval

**Files:**
- No repository changes required unless the workflow exposes a reproducible defect.

- [ ] **Step 1: Dispatch the full workflow manually**

Use the approved defaults and exact branch commit SHA.

- [ ] **Step 2: Verify preparation artifacts**

Confirm data manifest reports 100/100 symbols and asset manifest contains exact model/tokenizer revisions.

- [ ] **Step 3: Verify all ten shard artifacts**

Each must contain `COMPLETED`, one unique shard index, non-overlapping symbols, and matching fingerprints.

- [ ] **Step 4: Verify final reduced artifact**

Confirm all required files exist, report title/limitations are correct, and summary counts reconcile with predictions/skips.

- [ ] **Step 5: Record results in the Draft PR**

Post workflow run ID, final artifact ID/hash, model revisions, eligible/skipped counts, and high-level Kronos-vs-baseline results. Do not claim profitability.

- [ ] **Step 6: Request a separate merge decision**

Do not mark ready, merge, schedule recurring runs, fine-tune, or connect a broker without explicit user approval.

---

## Final Acceptance Checklist

- [ ] Common cohorts use one forecast origin and five target dates for every compared symbol.
- [ ] Every model context contains exactly 400 rows ending at the origin close.
- [ ] No target value enters context, normalization, baseline construction, or model input.
- [ ] Kronos-mini and Tokenizer-2k exact revisions are recorded and consistent across shards.
- [ ] All three baselines are present for every eligible window.
- [ ] Undefined metrics remain missing, not zero-filled.
- [ ] Ten shards are deterministic, complete, non-overlapping, and reducer-validated.
- [ ] Network-free tests run on PRs without Torch, Yahoo, or model downloads.
- [ ] Full model evaluation is manual only and resource bounded.
- [ ] Generated data, models, and results are not committed.
- [ ] Report prominently states research-only, survivorship bias, and no investment advice.
- [ ] No order, broker, or live-trading functionality is introduced.
