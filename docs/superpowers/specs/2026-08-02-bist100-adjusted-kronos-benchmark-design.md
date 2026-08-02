# BIST 100 Adjusted-Price Kronos Benchmark Design

## Status

Approved for specification on 2026-08-02. This document defines a research-only evaluation milestone. It does not authorize fine-tuning, portfolio deployment, broker integration, paper trading, or live trading.

## Purpose

Determine whether two controlled changes improve the previously completed BIST 100 zero-shot evaluation:

1. Replacing raw Yahoo OHLC prices with a corporate-action-adjusted OHLC view derived from `adj_close / close`.
2. Replacing `NeoQuasar/Kronos-mini` with `NeoQuasar/Kronos-small` while preserving the same evaluation population, dates, lookback, horizon, baselines, seeds, and reporting rules.

The benchmark must answer three separate questions without conflating them:

- Does adjusted data improve Kronos-mini relative to its raw-price reference run?
- Does Kronos-small outperform Kronos-mini when both use the same adjusted data?
- Does either adjusted Kronos configuration beat transparent baselines on return error, direction, and cross-sectional ranking?

Results are research evidence, not investment advice.

## Naming and Interpretation

The report must be named and described as:

> Paired adjusted-price zero-shot benchmark of Kronos-mini and Kronos-small on the 2026 Q3 BIST 100 constituent snapshot over 2023-2026.

It must not be described as a historical BIST 100 index backtest. The repository contains the 2026 Q3 constituent snapshot, not point-in-time historical index membership. Applying that snapshot backward introduces survivorship and selection bias.

The adjusted-price view is a Yahoo-derived research transformation. It is not equivalent to an official licensed Borsa Istanbul corporate-action history.

## Scope

### Included

- The 100 symbols in `data/universes/xu100_2026_q3.csv`.
- The immutable, previously verified Yahoo research artifact containing 100 raw symbol CSV files and its manifest.
- Raw reference results from the completed Kronos-mini benchmark when configuration and source fingerprints match.
- A derived adjusted OHLCVA dataset built from the raw artifact.
- Evaluation dates from 2023-01-01 through the last completed candle on or before 2026-08-02.
- A 400-trading-day lookback window.
- A five-trading-day forecast horizon.
- The same common monthly forecast origins and common five-date targets as the original benchmark.
- `NeoQuasar/Kronos-mini` with `NeoQuasar/Kronos-Tokenizer-2k`.
- `NeoQuasar/Kronos-small` with `NeoQuasar/Kronos-Tokenizer-base`.
- Deterministic baselines evaluated on the same adjusted series.
- Paired per-window comparisons and bootstrap confidence intervals.
- CSV, JSON, and Markdown outputs with complete run manifests.
- Network-free tests for all transformation, metric, pairing, and reducer logic.
- A separately triggered real-model benchmark workflow.

### Excluded

- Fine-tuning any Kronos model or tokenizer.
- Kronos-base or Kronos-large evaluation.
- Historical point-in-time BIST 100 reconstruction.
- Licensed corporate-action data ingestion.
- Intraday prediction.
- Portfolio optimization, position sizing, turnover controls, transaction costs, or order simulation.
- Ensemble models, calibration layers, or learned post-processing.
- Claims of profitability, production readiness, or investment suitability.

## Experimental Arms

The benchmark contains three logically distinct arms:

### Arm R: Raw Mini Reference

- Data view: raw Yahoo OHLCVA used by the completed benchmark.
- Model: `NeoQuasar/Kronos-mini`.
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-2k`.
- Purpose: frozen reference only.

The preferred implementation reuses the prior completed artifact instead of rerunning Arm R. Reuse is permitted only when the source-data fingerprint, universe, monthly cohorts, model revision, tokenizer revision, lookback, horizon, sampling parameters, and seed exactly match the recorded reference manifest. If any required field differs, the raw arm must be rerun or the paired raw-versus-adjusted comparison must be marked unavailable.

### Arm A-Mini: Adjusted Mini

- Data view: adjusted OHLCVA derived from the verified raw Yahoo artifact.
- Model: `NeoQuasar/Kronos-mini`.
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-2k`.
- Purpose: isolate the effect of the data transformation.

### Arm A-Small: Adjusted Small

- Data view: exactly the same adjusted OHLCVA files used by Arm A-Mini.
- Model: `NeoQuasar/Kronos-small`.
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-base`.
- Purpose: isolate the effect of model capacity and tokenizer family on the adjusted data.

All paired conclusions must use only windows present in both compared arms. Missing or skipped windows must never be imputed.

## Adjusted OHLCVA Transformation

The source raw CSV must contain finite `open`, `high`, `low`, `close`, `adj_close`, and `volume` values for every retained row.

For each row:

```text
adjustment_factor = adj_close / close
adjusted_open      = open  * adjustment_factor
adjusted_high      = high  * adjustment_factor
adjusted_low       = low   * adjustment_factor
adjusted_close     = close * adjustment_factor
adjusted_volume    = volume
adjusted_amount    = ((adjusted_high + adjusted_low + adjusted_close) / 3) * volume
```

### Factor Rules

- `close` must be strictly positive.
- `adj_close` must be strictly positive.
- `adjustment_factor` must be finite and strictly positive.
- No forward fill, backward fill, interpolation, clipping, or median substitution is allowed.
- A row failing any factor rule is fatal for that symbol in strict preparation mode.
- The transformation must not read future rows when creating any individual row; the factor is computed only from fields already present on the same source row.

### OHLC Envelope

After scaling, the adjusted frame must pass the repository's OHLC envelope contract:

- `high >= max(open, close, low)`
- `low <= min(open, close, high)`
- all price fields non-negative
- volume non-negative
- timestamps unique and strictly increasing after normalization

Floating-point noise may be handled only through the existing audited minimal envelope repair. Every changed cell must be recorded in the adjusted-data manifest. Silent repair is forbidden.

### Volume and Amount

Volume remains the raw reported Yahoo volume. The design intentionally does not reverse-adjust volume because the Yahoo artifact does not provide a separately verified split-volume contract. `amount` is therefore an estimated research field calculated from adjusted typical price and raw volume. Reports must state this limitation.

### Transformation Manifest

The adjusted-data artifact must include:

- source artifact identifier and digest
- source manifest digest
- transformation schema version
- formula identifier
- symbol count
- row count per symbol
- first and last timestamps per symbol
- minimum and maximum adjustment factor per symbol
- count of factor changes materially different from `1.0`
- audited OHLC repairs
- rejected symbols or rows with explicit reasons
- content fingerprint for every adjusted symbol file
- aggregate adjusted-data fingerprint

The transformation must be deterministic: the same source bytes and transformation version must produce identical adjusted CSV bytes and fingerprints.

## Population and Calendar Invariance

The adjusted benchmark must reuse the original experiment's population and calendar rules:

1. Start from all 100 symbols in the 2026 Q3 universe snapshot.
2. Build or load the canonical calendar from timestamp coverage only.
3. Use the first canonical date in each month as forecast origin.
4. Use the next five canonical dates as common targets.
5. Require exactly 400 complete symbol observations through the origin.
6. Require the origin and all five target dates for every eligible symbol-window.

Raw and adjusted views for one symbol are derived from the same source rows and therefore must have identical timestamp coverage. A mismatch in timestamps or row count between raw and adjusted files is fatal.

For a valid paired comparison, both model arms must share:

- symbol
- candidate month
- forecast origin
- five target timestamps
- history start and end timestamps
- lookback length
- horizon

## Leakage Prevention

For each symbol and forecast origin:

1. End the model context at the forecast-origin close.
2. Select exactly 400 observed rows ending at the origin.
3. Pass only the five common future timestamps as `y_timestamp`.
4. Keep actual target OHLC values inaccessible to the model and baseline functions until predictions are complete.
5. Compute adjustment factors only from each row's own raw and adjusted-close fields.
6. Do not use future corporate-action factors to rescale prior context rows beyond the per-row Yahoo adjusted value already present in the immutable source artifact.
7. Do not select windows, models, or sampling parameters based on target outcomes.

The prediction key becomes:

```text
(experiment_arm, symbol, forecast_origin, target_timestamp, method)
```

Duplicate keys are fatal.

## Model Configuration

### Shared Inference Parameters

- Lookback: `400`
- Prediction length: `5`
- Temperature: configurable, default `1.0`
- Top-p: configurable, default `0.9`
- Sample count: configurable, default `1`
- Required random seed: `20260802` unless explicitly overridden and recorded
- Device: CUDA when available, otherwise CPU
- Predictor maximum context: `512` for both comparison arms in this benchmark

The 400-row lookback is below the 512-row limit for Kronos-small and its base tokenizer.

### Adjusted Mini

- Model: `NeoQuasar/Kronos-mini`
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-2k`

### Adjusted Small

- Model: `NeoQuasar/Kronos-small`
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-base`

Exact resolved model and tokenizer revisions must be frozen before shard execution and recorded in every shard manifest. A model-tokenizer pairing mismatch is fatal.

## Baselines

Each adjusted eligible window must produce the existing transparent baselines from adjusted close history only:

- last close
- 20-day compounded momentum
- 20-day linear trend

The formulas must remain identical to the original benchmark. Baseline code must not branch on model arm. Baselines may be computed once per adjusted window and reused across A-Mini and A-Small outputs, provided the reducer validates exact equality.

The raw reference baseline metrics remain reference-only and are not mixed into adjusted-arm paired tests unless source and configuration compatibility is proven.

## Metrics

### Primary Per-Window Metric: Final Log-Return Absolute Error

For each method:

```text
actual_log_return    = log(actual_final_close / history_last_close)
predicted_log_return = log(predicted_final_close / history_last_close)
log_return_abs_error = abs(predicted_log_return - actual_log_return)
```

Rules:

- all three prices must be finite and strictly positive
- invalid denominators or non-positive values are rejected
- no epsilon substitution is allowed

This is the primary paired metric because it is comparable across securities with different price scales.

### Secondary Per-Window Metrics

- five-step close MAE
- five-step close RMSE
- final-horizon absolute percentage error
- predicted and realized five-day simple return
- predicted and realized five-day log return
- direction correctness
- final absolute price error

### Per-Symbol Metrics

For each experiment arm and method:

- eligible window count
- mean and median log-return absolute error
- mean and median final percentage error
- mean MAE and RMSE
- direction accuracy
- Pearson correlation of predicted and realized log returns when defined
- win rate versus each adjusted baseline

### Per-Period Ranking Metrics

For every common monthly cohort meeting the minimum cohort size:

- Spearman correlation of predicted versus realized five-day log return
- top-five overlap
- mean realized simple return of predicted top five
- mean realized log return of predicted top five
- eligible symbol count
- common origin and target range

The default minimum cohort remains 20 symbols.

## Paired Comparisons

The benchmark must produce window-level paired differences for:

1. A-Mini versus R-Mini, when the frozen raw reference is compatible.
2. A-Small versus A-Mini.
3. A-Mini versus adjusted last-close baseline.
4. A-Mini versus adjusted momentum baseline.
5. A-Mini versus adjusted linear-trend baseline.
6. A-Small versus each adjusted baseline.

For error metrics, define:

```text
difference = challenger_error - reference_error
```

A negative value favors the challenger.

For direction accuracy and ranking metrics, define the comparison orientation explicitly in the output schema so positive values always favor the challenger.

Paired files must include the common key, challenger value, reference value, signed difference, and winner label. Ties must remain ties.

## Bootstrap Confidence Intervals

The reducer must calculate deterministic non-parametric bootstrap intervals for the mean paired difference.

Default protocol:

- resampling unit: `(symbol, forecast_origin)` window
- bootstrap draws: `10,000`
- confidence level: `95%`
- random seed: required and recorded
- sampling: with replacement over valid paired windows

The primary interval is for mean final log-return absolute-error difference.

Decision rule:

- interval entirely below zero: challenger is statistically better on the paired error metric
- interval entirely above zero: challenger is statistically worse
- interval includes zero: no statistically clear difference

The report must include effect size, interval bounds, paired-window count, and the decision label. It must not convert statistical significance into profitability claims.

A secondary symbol-clustered bootstrap may be added only if explicitly included in the implementation plan. It is not required for this milestone.

## Architecture

Extend the existing isolated evaluation architecture without modifying Kronos model internals.

### `bist_eval/adjustments.py`

Owns:

- adjustment-factor calculation
- strict factor validation
- adjusted OHLCVA transformation
- deterministic transformation diagnostics
- adjusted manifest records

### `scripts/build_bist_adjusted_data.py`

CLI that converts the immutable raw Yahoo artifact into a versioned adjusted artifact. It accepts source/raw directories, source manifest, output directory, strict mode, and optional symbol subsets.

### `bist_eval/config.py`

Extend the evaluation configuration with explicit fields for:

- experiment arm
- data view
- model family
- tokenizer family
- adjustment schema version
- bootstrap configuration where relevant

Configuration fingerprints must differ between raw, adjusted-mini, and adjusted-small arms while retaining common cohort identifiers.

### `bist_eval/model_adapter.py`

Continue to own lazy loading and prediction. Add explicit validated model-tokenizer pairing metadata. Do not infer the tokenizer from the model name silently.

### `bist_eval/metrics.py`

Add guarded log-return metrics and paired-comparison helpers while preserving prior metric schemas where backward compatible.

### `bist_eval/comparison.py`

Owns:

- compatible-arm validation
- paired window joins
- signed differences
- winner/tie labels
- deterministic bootstrap intervals
- comparison decision labels

### `bist_eval/reporting.py`

Add stable schemas for experiment arms, paired comparisons, bootstrap intervals, and adjusted-data provenance.

### Evaluation CLI

Either extend `scripts/evaluate_bist100_zero_shot.py` with explicit arm/data-view parameters or add a thin benchmark-specific wrapper. The implementation plan must choose the smaller change that preserves backward compatibility and prevents ambiguous defaults.

## Data Flow

```text
verified Yahoo raw artifact
        |
        +--> raw mini reference artifact compatibility check
        |
        v
strict adj_close / close transformation
        |
        v
adjusted OHLCVA artifact + manifest + fingerprints
        |
        +--> shared canonical calendar and windows
        |
        +--> adjusted baselines
        |
        +--> Kronos-mini shards
        |
        +--> Kronos-small shards
        |
        v
arm-specific predictions and metrics
        |
        v
compatibility validation + paired joins
        |
        v
deterministic bootstrap intervals
        |
        v
CSV + JSON + Markdown report
```

## Execution Strategy

The real-model workflow is manual and resource bounded.

Recommended jobs:

1. `prepare-data`
   - obtain the immutable verified raw artifact
   - validate its digest and 100/100 manifest
   - build adjusted files
   - validate timestamp identity and adjusted manifest
   - upload the adjusted artifact

2. `prepare-model-mini`
   - resolve and freeze mini model and tokenizer revisions
   - upload immutable model artifact

3. `prepare-model-small`
   - resolve and freeze small model and base-tokenizer revisions
   - upload immutable model artifact

4. `evaluate-mini`
   - ten deterministic symbol shards
   - adjusted data only
   - load model once per shard

5. `evaluate-small`
   - ten deterministic symbol shards
   - same adjusted data and shard mapping
   - load model once per shard

6. `reduce`
   - validate all twenty shards
   - validate fingerprints and arm metadata
   - combine outputs
   - load compatible raw reference if available
   - compute adjusted aggregates, paired comparisons, bootstrap intervals, and final report

Concurrency and timeout limits must prevent uncontrolled compute use. The workflow should avoid redownloading the same model independently inside every shard.

## Output Contract

Default output directory:

```text
results/bist100-adjusted-benchmark/
├── adjusted_data_manifest.json
├── predictions.csv
├── window_metrics.csv
├── skipped_windows.csv
├── symbol_metrics.csv
├── period_metrics.csv
├── ranking_metrics.csv
├── paired_comparisons.csv
├── bootstrap_intervals.csv
├── summary.json
├── run_manifest.json
└── report.md
```

### Required Experiment Fields

Prediction and metric outputs must include:

- `experiment_arm`
- `data_view`
- `model_id`
- `tokenizer_id`
- `model_revision`
- `tokenizer_revision`
- `adjustment_schema_version`

### `paired_comparisons.csv`

At minimum:

- comparison identifier
- challenger arm/method
- reference arm/method
- symbol
- forecast origin
- primary challenger error
- primary reference error
- signed difference
- winner label

### `bootstrap_intervals.csv`

At minimum:

- comparison identifier
- metric
- paired-window count
- bootstrap draws
- seed
- mean difference
- lower bound
- upper bound
- confidence level
- decision

### `summary.json`

Must answer directly:

- whether adjusted mini improved over compatible raw mini
- whether adjusted small improved over adjusted mini
- whether either model beat each adjusted baseline
- direction accuracy for each model
- average ranking correlation for each model
- number of eligible and paired windows
- unavailable comparisons and explicit reasons

All schemas must be versioned.

## Failure Handling

- Missing `adj_close` is fatal for adjusted preparation in strict mode.
- Non-positive or non-finite `close`, `adj_close`, factor, or adjusted price is fatal for that symbol in strict mode.
- Timestamp or row-count mismatch between raw and adjusted views is fatal.
- Unrecorded OHLC repair is fatal.
- Model-tokenizer pairing mismatch is fatal.
- Missing model revision, tokenizer revision, source digest, or adjustment fingerprint is fatal for a real benchmark shard.
- A shard with NaN, infinite, non-positive predicted close, duplicate keys, or incomplete output must not write `COMPLETED`.
- Reducer configuration or fingerprint mismatch is fatal.
- Missing expected shards are fatal.
- Raw-reference incompatibility makes only the raw-versus-adjusted comparison unavailable; it must not invalidate the adjusted mini-versus-small benchmark.
- Insufficient history and missing target dates remain normal, reported skips.
- Bootstrap comparison with zero paired windows is unavailable, not zero effect.
- Outputs are written atomically, with a completion marker only after schema and row-count checks pass.

## Testing

### Adjustment Unit Tests

Network-free synthetic tests must cover:

- exact factor calculation
- identity factor
- split-like factor
- dividend-like factor
- adjusted OHLC scaling
- raw volume preservation
- adjusted amount calculation
- non-positive close rejection
- non-positive adjusted close rejection
- non-finite factor rejection
- timestamp and row-count preservation
- deterministic output bytes and fingerprints
- audited floating-point OHLC envelope repair

### Metric Tests

- exact simple and log return calculations
- log-return absolute error
- positive-price guards
- direction correctness
- per-symbol aggregates
- ranking metrics using log returns
- constant-series and insufficient-cohort handling

### Comparison Tests

- exact paired joins
- exclusion of unmatched windows
- signed-difference orientation
- tie handling
- model-arm metadata validation
- raw-reference compatibility checks
- deterministic bootstrap results with fixed seed
- confidence interval and decision labels
- empty and single-window edge cases

### Model Adapter Tests

Using fake predictors:

- mini and small pair with their explicit tokenizers
- mismatched model-tokenizer combinations fail
- 400 rows are passed without exceeding 512 context
- common timestamps and columns are preserved
- targets remain inaccessible
- invalid predictions fail closed

### Reducer Tests

- twenty expected shard validation
- missing mini or small shard rejection
- data fingerprint mismatch rejection
- adjustment schema mismatch rejection
- model revision mismatch rejection
- duplicate prediction key rejection
- shared-baseline consistency checks
- compatible raw artifact reuse
- output schema and completion marker

### Real-Model Smoke Test

Before the full benchmark, run both adjusted models on the same two symbols and two common forecast origins. The smoke test succeeds only when:

- adjusted data preparation succeeds
- both exact model-tokenizer pairs load
- both models return aligned five-date predictions
- output schemas validate
- paired mini-versus-small metrics and a deterministic bootstrap artifact are produced

The smoke result is a technical gate, not performance evidence.

## Acceptance Criteria

The milestone is complete when:

1. The adjusted artifact is reproducibly derived from the verified raw artifact with full provenance.
2. All network-free tests pass.
3. The two-model real smoke test succeeds.
4. Ten mini shards and ten small shards complete or report a reproducible infrastructure limitation.
5. The reducer validates every shard and produces schema-valid final artifacts.
6. Mini-versus-small comparisons use exactly matched symbol-window keys.
7. Adjusted-mini versus raw-mini is reported only when reference compatibility is proven.
8. The final report includes primary log-return error, direction, ranking, baseline comparisons, and bootstrap intervals.
9. The report states survivorship bias, Yahoo data limitations, raw-volume treatment, and research-only status.
10. No fine-tuning, portfolio construction, or order-placement capability is introduced.

## Security, Cost, and Operational Boundaries

- Repository and workflow permissions remain read-only except standard artifact operations.
- No brokerage, exchange, or private Hugging Face credentials are required.
- Public model assets must be pinned to exact resolved revisions.
- Data and model artifacts must carry digests.
- Full benchmark execution remains manual.
- Workflow concurrency and timeout controls must prevent duplicate or unbounded runs.
- Fine-tuning remains a separate future milestone requiring explicit design, plan, compute budget, and leakage controls.

## Known Limitations

- Yahoo is not an official licensed Borsa Istanbul source.
- `adj_close / close` is a provider-derived adjustment and may combine split and dividend effects in ways that require independent validation.
- Volume remains raw and is not reverse-adjusted.
- `amount` remains an estimate.
- The 2026 Q3 universe projected backward creates survivorship and selection bias.
- Monthly anchors do not measure every possible forecast origin.
- One stochastic sample per window may produce sampling variance even with fixed seeds.
- Statistical superiority on forecast error does not establish a profitable strategy.
- Comparing model families also changes tokenizer family, so the benchmark attributes improvement to the model-tokenizer system, not model weights alone.

## Future Milestones

Only after this benchmark is reviewed:

1. Validate corporate actions against a licensed source.
2. Test repeated sampling and uncertainty bands.
3. Reconstruct point-in-time constituent histories.
4. Add transaction-cost-aware portfolio diagnostics.
5. Consider BIST-specific fine-tuning only if adjusted zero-shot evidence justifies the additional complexity and compute.
