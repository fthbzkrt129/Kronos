# BIST 100 Origin-Rebased Kronos Benchmark Design

## Status

Approved for specification on 2026-08-02 and revised during specification review to remove a potential corporate-action look-ahead path. This document defines a research-only evaluation milestone. It does not authorize fine-tuning, portfolio deployment, broker integration, paper trading, or live trading.

## Purpose

Determine whether two controlled changes improve the previously completed BIST 100 zero-shot evaluation:

1. Replacing raw OHLC context with an origin-rebased corporate-action-adjusted view derived from Yahoo `adj_close / close` factors without exposing future factors to the predictor.
2. Replacing `NeoQuasar/Kronos-mini` with `NeoQuasar/Kronos-small` while preserving the same population, forecast origins, lookback, horizon, sampling parameters, and scoring targets.

The benchmark must answer three separate questions:

- Does origin-rebased adjusted context improve Kronos-mini relative to raw-context Kronos-mini when both are scored against the same adjusted target?
- Does Kronos-small outperform Kronos-mini when both use the same origin-rebased adjusted context?
- Does either adjusted Kronos configuration beat transparent baselines on return error, direction, and cross-sectional ranking?

Results are research evidence, not investment advice.

## Naming and Interpretation

The report must be named and described as:

> Paired origin-rebased zero-shot benchmark of Kronos-mini and Kronos-small on the 2026 Q3 BIST 100 constituent snapshot over 2023-2026.

It must not be described as a historical BIST 100 index backtest. The repository contains the 2026 Q3 constituent snapshot, not point-in-time historical index membership. Applying that snapshot backward introduces survivorship and selection bias.

The origin-rebased view is a Yahoo-derived research transformation. It is not equivalent to an official licensed Borsa Istanbul corporate-action history or total-return index.

## Why a Static Adjusted File Is Forbidden

Yahoo `adj_close` is a provider-maintained historical series that may reflect corporate actions occurring after an earlier forecast origin. Building one static historical OHLC file with `adj_close / close` and then using it for every past forecast could therefore expose later adjustment information to earlier model contexts.

This milestone must not create a single static adjusted-price CSV for model input.

Instead, every eligible forecast window receives its own transformation anchored at that window's forecast origin. The origin factor normalizes away adjustment components common to both the historical row and the origin. Target factors are applied only after prediction, for scoring.

## Scope

### Included

- The 100 symbols in `data/universes/xu100_2026_q3.csv`.
- The immutable, previously verified Yahoo research artifact containing 100 raw symbol CSV files and its manifest.
- A validated per-row factor series derived from raw `adj_close / close`.
- Window-specific origin-rebased adjusted contexts.
- A common origin-rebased adjusted target used to score every model arm and baseline.
- Evaluation dates from 2023-01-01 through the last completed candle on or before 2026-08-02.
- A 400-trading-day lookback window.
- A five-trading-day forecast horizon.
- The same common monthly forecast origins and common five-date targets as the original benchmark.
- Raw-context `NeoQuasar/Kronos-mini` as the controlled reference arm.
- Adjusted-context `NeoQuasar/Kronos-mini`.
- Adjusted-context `NeoQuasar/Kronos-small`.
- `NeoQuasar/Kronos-Tokenizer-2k` for mini and `NeoQuasar/Kronos-Tokenizer-base` for small.
- Deterministic adjusted-context baselines.
- Paired per-window comparisons and clustered bootstrap confidence intervals.
- CSV, JSON, and Markdown outputs with complete provenance.
- Network-free tests for transformation, leakage prevention, metrics, pairing, and reducer logic.
- A separately triggered real-model benchmark workflow.

### Excluded

- Fine-tuning any Kronos model or tokenizer.
- Kronos-base or Kronos-large evaluation.
- A static adjusted model-input dataset.
- Historical point-in-time BIST 100 reconstruction.
- Licensed corporate-action data ingestion.
- Intraday prediction.
- Portfolio optimization, position sizing, turnover controls, transaction costs, or order simulation.
- Ensemble models, calibration layers, or learned post-processing.
- Claims of profitability, production readiness, or investment suitability.

## Experimental Arms

### Arm R-Mini: Raw Context, Mini Model

- Context view: raw Yahoo OHLCVA.
- Model: `NeoQuasar/Kronos-mini`.
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-2k`.
- Scoring target: common origin-rebased adjusted target.
- Purpose: isolate the effect of adjusted context.

The prior full benchmark artifact may be reused only at the prediction level. Its old metrics used raw targets and are not valid for this benchmark's common adjusted target.

Prior predictions may be reused only when all of the following match exactly:

- source-data fingerprint
- universe and symbol order
- canonical monthly cohorts
- model and tokenizer revisions
- lookback and horizon
- temperature, top-p, sample count, and seed
- prediction schema

If compatible predictions are unavailable, Arm R-Mini must be rerun. Old aggregate metrics must never be imported as if they were comparable.

### Arm A-Mini: Origin-Rebased Context, Mini Model

- Context view: window-specific origin-rebased adjusted OHLCVA.
- Model: `NeoQuasar/Kronos-mini`.
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-2k`.
- Scoring target: the same origin-rebased adjusted target.
- Purpose: isolate the context transformation.

### Arm A-Small: Origin-Rebased Context, Small Model

- Context view: exactly the same transformed windows used by A-Mini.
- Model: `NeoQuasar/Kronos-small`.
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-base`.
- Scoring target: the same origin-rebased adjusted target.
- Purpose: isolate the model-tokenizer system change.

All conclusions must use only symbol-windows present in both compared arms. Missing windows must not be imputed.

## Factor Series Validation

For each raw symbol row:

```text
provider_factor_t = adj_close_t / close_t
```

Rules:

- `open`, `high`, `low`, `close`, `adj_close`, and `volume` must be finite.
- `close` and `adj_close` must be strictly positive.
- `provider_factor_t` must be finite and strictly positive.
- Timestamps must be unique and strictly increasing after normalization.
- No forward fill, backward fill, interpolation, clipping, or substitution is allowed.
- Invalid rows are fatal for that symbol in strict preparation mode.

The validated factor series is diagnostic input, not a model-ready adjusted file.

## Window-Specific Origin-Rebased Transformation

For a forecast window with origin row `o`, define:

```text
relative_factor_t(o) = provider_factor_t / provider_factor_o
```

For every context row `t <= o`:

```text
rebased_open_t   = open_t  * relative_factor_t(o)
rebased_high_t   = high_t  * relative_factor_t(o)
rebased_low_t    = low_t   * relative_factor_t(o)
rebased_close_t  = close_t * relative_factor_t(o)
rebased_volume_t = volume_t
rebased_amount_t = ((rebased_high_t + rebased_low_t + rebased_close_t) / 3) * volume_t
```

At the origin, `relative_factor_o(o) = 1`, so the rebased origin price equals the raw origin price. This gives every model arm and target a common origin scale.

### Leakage Boundary

The predictor and baseline functions may receive only:

- rebased context rows through the origin
- context timestamps
- future target timestamps

They must not receive:

- target raw prices
- target adjusted prices
- target provider factors
- target relative factors
- future corporate-action diagnostics

Target transformation occurs only after all predictions for that window are produced.

### Common Scoring Target

For each held-out target row `u > o`, calculate after prediction:

```text
relative_factor_u(o) = provider_factor_u / provider_factor_o
actual_rebased_close_u = raw_close_u * relative_factor_u(o)
```

The same `actual_rebased_close_u` is used to score R-Mini, A-Mini, A-Small, and every baseline. This prevents different arms from being compared against different definitions of realized price.

Provider adjustment components occurring after both `o` and `u` are expected to cancel in the factor ratio. This is a research assumption that must be stated in the report and independently validated before any production use.

## OHLC and Amount Rules

Each rebased context must pass the repository's OHLC envelope contract:

- `high >= max(open, close, low)`
- `low <= min(open, close, high)`
- all price fields strictly positive for model input
- volume non-negative

Floating-point noise may be handled only through the existing audited minimal envelope repair. Silent repair is forbidden.

Volume remains raw. The design does not reverse-adjust volume because the Yahoo artifact does not provide a separately verified split-volume contract. `amount` remains an estimated field based on rebased typical price and raw volume.

## Transformation Provenance

The benchmark preparation artifact must include:

- source artifact identifier and digest
- source manifest digest
- factor-validation schema version
- origin-rebasing formula version
- symbol and row counts
- first and last timestamp per symbol
- minimum and maximum provider factor per symbol
- count and dates of material factor changes
- invalid row diagnostics
- audited envelope repairs produced during test fixtures or window creation
- content fingerprint for every raw symbol file
- aggregate factor-series fingerprint

The same raw bytes, formula version, origin, and context rows must produce identical rebased context bytes or canonical numeric fingerprints.

## Population and Calendar Invariance

The benchmark reuses the original experiment's population and calendar rules:

1. Start from all 100 symbols in the 2026 Q3 universe snapshot.
2. Build or load the canonical calendar from timestamp coverage only.
3. Use the first canonical date in each month as forecast origin.
4. Use the next five canonical dates as common targets.
5. Require exactly 400 complete symbol rows through the origin.
6. Require the origin and all five target dates for every eligible symbol-window.

For a valid paired comparison, all arms must share:

- symbol
- candidate month
- forecast origin
- five target timestamps
- history start and end timestamps
- lookback length
- horizon
- common adjusted scoring target fingerprint

## Model Configuration

### Shared Inference Parameters

- Lookback: `400`
- Prediction length: `5`
- Temperature: configurable, default `1.0`
- Top-p: configurable, default `0.9`
- Sample count: configurable, default `1`
- Required random seed: `20260802` unless explicitly overridden and recorded
- Device: CUDA when available, otherwise CPU
- Predictor maximum context: `512` for comparison consistency

The 400-row lookback is below Kronos-small's 512-row context limit.

### Mini Pairing

- Model: `NeoQuasar/Kronos-mini`
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-2k`

### Small Pairing

- Model: `NeoQuasar/Kronos-small`
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-base`

Exact resolved revisions must be frozen before shard execution and recorded in every manifest. A model-tokenizer pairing mismatch is fatal. Tokenizer selection must never be inferred silently from a model string.

## Baselines

Every eligible window produces the existing transparent baselines from the rebased context only:

- last close
- 20-day compounded momentum
- 20-day linear trend

The formulas remain identical to the original benchmark. Baselines must not receive target factors or target prices.

Baselines may be computed once per transformed window and shared between A-Mini and A-Small outputs, provided the reducer verifies exact equality.

## Metrics

### Primary Per-Window Metric

Final five-day log-return absolute error:

```text
actual_log_return    = log(actual_rebased_final_close / origin_close)
predicted_log_return = log(predicted_final_close / origin_close)
log_return_abs_error = abs(predicted_log_return - actual_log_return)
```

All values must be finite and strictly positive. No epsilon substitution is allowed.

### Secondary Per-Window Metrics

- five-step close MAE on the common rebased target
- five-step close RMSE
- final-horizon absolute percentage error
- predicted and realized five-day simple return
- predicted and realized five-day log return
- direction correctness
- final absolute price error

### Per-Symbol Metrics

For each arm and method:

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

### Corporate-Action Exposure Diagnostics

Each window must record:

- whether provider factor changed anywhere in the 400-row context
- whether provider factor changed between origin and final target
- maximum absolute log change in provider factor within context
- maximum absolute log change between origin and target

Aggregate results must be reported for:

- all paired windows
- windows with no material factor change
- windows with a material context or target factor change

The material-change tolerance must be explicit, configurable, and recorded. Default: absolute relative factor change greater than `1e-8`.

## Paired Comparisons

Required comparisons:

1. A-Mini versus R-Mini.
2. A-Small versus A-Mini.
3. A-Mini versus rebased last-close baseline.
4. A-Mini versus rebased momentum baseline.
5. A-Mini versus rebased linear-trend baseline.
6. A-Small versus each rebased baseline.

For error metrics:

```text
difference = challenger_error - reference_error
```

A negative value favors the challenger.

Paired outputs must include the common key, challenger value, reference value, signed difference, winner label, and corporate-action exposure bucket. Ties remain ties.

## Clustered Bootstrap Confidence Intervals

Repeated monthly windows for one symbol are not independent, and symbols within one month share market conditions. A simple row-level bootstrap could overstate certainty.

The reducer must therefore calculate two deterministic non-parametric intervals for the mean paired difference:

### Symbol-Clustered Bootstrap

- Resample symbols with replacement.
- Include all paired windows belonging to each sampled symbol.
- Preserve within-symbol temporal dependence.

### Origin-Clustered Bootstrap

- Resample forecast origins with replacement.
- Include all paired symbols belonging to each sampled origin.
- Preserve cross-sectional dependence within a market period.

Shared defaults:

- bootstrap draws: `10,000`
- confidence level: `95%`
- fixed recorded seed

Decision labels:

- `robustly_better`: both intervals are entirely below zero
- `robustly_worse`: both intervals are entirely above zero
- `mixed_or_inconclusive`: otherwise

The report must include mean effect, both intervals, cluster counts, paired-window count, and decision. Statistical significance must not be translated into profitability claims.

## Architecture

Extend the existing evaluation architecture without modifying Kronos model internals.

### `bist_eval/adjustments.py`

Owns:

- provider-factor validation
- origin-relative factor calculation
- context rebasing
- post-prediction target rebasing
- corporate-action exposure diagnostics
- deterministic transformation fingerprints

### `bist_eval/windows.py`

Extend the forecast-window contract to retain raw context, raw target, validated provider factors, and a target accessor that is unavailable to predictor-facing code.

The implementation must make it structurally difficult to pass target factors or target prices to the model adapter.

### `bist_eval/config.py`

Add explicit fields for:

- experiment arm
- context view
- scoring target view
- model-tokenizer pairing
- adjustment formula version
- material factor-change tolerance

Arm fingerprints must differ while common cohort and target fingerprints remain comparable.

### `bist_eval/model_adapter.py`

Continue lazy loading and prediction. Add explicit validated model-tokenizer pairing metadata. The adapter accepts already prepared context only and must not import target-adjustment helpers.

### `bist_eval/metrics.py`

Add guarded log-return metrics using the common rebased target.

### `bist_eval/comparison.py`

Owns:

- compatible-arm validation
- paired window joins
- signed differences
- winner/tie labels
- exposure buckets
- symbol-clustered bootstrap
- origin-clustered bootstrap
- final decision labels

### `bist_eval/reporting.py`

Add stable schemas for experiment arms, common target provenance, paired comparisons, bootstrap intervals, and adjustment diagnostics.

### Evaluation CLI

Either extend `scripts/evaluate_bist100_zero_shot.py` with explicit arm/context/target parameters or add a thin benchmark wrapper. The implementation plan must choose the smaller backward-compatible option with no ambiguous default.

### Preparation CLI

Add a CLI that validates factor series and writes a factor/provenance manifest. It must not write static adjusted model-input CSVs.

## Data Flow

```text
immutable Yahoo raw artifact
        |
        +--> timestamp calendar
        |
        +--> validated provider-factor series
        |
        v
raw symbol window at common origin
        |
        +--> R-Mini raw context
        |
        +--> origin-rebased context --> A-Mini
        |                           --> A-Small
        |                           --> baselines
        |
        +--> model predictions complete
        |
        v
post-prediction common rebased target
        |
        v
arm metrics + paired joins
        |
        v
symbol- and origin-clustered bootstrap
        |
        v
CSV + JSON + Markdown report
```

## Execution Strategy

The real-model workflow remains manual and resource bounded.

Recommended jobs:

1. `prepare-data`
   - obtain the immutable verified raw artifact
   - validate digest and 100/100 manifest
   - validate factor series
   - freeze canonical calendar and factor manifest

2. `prepare-model-mini`
   - resolve and freeze mini model and tokenizer revisions

3. `prepare-model-small`
   - resolve and freeze small model and tokenizer revisions

4. `evaluate-raw-mini`
   - reuse compatible prior predictions or run ten deterministic shards
   - rescore against the common rebased target

5. `evaluate-adjusted-mini`
   - ten deterministic shards

6. `evaluate-adjusted-small`
   - ten deterministic shards using identical windows and context fingerprints

7. `reduce`
   - validate all required shards or compatible reference predictions
   - validate common target fingerprints
   - combine outputs
   - compute arm aggregates, paired comparisons, exposure diagnostics, clustered intervals, and report

Model artifacts should be resolved once per model family and reused by shards. Concurrency and timeout limits must prevent uncontrolled compute use.

## Output Contract

Default directory:

```text
results/bist100-origin-rebased-benchmark/
├── factor_manifest.json
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
- `context_view`
- `scoring_target_view`
- `model_id`
- `tokenizer_id`
- exact model and tokenizer revisions
- adjustment formula version
- common target fingerprint
- corporate-action exposure fields

### `paired_comparisons.csv`

At minimum:

- comparison identifier
- challenger arm/method
- reference arm/method
- symbol
- forecast origin
- challenger primary error
- reference primary error
- signed difference
- winner label
- exposure bucket

### `bootstrap_intervals.csv`

At minimum:

- comparison identifier
- metric
- clustering method
- cluster count
- paired-window count
- bootstrap draws
- seed
- mean difference
- lower bound
- upper bound
- confidence level
- interval decision
- combined robust decision

### `summary.json`

Must answer directly:

- whether adjusted context improved mini under a common target
- whether small improved over adjusted mini
- whether either model beat each rebased baseline
- direction accuracy for each arm
- average ranking correlation for each arm
- results by factor-exposure bucket
- paired-window and cluster counts
- unavailable comparisons and reasons

All schemas must be versioned.

## Failure Handling

- Missing `adj_close` is fatal in strict preparation mode.
- Non-positive or non-finite required prices or factors are fatal for that symbol.
- Any static adjusted model-input file path is rejected by this benchmark workflow.
- Target factors or target values entering predictor inputs are a test failure and runtime contract violation.
- Model-tokenizer mismatch is fatal.
- Missing exact revisions, source digest, factor fingerprint, or common target fingerprint is fatal for real-model shards.
- NaN, infinite, or non-positive predicted close is fatal in strict mode.
- Duplicate prediction keys are fatal.
- Reducer configuration, cohort, source, factor, or target fingerprint mismatch is fatal.
- Missing required shards are fatal.
- Incompatible prior predictions make reuse unavailable; they do not silently fall back to old metrics.
- Insufficient history and missing target dates remain normal reported skips.
- A comparison with no paired windows is unavailable, not zero effect.
- Completion markers are written only after schema and row-count validation.

## Testing

### Factor and Rebasing Tests

Network-free synthetic tests must cover:

- exact provider-factor calculation
- identity factors
- split-like and dividend-like factor changes
- origin factor equals one after rebasing
- cancellation of a common future multiplicative adjustment from all pre-origin factors
- context transformation uses no target factor
- target transformation is inaccessible until post-prediction scoring
- exact rebased OHLC values
- raw volume preservation
- rebased amount calculation
- non-positive and non-finite rejection
- deterministic context and target fingerprints
- audited floating-point envelope repair

### Leakage Tests

- predictor receives context rows and future timestamps only
- predictor cannot access raw target frame
- predictor cannot access target provider factors
- baseline functions cannot access target values or factors
- changing target factors must not change model or baseline predictions
- changing a factor after the final target must not change context or scored target ratios

### Metric Tests

- common-target simple and log returns
- log-return absolute error
- positive-price guards
- direction correctness
- symbol aggregates
- ranking metrics using log returns
- constant-series and insufficient-cohort handling
- exposure bucket assignment

### Comparison and Bootstrap Tests

- exact paired joins
- exclusion of unmatched windows
- signed-difference orientation
- tie handling
- arm and target fingerprint validation
- deterministic symbol-clustered intervals
- deterministic origin-clustered intervals
- robust decision labels
- empty, single-cluster, and degenerate cases

### Model Adapter Tests

Using fake predictors:

- mini and small pair with explicit tokenizers
- mismatches fail
- 400 rows fit the 512 context contract
- common timestamps and columns are preserved
- target data remains inaccessible
- invalid predictions fail closed

### Reducer Tests

- all required arm shards or compatible reference predictions
- missing shard rejection
- source/factor/target fingerprint mismatch rejection
- model revision mismatch rejection
- duplicate key rejection
- shared-baseline equality
- output schema and completion marker

### Real-Model Smoke Test

Run all three arms on the same two symbols and two common origins. The smoke test succeeds only when:

- factor validation succeeds
- raw and rebased contexts share timestamps and origin scale
- both exact model-tokenizer pairs load
- predictions align with five target dates
- common rebased target scoring succeeds
- paired comparisons and both clustered interval files are generated

The smoke result is a technical gate, not performance evidence.

## Acceptance Criteria

The milestone is complete when:

1. Factor series is reproducibly validated from the immutable raw artifact.
2. No static adjusted model-input file is used.
3. Leakage tests prove target factors and values cannot affect predictions.
4. All network-free tests pass.
5. The three-arm real smoke test succeeds.
6. Full required shards complete or report a reproducible infrastructure limitation.
7. The reducer produces schema-valid final artifacts with common target fingerprints.
8. Raw-mini versus adjusted-mini uses the same target and exactly matched windows.
9. Adjusted-small versus adjusted-mini uses identical transformed contexts.
10. The report includes primary log-return error, direction, ranking, exposure buckets, baseline comparisons, and both clustered intervals.
11. The report states survivorship bias, Yahoo assumptions, raw-volume treatment, and research-only status.
12. No fine-tuning, portfolio construction, or order placement is introduced.

## Security, Cost, and Operational Boundaries

- Repository and workflow permissions remain read-only except standard artifact operations.
- No brokerage, exchange, or private Hugging Face credentials are required.
- Public model assets must be pinned to exact resolved revisions.
- Data and model artifacts carry digests.
- Full benchmark execution remains manual.
- Concurrency and timeout controls prevent duplicate or unbounded runs.
- Fine-tuning remains a separate future milestone requiring explicit design, plan, compute budget, and leakage controls.

## Known Limitations

- Yahoo is not an official licensed Borsa Istanbul source.
- The factor-ratio cancellation behavior is a research assumption based on provider-adjusted series and requires independent validation.
- Volume remains raw and is not reverse-adjusted.
- `amount` remains an estimate.
- The 2026 Q3 universe projected backward creates survivorship and selection bias.
- Monthly anchors do not measure every possible forecast origin.
- One stochastic sample per window may produce sampling variance even with fixed seeds.
- Clustered intervals reduce but do not eliminate dependence concerns.
- Comparing mini and small also changes tokenizer family, so any improvement belongs to the model-tokenizer system.
- Better forecast error does not establish a profitable strategy.

## Future Milestones

Only after this benchmark is reviewed:

1. Validate corporate actions against a licensed source.
2. Test repeated sampling and uncertainty bands.
3. Reconstruct point-in-time constituent histories.
4. Add transaction-cost-aware portfolio diagnostics.
5. Consider BIST-specific fine-tuning only if origin-rebased zero-shot evidence justifies the complexity and compute.
