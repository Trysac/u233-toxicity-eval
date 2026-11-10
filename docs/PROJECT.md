# Project record

## Objective

Evaluate toxicity of model outputs with interchangeable classifiers: a classifier port, a lexicon baseline, an LLM classifier, thresholds and per-group flag rates.

## Scope

- In scope: the public API documented in [README.md](../README.md), typed and tested. The package exposes `ClassifierPort`, `LexiconClassifier`, `LLMClassifier`, `Sample`, `evaluate`, `Report`, the LLM port helpers and the documented exceptions.
- In scope: input validation for sample count, sample ids, text length and thresholds; score validation; batch evaluation; per-label rates; per-group rates; flagged-id truncation; event callbacks; JSON reply parsing for the LLM classifier.
- Out of scope: bundled word lists or models; callers supply their own terms, labels, classifiers and LLM ports.
- Out of scope: hosted moderation API clients.
- Out of scope: calibration and multilingual normalisation.

## Timeline

| Date | Event |
|---|---|
| 2026 | Repository created from the factory utility template. |

## Effort metrics

Values are filled only after measurement; unmeasured rows remain `pending measurement`.

| Metric | Value |
|---|---|
| Tokens (input / output) | 465844 / 50281 |
| Hours (human) | pending measurement |
| Cost (USD) | 0.0772 |

## Factory run u233 (measured from API usage)

| Date (UTC) | Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|---|
| 2026-11-10 | generation | deepseek-v4-pro | yes | 1699 | 15415 | 0 | 3328 | 0.0317 |
| 2026-11-10 | fix-1 | deepseek-flash | no | 13926 | 5778 | 0 | 71808 | 0.0058 |
| 2026-11-10 | audit | deepseek-flash | no | 24011 | 12706 | 0 | 144640 | 0.0117 |
| 2026-11-10 | fix-2 | deepseek-flash | no | 9364 | 8103 | 0 | 86656 | 0.0065 |
| 2026-11-10 | docs | deepseek-flash | no | 110156 | 8279 | 0 | 256 | 0.0215 |

Total: $0.0772. Prices: deepseek line, as measured by its adapter. Providers serve the current version of each model id, so the date tells which version answered; versions, launch stages and price sources per line are in the factory's model register (docs/MODELS.md).
