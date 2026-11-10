# Architecture

## Overview

`toxicity_eval` provides a unified evaluation framework for toxicity classifiers.
The core abstraction is the `ClassifierPort` protocol, which any classifier can implement.
The library includes two built-in classifiers:

- `LexiconClassifier`: case-insensitive whole-word matching against user-provided term lists.
- `LLMClassifier`: invokes an LLM (via the `LLMPort` protocol) and parses its JSON response.

The `evaluate` function processes samples in batches, validates scores, groups by sample `group`,
and computes per-label flag rates and per-group rates. It also enforces safety limits and emits
events for monitoring.

## Key Design Decisions

1. **Single report shape**: All classifiers produce `Mapping[str, float]` per text, normalizing
   downstream analysis.
2. **Validation before classification**: Sample size, text length, ID uniqueness, and threshold
   validity are checked before any classification call, ensuring deterministic failures.
3. **Error handling**: LLM classifier parsing errors are caught per sample and counted as errors,
   while port classifier contract violations raise `ToxicityEvalError`.
4. **Resource limits**: Lexicon size, sample count, text length, and flagged IDs are capped to
   prevent unbounded memory or computation.

## Limitations

- No bundled word lists or models; users provide their own.
- Not a hosted moderation API client.
- No calibration or multilingual normalization.
