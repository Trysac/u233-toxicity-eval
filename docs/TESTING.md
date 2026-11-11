# Testing

## Running

```sh
uv run pytest
uv run pytest tests/test_toxicity_eval.py
```

## Policy

- Tests exercise the public API through `import toxicity_eval` and check return values, exceptions, exception codes and emitted event payloads.
- LLM behaviour is tested with `RecordedLLM` and `FakeLLM`; tests do not use a network or a live model.
- Tests are deterministic: recorded replies are consumed in order, and `RecordingExhaustedError` is asserted to propagate.

## Tiers

The listed tests are a single pytest module. No separate conformance tier or external-tool test tier is present in the listed test structure.

## Layout

The test file is `tests/test_toxicity_eval.py`. It contains these tests:

- `test_lexicon_acceptance`: whole-word, case-insensitive lexicon matches.
- `test_evaluate_acceptance_example`: `evaluate` returns the expected `rates`, `group_rates`, `flagged` ids and `errors` for the sample set.
- `test_threshold_equal_is_flagged`: a score exactly equal to the threshold is flagged.
- `test_invalid_score_above_range`: a score above 1 raises `ToxicityEvalError` with code `invalid_score`.
- `test_invalid_score_nan`: a NaN score raises `ToxicityEvalError` with code `invalid_score`.
- `test_classifier_mismatch`: a classifier result count that differs from the text count raises `ToxicityEvalError` with code `classifier_mismatch`.
- `test_llm_classifier_acceptance`: `LLMClassifier` parses a JSON reply and sends the exact documented system and user messages.
- `test_llm_classifier_fenced_block`: a reply wrapped in a fenced JSON block is parsed.
- `test_llm_classifier_invalid_reply`: a non-JSON reply raises `InvalidReply`.
- `test_llm_classifier_missing_label`: a reply missing a configured label raises `InvalidReply`.
- `test_evaluate_llm_invalid_reply_counts_error`: invalid LLM replies are counted as errors by `evaluate`.
- `test_evaluate_llm_invalid_reply_excluded_from_denominator`: invalid LLM replies are excluded from every rate denominator.
- `test_evaluate_llm_recording_exhausted_propagates`: `RecordingExhaustedError` from the LLM port propagates.
- `test_text_too_large`: a text of 100,001 characters raises `text_too_large`.
- `test_text_under_limit_ok`: a text of 100,000 characters is accepted.
- `test_escape_text_in_llm_classifier`: `</text>` in a sample is sent as `&lt;/text&gt;`.
- `test_max_flagged_truncation`: with `max_flagged=2` and three flagged samples, `flagged` holds the first two ids in input order and `flagged_truncated` is `True`.
