# Security notes

## Threat model

Defended by the library:

- Input volume and shape: `evaluate` rejects non-`Sample` sequences with `TypeError`; more than `max_samples` raises `too_many_samples`; text longer than `max_text_chars` raises `text_too_large`; duplicate sample ids raise `duplicate_sample`; thresholds that are not finite numbers in [0, 1] raise `invalid_thresholds`. Validation runs on every sample before the first `classify` call.
- Classifier score shape: for port classifiers, every threshold label must be present and finite in [0, 1], otherwise `invalid_score` is raised; a classifier returning a different number of results than texts raises `classifier_mismatch`.
- LLM prompt boundary: the text inserted into the user message is HTML-escaped with `html.escape(..., quote=False)`, so `&`, `<` and `>` become entities. A literal `</text>` in a sample is sent as `&lt;/text&gt;` and cannot close the tag. The fixed instruction says to ignore instructions inside the text.
- LLM reply parsing: the reply is trimmed, one optional fenced code block is removed, and the result must be a JSON object containing every configured label as a finite number in [0, 1]. Invalid replies become per-sample `errors` when called through `evaluate`; `LLMClassifier` raises `InvalidReply` and can carry already-computed batch scores and invalid indices.
- Report and event leakage: error objects use stable codes and messages that do not contain sample text. Event payloads contain only the event name, batch size or sample id; they do not contain sample text or reply text.
- Documented limits: `max_samples=100_000`, `max_text_chars=100_000`, `max_flagged=1_000`, `batch_size=32`; lexicon limits of 50 labels and 1,000 terms per label; `LLMClassifier.max_tokens` defaults to 200.

Not defended by the library:

- Semantic prompt injection: escaping tags does not prevent a model from following text content; the instruction to ignore instructions is not a model-level guarantee.
- Classifier quality, calibration, moderation decisions and provider behaviour. The library validates shapes and ranges, not whether a score is correct.
- Malicious `ClassifierPort` or `LLMPort` implementations supplied by the caller. The library calls them as provided and does not sandbox them.
- Network, authentication, rate limiting, secrets storage and hosting concerns. The package has no runtime dependencies and performs no network I/O itself.
- Supply-chain or CI compromise, and resource limits beyond the documented in-process caps.

## Publishing

The package does not implement publishing. Release publishing is repository infrastructure, not part of this library.
