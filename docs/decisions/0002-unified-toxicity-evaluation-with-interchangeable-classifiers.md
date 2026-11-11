# Unified toxicity evaluation with interchangeable classifiers

Status: accepted (generated, pending human review)

## Context

Evaluating toxicity requires switching between hosted APIs, local models, and lexicons,
each with different output shapes. This leads to repeated evaluation code and inconsistency.

## Decision

Define a single ClassifierPort interface and evaluate function that outputs a standard Report
with flag rates per label and group. Provide built-in LexiconClassifier and LLMClassifier implementations.

## Consequences

Users can plug in any classifier that conforms to the port. Evaluation code is reusable and
results are comparable across classifiers. Built-in classifiers are included but users must supply terms or LLM.
