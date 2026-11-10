# toxicity-eval

[![PyPI version](https://img.shields.io/pypi/v/trysac-toxicity-eval)](https://pypi.org/project/trysac-toxicity-eval/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-toxicity-eval)](LICENSE)
[![CI](https://github.com/Trysac/u233-toxicity-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u233-toxicity-eval/actions/workflows/ci.yml)

Evaluate toxicity of model outputs with interchangeable classifiers: a classifier port, a lexicon baseline, an LLM classifier, thresholds and per-group flag rates

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-toxicity-eval
# or
uv add trysac-toxicity-eval
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import toxicity_eval

# normalize() trims surrounding whitespace and lower-cases the text.
print(toxicity_eval.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    toxicity_eval.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
