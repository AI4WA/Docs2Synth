# Docs2Synth

Docs2Synth is a Python package aimed at helping you synthesise and transform document data. It now includes tooling for generating question-answer pairs and training retrievers over your document corpus.

## Features (planned)

* Generate QA pairs from document sets.
* Train and evaluate dense/embedding-based retrievers.
* Clean, well-tested Python APIs for working with document sets.
* Command-line interface for common synthesis tasks.
* Extensible plugin system.

## Installation

```bash
pip install git+https://github.com/AI4WA/Docs2Synth.git
```

or, once released on PyPI:

```bash
pip install docs2synth
```

## Development setup

```bash
# clone repository
$ git clone https://github.com/AI4WA/Docs2Synth.git
$ cd Docs2Synth

# create virtual environment (Python ≥3.8)
$ python -m venv .venv && source .venv/bin/activate

# install editable package with test dependencies
$ pip install -e ".[dev]"
```

Run tests with:

```bash
pytest
```

## License

[MIT](LICENSE) 