# Docs2Synth

Docs2Synth is a Python package aimed at helping you convertt, synthesise and train a retriever for your document datasets.

The workflow typically involves:

- Document processing
  - MinerU or Other OCR methods
- Agent based QA generation
  - QA Pair generation
    - For now, given content, generate a question and answer
    - Later we can extend to multiple page contexts in a single QA pair.
  - Two step verification
    - Meaningful verifier
    - Correctness checker
  - Human judgement
    - Human quickly annotate keep/discard
- Retriever training
  - Train LayoutLMv3 as a retriever
  - Extend to bert, etc
- RAG Path
  - Other out-of-box retriever strageties without training
- Framework pipeline
  - Combine all the above steps into a single pipeline and control flow based on parameters
  - Benchmarking
    - Retrieveral Hit@K Performance
    - End to end latency metrics

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

## Quick start (Conda)

```bash
# create and activate env (Python 3.10)
conda env create -f environment.yml
conda activate Docs2Synth

# verify installation
pytest
```

## License

[MIT](LICENSE)