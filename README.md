# mlp

Introduction to artificial neural networks, with the implementation of a multilayer perceptron.

## Install

```bash
uv venv --python 3.12 .venv && uv pip install -e .
```

## Run

```bash
uv run python examples/mlp.py
```

## ToDo

my intuition is that giving 80% of the dataset in one pass of our model will make it learn so fast.
it does not need to reajust much its weight.
