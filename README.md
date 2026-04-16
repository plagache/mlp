# mlp

Introduction to artificial neural networks, with the implementation of a multilayer perceptron.

## Install

```bash
uv venv --python 3.12 .venv && uv pip install -e .
```

## Run

```bash
uv run python examples/train_mlp.py
uv run python examples/inference_mlp.py
```

## ToDo

- split train and prediction program
    - seems to have a safetensor.numpy module
- [x] refacto loss function

my intuition is that giving 80% of the dataset in one pass of our model will make it learn so fast.
it does not need to reajust much its weight.
