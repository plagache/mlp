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

- look into the utility of safetensor for numpy arrays, the model, and the tensor class
- redo the loss function
    - taking into account that we may want to change the output data shape after the SOFTMAX

my intuition is that giving 80% of the dataset in one pass of our model will make it learn so fast.
it does not need to reajust much its weight.
