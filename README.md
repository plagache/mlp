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

## Other commands

```bash
uv python list
uv pip list
du -sh .venv
watch -n0.1 nvidia-smi
watch -n0.1 rocm-smi
uv run python -m http.server 3635 --bind 0.0.0.0
http://machine_ip:3635
```

## ToDo

my intuition is that giving 80% of the dataset in one pass of our model will make it learn so fast.
it does not need to reajust much its weight.
