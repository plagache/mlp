# mlp

Introduction to artificial neural networks, with the implementation of a multilayer perceptron.
If you want to learn more about the way [deep learning framework](dlf/README.md) work.

## Install

```bash
uv venv --python 3.12 .venv && uv pip install -e .
```

## Training

```bash
uv run python examples/train_mlp.py
```
this will also save plot about the training data.

## Inference

```bash
uv run python examples/inference_mlp.py
```
Load weight from training and perform inference on unseen data.

there are other [commands](docs/commands.md:3) you can use, to display the directory and view images in your browser for examples.

## ToDo

- [x] split train and prediction program
- [x] refacto loss function
- [ ] implement SGD

my intuition is that giving 80% of the dataset in one pass of our model will make it learn so fast.
it does not need to reajust much its weight.
