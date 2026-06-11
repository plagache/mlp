# mlp

Introduction to neural networks, with the implementation of a multilayer perceptron.


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

Other [commands](docs/commands.md) you can use, [run a python server of the directory](docs/commands.md#L19) and view images in your browser for examples.

## ToDo

- [x] split train and prediction program
- [x] refacto loss function
- [x] add BCE to inference
- [x] add split dataset program #output data_train.csv and data_valid.csv
- [ ] refacto load dataset with new path
- [ ] polars query to detect null or 0 value
- [ ] make some link about the data to the actual images of a breast cancer
- [ ] implement SGD

my intuition is that giving 80% of the dataset in one pass of our model will make it learn so fast.
it does not need to reajust much its weight.
