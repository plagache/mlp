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

- [ ] shift to model in a json
- [ ] add the __call__ function
- [ ] can load weight differently in Inference ?
- [ ] think about the dataset split in training and inference
- [ ] make some link about the data to the actual images of a breast cancer using some http balise like in roryclearcam
- [ ] implement SGD
- [x] split train and prediction program
- [x] refacto loss function
- [x] add BCE to inference
- [x] add split dataset program #output data_train.csv and data_valid.csv
- [x] polars query to detect null or 0 value
- [x] refacto load dataset with new path
- [x] refacto with pathlib
- [x] refacto load_dataset with encoder
- [x] store the normalisation technique in a safetensors

my intuition is that giving 80% of the dataset in one pass of our model will make it learn so fast.
it does not need to reajust much its weight.

the Formula we are using is a contrastive method, this mean that we are pushing up the probability of benign when its benign but we are also pushing down the probability of malign when its benign
Actually its call a contrastive embedding, the embedding for Malign [1.0, 0.0] and benign [0.0, 1.0]
and for a Malign examples we would want to push the first column of our output to 1 and the 2nd to zero
```python
def log_loss(y, p):
    return -((y * (p).log() + (1 - y) * (1 - p).log()).MEAN())
```
