# mlp

Introduction to neural networks, with the implementation of a multilayer perceptron.


If you want to learn more about the way [deep learning framework](dlf/README.md) work.

## Install

```bash
rm -rf .venv && uv venv --python 3.12 .venv && uv pip install -e .
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


## Clean
```bash
rm -rf .venv
rm -rf data_*
rm -rf *.safetensors
rm -rf *.png
```

## ToDo

- [ ] refacto Inference

- [ ] Optimizer base class / proper GD / Proper SGD
    - [ ] explain the different optimization: Weight decay, Momentum: see if saving previous values is heavy in compute
- [ ] Encoder - Decoder base class ?
- [ ] Dataclass ? can transpose from json to dataclass directly
- [ ] normalisation should be perform only on training data
- [ ] add third program split data
- [ ] make some link about the data to the actual images of a breast cancer using some http balise like in roryclearcam
- [ ] implement SGD
- [ ] maybe reduce the .gitignore
    - [ ] explain momentum
    - [ ] explain weight decay
    - [ ] explain information collapse

- [ ] testing:
    - [ ] shape/type missmatch
- [ ] Explain contrastive method in log_loss maybe renamed to BCE / penalise wrong answer, reward good one
- [ ] Explain Topo_sort with a graphviz would be perfect not sure its easaly done tho

- [x] everything is a function, train should be a loop
    - [x] all training variable in one place / no more changing directly in the function call
    - [x] train function for each epoch what do we do ? whats the return ?
    - [x] extract evaluate function -> return tuple[float, float]
    - [x] metrics dictionary that contain the 4 list;
    - [x] finally a fit function that takes epoch and data and train and evaluate epochs
- [x] look into the save_model comment in train_mlp.py
- [x] can load weight differently in Inference ?
    - [x] we are protecting against missmatch
    - [x] problem with inference, when shape missmatch
- [x] think about the dataset split in training and inference
    - [x] we fixed with a check before split
- [x] change get_parameters() to a yield and yield from
    - [x] don't see the interest anymore, simple is better to explain

the Formula we are using is a contrastive method, this mean that we are pushing up the probability of benign when its benign but we are also pushing down the probability of malign when its benign
Actually its call a contrastive embedding, the embedding for Malign [1.0, 0.0] and benign [0.0, 1.0]
and for a Malign examples we would want to push the first column of our output to 1 and the 2nd to zero
```python
def log_loss(y, p):
    return -((y * (p).log() + (1 - y) * (1 - p).log()).MEAN())
```
