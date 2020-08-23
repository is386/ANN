# Artificial Neuron

This is an artificial neural network built from scratch. It uses a sigmoid activation function on the input, hidden, and output layers. The gradient is computed by using the log likelihood function. It is used on the Yalefaces dataset and categorizes each face into 1 of 14 different classes.

## Usage

`python3 ann.py`

You will be prompted to give the number of layers, and the number of nodes for each layer.

## Dependencies

- `python 3.8+`

### Python Dependencies

- `pillow`
- `numpy`
- `matplotlib`
- `seaborn`
- `pandas`

## Hyper Parameters

- Learning Rate: `0.001`

- Termination Criteria: `1000 iterations`

- L2 Regularization Term: `0.001`

- Bias: `1`

## Results

The following network was used for this test:

- Input Layer: `64 nodes`
- Hidden Layer: `32 nodes`
- Output Layer: `16 nodes`

### Accuracy:

Testing Accuracy: `97.62%`

### Average Log Likelihood:

![](https://github.com/is386/ANN/blob/master/log.png?raw=true)

### Confusion Matrix:

![](https://github.com/is386/ANN/blob/master/confuse.png?raw=true)
