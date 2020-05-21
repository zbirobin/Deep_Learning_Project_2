# Helpers methods


## Activations functions and their derivatives

### Hyperbolic tangent
def tanh(x):
    return x.tanh()

def dtanh(x):
    return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

### Sigmoid
def sigmoid(x):
    return 1 / (1 + (-x).exp())

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

### ReLU
def relu(x):
    return x * (x > 0)

def drelu(x):
    return 1. * (x > 0)


## Losses and their derivatives

### Mean Square Error
def lossMSE(v, t):
    return (v - t).pow(2).mean()

def dlossMSE(v, t):
    return 2 * (v - t)

### Mean Absolute Error
def lossMAE(v, t):
    return (v - t).abs().mean()

def dlossMAE(v, t):
    return (v - t).sign()
