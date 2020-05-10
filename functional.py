# Activations functions and its derivatives

def tanh(x):
    return x.tanh()


def dtanh(x):
    return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)


def sigmoid(x):
    return 1 / (1 + (-x).exp())


def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return x * (x > 0)


def drelu(x):
    return 1. * (x > 0)


# Losses and its derivatives

def lossMSE(v, t):
    return (v - t).pow(2).sum()


def dlossMSE(v, t):
    return 2 * (v - t)


def lossMAE(v, t):
    return (v - t).abs().sum()


def dlossMAE(v, t):
    return (v - t).sign()
