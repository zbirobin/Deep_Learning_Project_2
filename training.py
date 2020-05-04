from random import sample
from torch import empty

from functional import lossMSE, dlossMSE, lossMAE, dlossMAE


def one_hot_encode(target):
    ohe_target = empty(target.shape[0], 2)
    ohe_target[:, 0] = target == 0
    ohe_target[:, 1] = target == 1
    return ohe_target


def train_model_SGD(model, train_input, train_target, nb_epoch=25, mini_batch_size=1):
    nb_train_samples = train_input.shape[0]
    ohe_train_target = one_hot_encode(train_target)
    mse = LossMSE()
    eta = 1e-1 / nb_train_samples
    nb_batch_per_epoch = nb_train_samples // mini_batch_size
    losses = empty(nb_epoch, nb_batch_per_epoch)
    for epoch in range(nb_epoch):
        for batch in range(nb_batch_per_epoch):
            loss = 0
            indices = sample(range(nb_train_samples), mini_batch_size)
            for i in indices:
                # Run forward pass
                output = model.forward(train_input[i])
                # Compute loss
                loss += mse.compute_loss(output, ohe_train_target[i])
                # Run back propagation
                model.backward(mse.gradient())

            loss /= nb_batch_per_epoch
            losses[epoch][batch] = loss
            model.update_param(eta)
            model.zero_grad()

        mean_loss = losses.mean(1)[epoch]
        print(f"Epoch: {epoch}, Mean loss: {mean_loss}")


def accuracy(model, input, target):
    output = empty(input.shape[0], 2)
    for n in range(output.shape[0]):
        output[n] = model.forward(input[n])
    return (output.argmax(1) == target).sum() / float(output.shape[0])


# Losses classes

class LossMSE:

    def __init__(self):
        self.output = None
        self.target = None
        self.name = "MSE Loss"

    def compute_loss(self, output, target):
        self.output = output
        self.target = target
        return lossMSE(self.output, self.target)

    def gradient(self):
        if self.target is None or self.output is None:
            raise Exception("The loss must be computed first for " + self.name)

        return dlossMSE(self.output, self.target)


class LossMAE:

    def __init__(self):
        self.output = None
        self.target = None
        self.name = "MAE Loss"

    def compute_loss(self, output, target):
        self.output = output
        self.target = target
        return lossMAE(self.output, self.target)

    def gradient(self):
        if self.target is None or self.output is None:
            raise Exception("The loss must be computed first for " + self.name)

        return dlossMAE(self.output, self.target)
