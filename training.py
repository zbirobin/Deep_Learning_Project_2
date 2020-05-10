from torch import empty

from functional import lossMSE, dlossMSE, lossMAE, dlossMAE


def one_hot_encode(target):
    ohe_target = empty(target.shape[0], 2)
    ohe_target[:, 0] = target == 0
    ohe_target[:, 1] = target == 1
    return ohe_target


def train_model_SGD(model, train_input, train_target, nb_epoch=25, mini_batch_size=10):
    nb_train_samples = train_input.shape[0]
    ohe_train_target = one_hot_encode(train_target)
    mse = LossMSE()
    eta = 1e-3
    nb_batch_per_epoch = nb_train_samples // mini_batch_size
    for epoch in range(nb_epoch):
        loss = 0
        for b in range(0, nb_train_samples, mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            # Compute the loss
            loss += mse.compute_loss(output, ohe_train_target.narrow(0, b, mini_batch_size))
            # Run back propagation
            model.backward(mse.gradient())

            model.update_param(eta)

        mean_loss = loss/nb_batch_per_epoch
        print(f"Epoch: {epoch}, Mean loss: {mean_loss}")


def accuracy(model, input, target):
    output = model.forward(input)
    return (output.argmax(1) == target).sum() / float(output.shape[0])


# Losses classes

class LossMSE:

    def __init__(self):
        self.output = None
        self.target = None
        self.name = "MSE Loss"

    def compute_loss(self, output, target):
        self.output = output.clone()
        self.target = target.clone()
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
        self.output = output.clone()
        self.target = target.clone()
        return lossMAE(self.output, self.target)

    def gradient(self):
        if self.target is None or self.output is None:
            raise Exception("The loss must be computed first for " + self.name)

        return dlossMAE(self.output, self.target)
