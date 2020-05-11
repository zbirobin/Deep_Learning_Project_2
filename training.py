from torch import empty

from functional import lossMSE, dlossMSE, lossMAE, dlossMAE


def one_hot_encode(target):
    """
    Encode the targets in 1-hot encoding
    :param target: target array of 1s and 0s
    :return: targets encoded in 1-hot
    """
    ohe_target = empty(target.shape[0], 2)
    ohe_target[:, 0] = target == 0
    ohe_target[:, 1] = target == 1
    return ohe_target


def train_model_SGD(model, train_input, train_target, nb_epoch=25, mini_batch_size=10, eta=1e-3):
    """
    Uses Stochastic gradient descent algorithm to train the model
    :param model: the model to train
    :param train_input: the input data for the model
    :param train_target: the output data for the model
    :param nb_epoch: the number of training epochs
    :param mini_batch_size: the batch size to use for the training
    :param eta: the learning rate of the stochastic gradient descent algorithm used
    :return: None
    """
    nb_train_samples = train_input.shape[0]
    ohe_train_target = one_hot_encode(train_target)
    mse = LossMSE()

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

        mean_loss = loss / nb_batch_per_epoch
        print(f"Epoch: {epoch}, Mean loss: {mean_loss}")


def accuracy(model, input, target):
    """
    Computes the accuracy metric for the model using the a set of input/target pair
    :param model: the model to test
    :param input: the input data
    :param target: the output data
    :return: the accuracy of the model
    """
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
