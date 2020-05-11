from torch import empty

from functional import lossMSE, dlossMSE, lossMAE, dlossMAE


def one_hot_encode(target):
    """One hot encodes the target set.

    :param target: The target set to one hot encode
    :return: The target set one hot encoded
    """
    ohe_target = empty(target.shape[0], 2)
    ohe_target[:, 0] = target == 0
    ohe_target[:, 1] = target == 1
    return ohe_target


def train_model_SGD(model, train_input, train_target, nb_epoch=25, mini_batch_size=10, loss_type="MSE",
                    learning_rate=1e-3, learning_rate_type="constant", verbose=False):
    """Trains a model using Stochastic Gradient Descent (SGD) with the parameters given.
    Returns the losses of each epoch.

    :param model: Model to train of type Module
    :param train_input: Training input set
    :param train_target: Training target set
    :param nb_epoch: Number of epoch
    :param mini_batch_size: Size of the mini batch
    :param loss_type: Type of the loss, either "MSE" or "MAE"
                        for respectively Mean Squared Error and Mean Absolute Error
    :param learning_rate: Learning rate of the gradient descent
    :param learning_rate_type: Type of the learning rate, either "constant" or "decay".
                        "decay" is learning_rate/(current epoch * number of training samples)
    :param verbose: Print the losses and the accuracies if True
    :return: The list of all the losses of each epoch
    """

    nb_train_samples = train_input.shape[0]
    ohe_train_target = one_hot_encode(train_target)

    # Verify the loss type
    if loss_type == "MSE":
        loss_function = LossMSE()
    elif loss_type == "MAE":
        loss_function = LossMAE()
    else:
        raise ValueError(str(loss_type) + " is an unknown loss type.")

    # Verify the learning rate type
    if learning_rate_type in ["constant", "decay"]:
        lr = learning_rate
    else:
        raise ValueError(str(learning_rate_type) + " is an unknown learning rate type.")

    losses = []
    for epoch in range(nb_epoch):
        loss = 0
        for batch in range(0, nb_train_samples, mini_batch_size):
            # Run forward pass
            output = model.forward(train_input.narrow(0, batch, mini_batch_size))
            # Compute the loss
            loss += loss_function.compute_loss(output, ohe_train_target.narrow(0, batch, mini_batch_size))
            # Run back propagation
            model.backward(loss_function.gradient())
            # Update the parameters
            model.update_param(lr)

        # Update the learning rate
        if learning_rate_type == "decay":
            lr /= epoch * nb_train_samples

        # Print the loss and the accuracy
        if verbose and epoch % 5 == 0:
            acc = accuracy(model, train_input, train_target)
            print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {acc:.3f}")

        losses.append(loss)

    return losses


def accuracy(model, input, target):
    """Compute the accuracy of the predictions of the model on the input set and the target set.

    :param model: Model used to compute the accuracy
    :param input: Input set
    :param target: Target set
    :return: The accuracy
    """
    output = model.forward(input)
    return (output.argmax(1) == target).sum() / float(output.shape[0])


# Losses classes

class LossMSE:
    """Class representing a Mean Squared Loss."""

    def __init__(self):
        """Builds the Mean Squared Loss instance.
        """
        self.output = None
        self.target = None
        self.name = "MSE Loss"

    def compute_loss(self, output, target):
        """Computes the Mean Squared Loss w.r.t the output and target.

        :param output: Tensor of output values
        :param target: Tensor of target values
        :return: MSE loss
        """
        self.output = output.clone()
        self.target = target.clone()
        return lossMSE(self.output, self.target)

    def gradient(self):
        """Computes the gradient of the Mean Square Loss w.r.t. to the output and target used when
        compute_loss was called, then compute_loss must be called before gradient.

        :return: The gradient
        """
        if self.target is None or self.output is None:
            raise Exception("The loss must be computed first for " + self.name)

        return dlossMSE(self.output, self.target)


class LossMAE:
    """Class representing a Mean Absolute Loss."""

    def __init__(self):
        """Builds the Mean Absolute Loss instance.
        """
        self.output = None
        self.target = None
        self.name = "MAE Loss"

    def compute_loss(self, output, target):
        """Computes the Mean Absolute Loss w.r.t the output and target.

        :param output: Tensor of output values
        :param target: Tensor of target values
        :return: MSE loss
        """
        self.output = output.clone()
        self.target = target.clone()
        return lossMAE(self.output, self.target)

    def gradient(self):
        """Computes the gradient of the Mean Absolute Loss w.r.t. to the output and target used when
        compute_loss was called, therefore compute_loss must be called before gradient.

        :return: The gradient
        """
        if self.target is None or self.output is None:
            raise Exception("The loss must be computed first for " + self.name)

        return dlossMAE(self.output, self.target)
