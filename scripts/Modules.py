from abc import ABC, abstractmethod
from torch import empty

from scripts.functional import relu, drelu, tanh, dtanh, sigmoid, dsigmoid


class Module(ABC):
    """Abstract class representing a module of a Neural Network.
    """

    @abstractmethod
    def forward(self, *input):
        pass

    @abstractmethod
    def backward(self, *gradwrtoutput):
        pass

    @abstractmethod
    def param(self):
        pass

    @abstractmethod
    def update_param(self, eta):
        pass


class Sequential(Module):
    """Class representing a Neural Network which is a sequence of the modules given in the constructor.
    """

    def __init__(self, *modules):
        """Builds a Neural Network using the modules given.

        :param modules: modules of the Neural Network
        """
        self.modules = list(modules)
        self.name = " => ".join([m.name for m in modules])

    def forward(self, input):
        """Computes the forward pass of the Neural Network for the given input of N samples.

        :param input: Tensor (N x input_dim of the first module)
        :return: Tensor output (N x output_dim of the last module)
        """
        output = input.clone()
        for module in self.modules:
            output = module.forward(output)
        return output

    def backward(self, gradwrtoutput):
        """Computes the backward pass of the Neural Network for the given gradient w.r.t the output.

        :param gradwrtoutput: (N x output_dim of the last module)
        :return: Tensor the gradients w.r.t the input (N x input_dim of the first module)
        """
        grad = gradwrtoutput.clone()
        for module in self.modules[::-1]:
            grad = module.backward(grad)
        return grad

    def param(self):
        """Returns the parameters of the modules of the Neural Network.

        :return: the parameters of the modules in an array
        """
        params = []
        for module in self.modules:
            params.append(module.param())
        return params

    def update_param(self, eta):
        """Updates the parameters of the modules using the gradient descent rule with learning rate eta.

        :param eta: learning rate
        """
        for module in self.modules:
            module.update_param(eta)


class Linear(Module):
    """Class representing a linear fully connected layer of a Neural Network.
    """

    def __init__(self, input_dim, output_dim):
        """Builds a linear fully connected Layer of a Neural Network of input dimension input_dim and output dimension
        output_dim.

        :param input_dim: integer
        :param output_dim: integer
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input = None

        self.weights, self.bias = self.__init_parameters()
        self.weights_grad = None
        self.bias_grad = None
        self.name = "Linear({}x{})".format(self.input_dim, self.output_dim)

    def __init_parameters(self):
        """Initializes the weights and the bias of the layer using normal distributions.
        :return: Tensors weights matrix (input_dim x output_dim), bias (output_dim)
        """
        return empty(self.input_dim, self.output_dim).normal_(), empty(self.output_dim).normal_()

    def forward(self, input):
        """Computes the forward pass of the layer, i.e., XW + B (broadcasting on b).

        :param input: Tensor (N x input_dim)
        :return: Tensor output (N x output_dim)
        """
        self.input = input
        return input.matmul(self.weights).add(self.bias)

    def backward(self, gradwrtoutput):
        """Computes the backward pass of the layer, i.e., the gradients w.r.t. the input using the gradients w.r.t. the
        output given.

        :param gradwrtoutput: Tensor (N x output_dim)
        :return: Tensor gradients (N x input_dim)
        """
        if self.input is not None:
            self.bias_grad = gradwrtoutput.sum(0)  # Gradient of sum is the sum of gradient
            weights_grad = self.input.t().matmul(gradwrtoutput)
            self.weights_grad = weights_grad
            return gradwrtoutput.matmul(self.weights.t())
        else:
            raise Exception("Forward must be called before backward for " + self.name)

    def param(self):
        """Returns the parameters of the layer.

        :return: [(Weights matrix (input_dim x output_dim), Gradients weights matrix (input_dim x output_dim)), 
                    (Bias vector (output_dim), Gradients bias vector (output_dim))]
        """
        return [(self.weights, self.weights_grad), (self.bias, self.bias_grad)]

    def update_param(self, eta):
        """Updates the parameters of the layer using the gradient descent rule with learning rate eta.

        :param eta: learning rate
        """
        self.weights -= eta * self.weights_grad
        self.bias -= eta * self.bias_grad


# Activations functions modules

class ReLU(Module):
    """Class representing a ReLU module of a Neural Network.
    """

    def __init__(self):
        """Builds the ReLU module.
        """
        self.input = None
        self.name = "ReLU"

    def forward(self, input):
        """Computes the forward pass of the ReLU module for the given input.

        :param input: Tensor
        :return: Tensor with shape as input
        """
        self.input = input.clone()
        return relu(self.input)

    def backward(self, gradwrtoutput):
        """Computes the backward pass of the ReLU module, i.e., the gradients w.r.t. the input using the gradients
        w.r.t. the output given.

        :param gradwrtoutput: Tensor with shape as input
        :return: Tensor with shape as input
        """
        if self.input is not None:
            return gradwrtoutput.mul(drelu(self.input))
        else:
            raise Exception("Forward must be called before backward for " + self.name)

    def param(self):
        """Returns None as ReLU has no parameter.

        :return: None
        """
        return None

    def update_param(self, eta):
        """Does nothing as ReLU has no parameter."""
        pass


class Tanh(Module):
    """Class representing a Tanh module of a Neural Network.
    """

    def __init__(self):
        """Builds the Tanh module.
        """
        self.input = None
        self.name = "Tanh"

    def forward(self, input):
        """Computes the forward pass of the Tanh module for the given input.

        :param input: Tensor
        :return: Tensor with shape as input
        """
        self.input = input.clone()
        return tanh(self.input)

    def backward(self, gradwrtoutput):
        """Computes the backward pass of the Tanh module, i.e., the gradients w.r.t. the input using the gradients
        w.r.t. the output given.

        :param gradwrtoutput: Tensor with shape as input
        :return: Tensor with shape as input
        """
        if self.input is not None:
            return gradwrtoutput.mul(dtanh(self.input))
        else:
            raise Exception("Forward must be called before backward for " + self.name)

    def param(self):
        """Returns None as Tanh has no parameter.

        :return: None
        """
        return None

    def update_param(self, eta):
        """Does nothing as Tanh has no parameter."""
        pass


class Sigmoid(Module):
    """Class representing a sigmoid module of a Neural Network.
    """

    def __init__(self):
        """Builds the Sigmoid module.
        """
        self.input = None
        self.name = "Sigmoid"

    def forward(self, input):
        """Computes the forward pass of the Sigmoid module for the given input.

        :param input: Tensor
        :return: Tensor with shape as input
        """
        self.input = input.clone()
        return sigmoid(self.input)

    def backward(self, gradwrtoutput):
        """Computes the backward pass of the Sigmoid module, i.e., the gradients w.r.t. the input using the gradients
        w.r.t. the output given.

        :param gradwrtoutput: Tensor with shape as input
        :return: Tensor with shape as input
        """
        if self.input is not None:
            return gradwrtoutput.mul(dsigmoid(self.input))
        else:
            raise Exception("Forward must be called before backward for " + self.name)

    def param(self):
        """Returns None as Sigmoid has no parameter.

        :return: None
        """
        print(self.name + " has no parameters")
        return None

    def update_param(self, eta):
        """Does nothing as Sigmoid has no parameter."""
        pass
