from abc import ABC, abstractmethod
from torch import empty

from functional import relu, drelu, tanh, dtanh, sigmoid, dsigmoid, lossMSE, dlossMSE


class Module(ABC):
    
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
    
    def __init__(self, *modules):
        self.modules = list(modules)
        self.name = " => ".join([m.name for m in modules])
        
    def forward(self, input):
        output = input.clone()
        for module in self.modules:
            output = module.forward(output)
        return output
    
    def backward(self, gradwrtoutput):
        grad = gradwrtoutput.clone()
        for module in self.modules[::-1]:
            grad = module.backward(grad)
        return grad
    
    def param(self):
        params = []
        for module in self.modules:
            params.append(module.param())
        return params
    
    def update_param(self, eta):
        for module in self.modules:
            module.update_param(eta)
            
    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()


class Linear(Module):
    
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input = None
        
        self.weights, self.bias = self.__init_parameters()
        self.weights_grad = 0
        self.bias_grad = 0
        self.name = "Linear({}x{})".format(self.input_dim, self.output_dim)
        
    def __init_parameters(self):
        """Initialize the weights and the bias of the layer.  
        :return: weights matrix, bias
        """
        return empty(self.output_dim, self.input_dim).normal_(), empty(self.output_dim)
        
    def forward(self, input):
        self.input = input
        return self.weights.mv(input).add(self.bias)
        
    def backward(self, gradwrtoutput):
        if self.input is not None:
            self.bias_grad += gradwrtoutput
            self.weights_grad += gradwrtoutput.view(-1, 1).mm(self.input.view(1, -1))
            return self.weights.t().mv(gradwrtoutput)
        else:
            raise Exception("Forward must be called before backward for " + self.name)
    
    def param(self):
        return [(self.weights, self.weights_grad), (self.bias, self.bias_grad)]
    
    def update_param(self, eta):
        self.weights -= eta * self.weights_grad
        self.bias -= eta * self.bias_grad
    
    def zero_grad(self):
        self.weights_grad = 0
        self.bias_grad = 0
        

# Activations functions modules

class ReLU(Module):
    
    def __init__(self):
        self.input = None
        self.name = "ReLU"
    
    def forward(self, input):
        self.input = input
        return relu(self.input)
    
    def backward(self, gradwrtoutput):
        if self.input is not None:
            return gradwrtoutput.mul(drelu(self.input))
        else:
            raise Exception("Forward must be called before backward for " + self.name)
    
    def param(self):
        print(self.name + " has no parameters")
        return None
    
    def update_param(self, eta):
        pass
    
    def zero_grad(self):
        pass


class Tanh(Module):
    
    def __init__(self):
        self.input = None
        self.name = "Tanh"
    
    def forward(self, input):
        self.input = input
        return tanh(self.input)
    
    def backward(self, gradwrtoutput):
        if self.input is not None:
            return gradwrtoutput.mul(dtanh(self.input))
        else:
            raise Exception("Forward must be called before backward for " + self.name)
    
    def param(self):
        print(self.name + " has no parameters")
        return None
    
    def update_param(self, eta):
        pass    
    
    def zero_grad(self):
        pass


class Sigmoid(Module):
    
    def __init__(self):
        self.input = None
        self.name = "Sigmoid"
    
    def forward(self, input):
        self.input = input
        return sigmoid(self.input)
    
    def backward(self, gradwrtoutput):
        if self.input is not None:
            return gradwrtoutput.mul(dsigmoid(self.input))
        else:
            raise Exception("Forward must be called before backward for " + self.name)
    
    def param(self):
        print(self.name + " has no parameters")
        return None
    
    def update_param(self, eta):
        pass
    
    def zero_grad(self):
        pass


# MSE Loss, not a module
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
        if not self.target or not self.output:
            raise Exception("The loss must be computed first for " + self.name)
        
        return dlossMSE(self.output, self.target)
