import math
from torch import empty

def generate_quarter_disc_set(N):
    """Generates a set of N sampled uniformly in [0, 1]^2, each with a
       label 0 if outside the disk of radius 1/√2π and 1 inside. 
       
       Return 2-tuple: input, target
    """
    input = empty(N, 2).uniform_(0, 1)
    target = input.pow(2).sum(1).sub(1 / math.sqrt(2 * math.pi)).sign().add(1).div(2).long()
    return input, target

def generate_data(N, normalize=False):
    """Generates a training and a test set of N sampled uniformly in [0, 1]^2, each with a
       label 0 if outside the disk of radius 1/√2π and 1 inside. 
       
       Return 4-tuple: train_input, train_target, test_input, test_target
    """
    train_input, train_target = generate_quarter_disc_set(N)
    test_input, test_target = generate_quarter_disc_set(N)
    if normalize:
        mean, std = train_input.mean(), train_input.std()
        train_input.sub_(mean).div_(std)
        test_input.sub_(mean).div_(std)
    
    
    return train_input, train_target, test_input, test_target