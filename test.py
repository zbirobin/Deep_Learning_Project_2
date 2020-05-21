# Imports from classes containing modules, generator and training function
from generate_data import generate_data
from Modules import Sequential, Linear, ReLU, Tanh, Sigmoid
from training import train_model_SGD, accuracy

# Generate normalize training and testing dataset of size N = 1000
N = 1000 # Number of samples in each dataset
train_input, train_target, test_input, test_target = generate_data(N, normalize=True)

# Builds a network with three hidden layer of size 25 using sequential Linear modules and activation functions
l1 = Linear(2,25)
a1 = ReLU()
l2 = Linear(25, 25)
a2 = Tanh()
l3 = Linear(25, 25)
a3 = Sigmoid()
l4 = Linear(25, 2)
model = Sequential(l1, a1, l2, a2, l3, a3, l4)

# Train the model using MSE and logging the losses (verbose = True)
losses = train_model_SGD(model, train_input, train_target, nb_epoch=300, learning_rate=1e-1, mini_batch_size=100, verbose=True)

# Compute and print the final train and test accuracy
print("\nFinal train accuracy : " + str(accuracy(model, train_input, train_target).item()*100)[:5] + "%")
print("\nFinal test accuracy : " + str(accuracy(model, test_input, test_target).item()*100)[:5] + "%")

