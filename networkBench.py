import numpy as np
from matplotlib import pyplot as plt
from trainer import *
from network import *
from dataloader import dataManager

#Prepare data
# A linear dataset
x = np.arange(-100.0, 100.0)
y = 2 * x

(x_train, x_test, y_train, y_test) = dataManager(x.reshape(x.size,1), y.reshape(y.size, 1), 0.7)

# Establish network
network = Network(1, [1], 'identity')
# network.setWeights(np.array([[2.0]]), 0)
v = network.propagate(x_train[:5])
loss = network.getLoss(x_train[:5], y_train[:5])
trainer = Trainer(network, 'sgd', x_train, y_train, x_test, y_test, verbose=True)
trainer.train()