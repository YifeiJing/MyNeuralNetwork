import numpy as np
from matplotlib import pyplot as plt
from trainer import *
from network import *

""" A simple dataloader and also a training script with some helper functions
"""
def dataManager(x, y, rate):
    datasize = x.shape[0]
    x_train, x_test = x[:int(datasize*rate)], x[int(datasize*rate):]
    y_train, y_test = y[:int(datasize*rate)], y[int(datasize*rate):]
    return (x_train, x_test, y_train, y_test)

def sqaureHelper(x):
    return x**2

def productHelper(x):
    res = x[:,0] * x[:,1]
    return res.reshape(x.shape[0],1)

featuresDic={'sin':np.sin, 'cos':np.cos, 'square':sqaureHelper, 'reciprocal':np.reciprocal, 'product':productHelper}
def addFeatures(inputs, features):
    dataSize = inputs.shape[0]
    metaData = inputs
    for f in features:
        newData = featuresDic[f](metaData)
        inputs = np.concatenate((inputs, newData), axis=1)
    return inputs

def spiralDataSetTest(visualize=False):
    # prepare data
    train_test_rate = 0.5
    x_buf, y_buf = [], []
    with open('two_spirals.dat') as f:
        for line in f.readlines():
            data_strs = line.split()
            x_buf.append(float(data_strs[0]))
            x_buf.append(float(data_strs[1]))
            y_buf.append(float(data_strs[2]))

    x, y = np.array(x_buf).reshape(int(len(x_buf)/2), 2), np.array(y_buf)

    # plot x v y
    if visualize:
        plt.title('Data visualization')
        plt.xlabel('x')
        plt.ylabel('y')
        focus_positive = x[y == 1]
        focus_negative = x[y == 0]
        plt.scatter(focus_positive[:,0], focus_positive[:,1])
        plt.scatter(focus_negative[:,0], focus_negative[:,1])
        plt.show()

    # add more features
    x = addFeatures(x, ['sin'])
    datasize = x.shape[0]
    x_train, x_test = x[:int(datasize*train_test_rate)], x[int(datasize*train_test_rate):]
    y_train, y_test = y[:int(datasize*train_test_rate)], y[int(datasize*train_test_rate):]

    # Establish neural network
    net = Network(x.shape[1], [4,4,2,1], activationFunction='sin')
    net.setActiviationFunction('tanh', 3)
    trainer = Trainer(net, 'pso', x_train, y_train, x_test, y_test, batch_size=20, verbose=True, epoche=300, optimizer_para={'alpha1':1.0, 'alpha2':1.0, 'omega':1.0})
    trainer.train()
    if visualize:
        trainer.visualize()
    print('Parameters of current network:')
    print(net.getAllParas())
    print('==================================')
    

spiralDataSetTest(visualize=False)