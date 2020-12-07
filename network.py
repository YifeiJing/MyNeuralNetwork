from layer import *
from node import *
from activationFunctions import *
from lossFunctions import *
from gradientFunctions import *
import numpy as np

class Network(object):
    """
    Abstraction for a neural network
    """
    def __init__(self, inputSize, networkDimension, activationFunction='sigmoid', lossFunction='mean_squared_loss'):
        super().__init__()
        self._inputSize = inputSize
        self._depth = len(networkDimension)
        self._layerBuf = []
        self._activationFuncDic = {'identity':identity, 'step':step_function, 'sigmoid':sigmoid, 'relu':relu, 'softmax':softmax, 'sin':sin, 'tanh':tanh}
        self._lossFuncDic = {'mean_squared_loss':mean_squared_loss, 'cross_entropy_loss': cross_entropy_loss}
        self._lossFunc = self._lossFuncDic[lossFunction]
        self._ndim = networkDimension
        for i in range(self._depth):
            if i == 0:
                layer = Layer(networkDimension[i], inputSize, self._activationFuncDic[activationFunction])
            else:
                layer = Layer(networkDimension[i], networkDimension[i-1], self._activationFuncDic[activationFunction])
            self._layerBuf.append(layer)

    def propagate(self, inputs):
        if inputs.size % self._inputSize != 0:
            raise Exception("The size of inputs is not right: ", inputs)
        output = self._layerBuf[0].getOutput(inputs)
        for i in range(self._depth):
            if i == 0: continue
            output = self._layerBuf[i].getOutput(output)
        return output
    
    def getNetworkDimension(self):
        return self._ndim

    def getInputSize(self):
        return self._inputSize

    def getLayer(self, index):
        if index >= len(self._layerBuf):
            raise Exception ('Layer index out of range: ', index)
        return self._layerBuf[index]
    
    def setWeights(self, weights, index):
        layer = self.getLayer(index)
        layer.setWeights(np.array(weights))

    def getWeights(self, index):
        return self.getLayer(index).getWeights()

    def setBiases(self, biases, index):
        layer = self.getLayer(index)
        layer.setBias(np.array(biases))
    
    def getBiases(self, index):
        return self.getLayer(index).getBias()

    def getParas(self, index):
        """
        weight and bias
        """
        weights = self.getWeights(index)
        biases = self.getBiases(index)
        biases = biases.reshape(biases.size, 1)
        return np.concatenate((weights, biases), axis=1)
    
    def setParas(self, paras, index):
        weights = paras[..., :-1]
        biases = paras[..., -1]
        self.setWeights(weights, index)
        self.setBiases(biases.reshape(biases.size), index)
    
    def getLoss(self, x, y):
        y_hat = self.propagate(x)
        if y_hat.size != y.size:
            raise Exception ("The size of target value is not right: ", y)
        return self._lossFunc(y_hat, y)
    
    def getGradient(self, x, y, index):
        def lossWeightSetter(w):
            self.setParas(w, index)
            return self.getLoss(x, y)
        return numerical_grad(lossWeightSetter, self.getParas(index))
    
    def getAllParas(self):
        """ Return all parameters as a list
        """
        res = []
        for i in range(self._depth):
            res.append(self.getParas(i))
        return res
    
    def setAllParas(self, paras):
        """ Expecting paras as a list of numpy arrays with the length equal depth
        """
        if len(paras) != self._depth:
            raise Exception ('The lenght of new parameters is not right: ', paras)
        for i in range(self._depth):
            self.setParas(paras[i], i)
    
    def getGradients(self, x, y):
        res = []
        for i in range(self._depth):
            res.append(self.getGradient(x, y, i))
        return res

    def setActiviationFunction(self, func, index):
        if index >= self._depth:
            raise Exception ('The index is out of range: ', index)
        self._layerBuf[index].setActiviationFunction(self._activationFuncDic[func])
    
    def setActiviationFunctions(self, funcs):
        if len(funcs) != self._depth:
            raise Exception ('The length of the function list is not right: ', funcs)
        for i in range(self._depth):
            self.setActiviationFunction(funcs[i], i)
    
    def setLossFunction(self, lossFunction):
        self._lossFunc = self._lossFuncDic[lossFunction.lower()]
    

if __name__ == "__main__":
    net = Network(3, [4,4,4,1])
    weights = np.array([[[0.1 for i in range(3)] for j in range(4)],
                        [[0.2 for i in range(4)] for j in range(4)],
                        [[0.3 for i in range(4)] for j in range(4)],
                        [[0.4 for i in range(4)]]])
    print("weights buffer", weights)
    print('Current weights of the network: ')
    for i in range(4):
        print(net.getWeights(i))
    print('Set new weights')
    for i in range(4):
        print(net.setWeights(weights[i], i))
        print(net.getWeights(i))
    inputs = np.array([[j for i in range(3)] for j in range(10)])
    targets = np.array([i for i in range(10)])
    print('Test propagate:')
    print(net.propagate(inputs))
    print('Test loss:')
    print(net.getLoss(inputs, targets))
    print('Test gradient:')
    for i in range(4):
        print(i, 'th layer:')
        print(net.getGradient(inputs, targets, i))

    print('Test getparas:')
    print(net.getParas(0))
    