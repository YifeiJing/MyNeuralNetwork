import random
import numpy as np
class Node(object):
    """
    Abstraction for a node in the network
    """
    def __init__(self, inputSize):
        super().__init__()
        self._inputSize = inputSize
        self._weight = np.random.randn(inputSize)
        self._bias = .0
        
    def getWeight(self):
        return self._weight.reshape(self._inputSize)
    
    def setWeight(self, weights):
        if weights.size != self._inputSize:
            raise Exception ("The size of the weights is not right", weights)
        self._weight = np.array(weights, copy = True)
    
    def getBias(self):
        return self._bias

    def setBias(self, bias):
        self._bias = bias
    
    def output(self, inputs):
        if inputs.size != self._inputSize:
            raise Exception ("The size of inputs is not right", inputs)
        return np.dot(inputs.reshape(1,self._inputSize), self._weight.reshape(self._inputSize,1))[0,0] + self._bias

if __name__ == '__main__':
    a = Node(5)
    print('Test for weight getter:')
    print('The weights of a:', a.getWeight())
    print('Test for weight setter:')
    newWeights = np.arange(5)
    a.setWeight(newWeights)
    print('New weights to set:')
    print(newWeights)
    print('The weights of a:', a.getWeight())
    print('The bias of a:', a.getBias())
    inputBuf = np.arange(10).reshape(2,5)
    print('Test for output:')
    print('The input buffer:')
    print(inputBuf)
    print('Output of a:')
    for e in inputBuf:
        print(a.output(e))
    print('All tests passed')
