import node
import numpy as np

class Layer(object):
    """
    Abstraction for a network layer
    """
    def __init__(self, length, inputSize, activationFunction):
        super().__init__()
        self._length = length
        self._inputSize = inputSize
        self._activationFunction = activationFunction
        self._nodeBuf = [node.Node(inputSize) for i in range(length)]
    
    def setWeights(self, weights):
        if weights.size != self._length * self._inputSize:
            raise Exception ("The size of weights is not right!", weights)
        for i in range(self._length):
            self._nodeBuf[i].setWeight(weights[i])
    
    def getWeights(self):
        return np.array([n.getWeight() for n in self._nodeBuf])

    def getLength(self):
        return self._length
    
    def getBias(self):
        return np.array([n.getBias() for n in self._nodeBuf])
    
    def setBias(self, biases):
        if biases.size != self._length:
            raise Exception ('The size of the bias list is not right: ', biases)
        for i in range(self._length):
            self._nodeBuf[i].setBias(biases[i])
    
    def getInputSize(self):
        return self._inputSize
    
    def removeNode(self, index):
        if index >= self._length:
            raise Exception ("The index is out of range: ", index)
        self._length -= 1
        self._nodeBuf.remove(self._nodeBuf[index])
    
    def setActiviationFunction(self, func):
        self._activationFunction = func
    
    def getActiviationFuntion(self):
        return self._activationFunction

    def getOutput(self, inputs):
        if inputs.size % self._inputSize != 0:
            raise Exception ("The size of inputs is not right!", inputs)
        sampleNumber = inputs.size // self._inputSize
        weightsBuf = np.array([n.getWeight() for n in self._nodeBuf])
        biasBuf = np.array([n.getBias() for n in self._nodeBuf])
        return self._activationFunction(np.dot(inputs.reshape(sampleNumber, self._inputSize), weightsBuf.T) + biasBuf)
    

if __name__ == "__main__":
    layer = Layer(2,3,lambda x:x)
    inputs = np.array([[1,2,3], [2,3,4], [5,6,7]])
    newWeights = np.array([[1.0,1.0,1.0], [0.1,0.1,0.1]])
    print(layer)
    print(inputs)
    print(newWeights)
    print(layer.getOutput(inputs))
    print('Change weights')
    layer.setWeights(newWeights)
    print(layer.getOutput(inputs))