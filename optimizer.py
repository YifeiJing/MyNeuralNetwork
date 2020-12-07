import numpy as np
import sys
from network import *

class SGD:
    def __init__(self, network, lr=0.01):
        super().__init__()
        self._network = network
        self._lr = lr
    
    def update(self, x, y):
        # """ Expecting paras as a list of layer parameters
        # """
        # if len(paras) != len(grads):
        #     raise Exception ('The size of paras is not equale to grads: ', len(paras), len(grads))
        # for i in range(len(paras)):
        #     paras[i] -= self.lr * grads[i]
        """ Update network parameters based on Stochastic Gradient Descent
        """
        grads = self._network.getGradients(x, y)
        paras = self._network.getAllParas()
        for i in range(len(paras)):
            paras[i] -= self._lr * grads[i]
        self._network.setAllParas(paras)

class Particle:
    """ Particle of swarm
    """
    def __init__(self, nndim, inputSize, omega, alpha1, alpha2, limit):
        self._nndim = nndim
        self._inputSize = inputSize
        self._omega = omega
        self._alpha1 = alpha1
        self._alpha2 = alpha2
        self._limit = limit
        self.x = []
        self.v = []
        self.p = []
        self.pf = sys.maxsize

        # establish x, v, p arrays
        tmp_x = (np.random.random((self._nndim[0], self._inputSize + 1)) - 0.5) * self._limit * 2.0
        self.x.append(tmp_x)
        self.p.append(tmp_x)
        tmp_v = (np.random.random((self._nndim[0], self._inputSize + 1)) - 0.5) * 2.0
        self.v.append(tmp_v)
        for i in range(len(self._nndim)):
            if i == 0:
                continue
            tmp_x = (np.random.random((self._nndim[i], self._nndim[i-1] + 1)) - 0.5) * self._limit * 2.0
            self.x.append(tmp_x)
            self.p.append(tmp_x)
            tmp_v = (np.random.random((self._nndim[i], self._nndim[i-1] + 1)) - 0.5) * 2.0
            self.v.append(tmp_v)
    
    def updatePersonalBest(self, pf):
        if pf < self.pf:
            self.pf = pf
            # update p[]
            for i in range(len(self.x)):
                self.p[i] = self.x[i].copy()
    
    def getX(self):
        return self.x
    
    def updateParticlePosVel(self, g):
        for i in range(len(g)):
            self.v[i] = self._omega * self.v[i] + self._alpha1 * np.random.random(self.v[i].shape) * (self.p[i] - self.x[i]) + self._alpha2 * np.random.random(self.v[i].shape) * (g[i] - self.x[i])
            self.x[i] += self.v[i]
            self.x[i][np.abs(self.x[i]) > 10] = 10.0 * (np.random.random() - 0.5)

class Swarm:
    """ Implementation of a swarm
    """
    def __init__(self, network, omega = 1.0, alpha1 = 1.0, alpha2 = 1.0, swarmsize = 10, limit = 0.5):
        self._network = network
        self._nndim = network.getNetworkDimension()
        self._inputSize = network.getInputSize()
        self._swarmsize = swarmsize
        self._particleBuf = []
        for i in range(swarmsize):
            particle = Particle(self._nndim, self._inputSize, omega, alpha1, alpha2, limit)
            self._particleBuf.append(particle)
        self.g = []
        self.gf = sys.maxsize
        for i in range(len(self._nndim)):
            self.g.append(self._particleBuf[0].x[i].copy())
    
    def updateGlobalBest(self, i, gf):
        if gf < self.gf:
            self.gf = gf
            for j in range(len(self._nndim)):
                self.g[j] = self._particleBuf[i].x[j].copy()
    
    def updateSwarm(self, x, y):
        loss = -1
        # find global best
        for i in range(self._swarmsize):
            self._network.setAllParas(self._particleBuf[i].getX())
            loss = self._network.getLoss(x, y)
            self._particleBuf[i].updatePersonalBest(loss)
            self.updateGlobalBest(i, loss)
        # update particles
        for i in range(self._swarmsize):
            self._particleBuf[i].updateParticlePosVel(self.g)
        # use the best of the paras to terminate
        self._network.setAllParas(self.g)



class PSO:
    """ Particle swarm optimizer
    """
    def __init__(self, network, omega = 1.0, alpha1 = 1.0, alpha2 = 1.0, swarmsize = 10, limit = 0.5):
        self._network = network
        self._omega = omega
        self._alpha1 = alpha1
        self._alpha2 = alpha2
        self._swarmsize = swarmsize
        self._limit = limit
        self._swarm = Swarm(network, omega, alpha1, alpha2, swarmsize, limit)
        
    
    def update(self, x, y):
        self._swarm.updateSwarm(x, y)
