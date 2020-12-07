from optimizer import *
from network import *
import numpy as np
from matplotlib import pyplot as plt


class Trainer(object):
    """
    A simple trainer
    """
    def __init__(self, network, optimizer, x_train, y_train, x_test, y_test, epoche=20, batch_size=100, optimizer_para={}, verbose=False):
        super().__init__()
        self._network = network
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._optimizerDic = {'sgd':SGD, 'pso': PSO}
        self._optimizer = self._optimizerDic[optimizer.lower()](network, **optimizer_para)
        self._epoche = epoche
        self._batch_size = batch_size
        self._verbose = verbose
        self._current_iter = 0
        self._current_epoche = 0
        self._iter_per_epoche = int(max(x_train.shape[0] / batch_size, 1))
        self._max_iter = int(epoche * self._iter_per_epoche)
        self._batch_loss_list = []
        self._train_loss_list = []
        self._test_loss_list = []
        self._test_acc_list = []
    
    def train_step(self):
        """ One train step.
         one batch one step
        1. shuffled input batch
        2. Compute loss and gradient
        3. Optimizing
        4. Epoche test
        """
        index_mask = np.random.choice(self._x_train.shape[0], self._batch_size)
        x_batch = self._x_train[index_mask]
        y_batch = self._y_train[index_mask]

        self._optimizer.update(x_batch, y_batch)
        loss = self._network.getLoss(x_batch, y_batch)
        self._batch_loss_list.append(loss)
        if self._verbose: print('Train batch loss: ', str(loss))

        self._current_iter += 1
        if self._current_iter % self._iter_per_epoche == 0:
            self._current_epoche += 1

            x_train_sample, y_train_sample = self._x_train, self._y_train
            x_test_sample, y_test_sample = self._x_test, self._y_test

            train_loss = self._network.getLoss(x_train_sample, y_train_sample)
            test_loss = self._network.getLoss(x_test_sample, y_test_sample)
            self._train_loss_list.append(train_loss)
            self._test_loss_list.append(test_loss)
            if self._verbose: print('=== Epoche', str(self._current_epoche), ': train loss:', str(train_loss), ': test loss:', str(test_loss))
    
    def train(self):
        for i in range(self._max_iter):
            self.train_step()

        final_loss = self._network.getLoss(self._x_test, self._y_test)

        if self._verbose:
            print('=== Totol Iterations:', str(self._max_iter), '===')
            print('=== Final loss test:', str(final_loss), '===')
    
    def visualize(self):
        x = np.arange(self._current_epoche)
        y = self._test_loss_list
        plt.title('Loss visualization')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(x, y)
        plt.show()