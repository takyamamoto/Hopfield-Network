# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 08:40:49 2018

@author: user
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

class HopfieldNetwork(object):      
    def train_weights(self, train_data):
        print("Start to train weights...")
        num_data =  len(train_data)
        num_neuron = train_data[0].shape[0]
        # initialize weights
        W = np.zeros((num_neuron, num_neuron))
        
        # hebb rule
        for i in tqdm(range(num_data)):
            W += np.outer(train_data[i], train_data[i])
        # Make diagonal element of W into 0
        diagW = np.diag(np.diag(W))
        W = W - diagW
        W /= num_data
        
        self.W = W 
    
    def predict(self, data, num_iter=10, threshold=0, asyn=False):
        print("Start to predict...")
        self.num_iter = num_iter
        self.threshold = threshold
        self.asyn = asyn
        
        # Define predict list
        predicted = []
        for i in tqdm(range(len(data))):
            predicted.append(self._run(data[i]))
        return predicted
    
    def _run(self, init_s):
        """
        synchronous update
        """
        if self.asyn==False:
            # Compute initial state energy
            s = init_s
            e = self.energy(s)
            
            # Iteration
            for i in range(self.num_iter):
                # Update s
                s = np.sign(self.W @ s - self.threshold)
                # Compute new state energy
                e_new = self.energy(s)
                
                # s is converged
                if e == e_new:
                    return s
                # Update energy
                e = e_new
            return s
                
        """
        asynchronous update
        """
    
    
    def energy(self, s):
        return -0.5 * s @ self.W @ s + np.sum(s * self.threshold)

    def plot_weights(self):
        plt.figure(figsize=(12, 9))
        w_mat = plt.imshow(self.W, cmap=cm.coolwarm)
        plt.colorbar(w_mat)

    def plot_data(self, ax, data):
        dim = int(np.sqrt(len(data)))
        assert dim * dim == len(data)

        img = (data.reshape(dim, dim) + 1) / 2
        ax.imshow(img, cmap=cm.Greys_r, interpolation='nearest')
        return ax