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
        print("Start to train weights")
        num_data =  len(train_data)
        num_neuron = train_data[0].shape[0]
        # initialize weights
        W = np.zeros((num_neuron, num_neuron))
        
        # hebb rule
        for i in tqdm(range(num_data)):
            W += np.outer(train_data[i], train_data[i])
        # 対角項を0に
        diagW = np.diag(np.diag(W))
        W = W - diagW
        # データ数で割る
        W /= num_data
        self.W = W 
    
    def predict(self, data, num_iter=10, threshold=0, asyn=False):
        print("Start to predict")
        self.num_iter = num_iter
        self.threshold = threshold
        self.asyn = asyn
        predicted = []
        for i in tqdm(range(len(data))):
            predicted.append(self._run(data[i]))
        return predicted
    
    def _run(self, init_s, epsilon=0.001):
        """
        synchronous update
        """
        if self.asyn==False:
            # Compute initial state energy
            s = init_s
            e = self.energy(s)
            # Run 
            for i in range(self.num_iter):
                # Update s
                s = np.sign(self.W @ s - self.threshold)
                # Compute new state energy
                e_new = self.energy(s)
                
                # 収束した場合
                if abs(e_new - e) < epsilon:
                    return s
                # Update energy
                e = e_new
            return s
                
        """
        asynchronous update
        """
    
    
    def energy(self, s):
        return -0.5 * s @ self.W @ s + np.sum(s * self.threshold)

    def plot_weight(self):
        plt.imshow(self.W)
        """
        fig, ax = plt.subplots(figsize=(5, 3))
        heatmap = ax.pcolor(self.W, cmap=cm.coolwarm)
        plt.colorbar(heatmap)

        ax.set_xlim(0, self.dim)
        ax.set_ylim(0, self.dim)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        return fig, ax
        """
        
    def plot_data(self, ax, data, with_energy=False):
        dim = int(np.sqrt(len(data)))
        # このサンプルで扱う画像は縦横 同じ長さのもののみ
        assert dim * dim == len(data)

        img = (data.reshape(dim, dim) + 1) / 2
        ax.imshow(img, cmap=cm.Greys_r, interpolation='nearest')
        if with_energy:
            e = np.round(self.energy(data), 1)
            ax.text(0.95, 0.05, e, color='r', ha='right',
                    transform=ax.transAxes)
        return ax