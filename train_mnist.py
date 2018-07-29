# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 08:40:49 2018

@author: user
"""

import numpy as np
np.random.seed(1)
from matplotlib import pyplot as plt
from skimage.filters import threshold_mean
import matplotlib.cm as cm
import network
from keras.datasets import mnist

# Utils
def get_corrupted_input(input, corruption_level):
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted


def plot_data(ax, data):
    dim = int(np.sqrt(len(data)))
    
    img = (data.reshape(dim, dim) + 1) / 2
    ax.imshow(img, cmap=cm.Greys_r, interpolation='nearest')
    return ax

def plot(data, test, predicted, figsize=(3, 3)):
    fig, axes = plt.subplots(len(data), 3, figsize=figsize)
    for i, axrow in enumerate(axes):
        if i == 0:
            axrow[0].set_title('Train data')
            axrow[1].set_title("Input data")
            axrow[2].set_title('Output data')
        
        plot_data(axrow[0], data[i])
        plot_data(axrow[1], test[i])
        plot_data(axrow[2], predicted[i])

        for ax in axrow:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
    return fig, axes

def preprocessing(img):
    w, h = img.shape
    # Thresholding
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2*(binary*1)-1 # Boolian to int
    
    # Reshape
    flatten = np.reshape(shift, (w*h))
    return flatten

def main():
    # Load data
    (x_train, y_train), (_, _ )= mnist.load_data()
    data = []
    for i in range(3):
        xi = x_train[y_train==i]
        data.append(xi[0])
    
    # Preprocessing
    print("Start to data preprocessing...")
    data = [preprocessing(d) for d in data]
    
    # Create Hopfield Network Model
    hn = network.HopfieldNetwork()
    hn.train_weights(data)
    
    # Make test datalist
    test = []
    for i in range(3):
        xi = x_train[y_train==i]
        test.append(xi[1])
    test = [preprocessing(d) for d in test]
    
    predicted = hn.predict(test, threshold=50, asyn=True)
    plot(data, test, predicted, figsize=(5, 5))
    hn.plot_weights()

main()