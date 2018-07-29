# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 08:40:49 2018

@author: user
"""

import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_mean
import network
from keras.datasets import mnist

# Utils
def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data

def plot(data, test, predicted, figsize=(3, 3)):
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]
    
    fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
    for i in range(len(data)):
        if i==0:
            axarr[i, 0].set_title('Train data')
            axarr[i, 1].set_title("Input data")
            axarr[i, 2].set_title('Output data')
            
        axarr[i, 0].imshow(data[i])
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(test[i])
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted[i])
        axarr[i, 2].axis('off')
            
    plt.tight_layout()
    plt.savefig("result_mnist.png")
    plt.show()

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
    model = network.HopfieldNetwork()
    model.train_weights(data)
    
    # Make test datalist
    test = []
    for i in range(3):
        xi = x_train[y_train==i]
        test.append(xi[1])
    test = [preprocessing(d) for d in test]
    
    predicted = model.predict(test, threshold=50, asyn=True)
    print("Show prediction results...")
    plot(data, test, predicted, figsize=(5, 5))
    print("Show network weights matrix...")
    model.plot_weights()
    
if __name__ == '__main__':
    main()