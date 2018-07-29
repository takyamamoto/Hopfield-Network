# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 08:40:49 2018

@author: user
"""

import numpy as np
np.random.seed(1)
from matplotlib import pyplot as plt
from skimage.filters import threshold_mean
from skimage.transform import resize
from tqdm import tqdm
import network
# Load data
from sklearn.datasets import fetch_olivetti_faces
 
# Utils
def get_corrupted_input(input, corruption_level):
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted

def plot(hn, data, test, predicted, figsize=(10, 14)):
    """
    fig = plt.figure(figsize=(6,6))
    plt.title("Eigen Faces")
    plt.axis("off")
    for i in range(num_people):         
        #　Show result
        for j in range(n_comp):
            ax = fig.add_subplot(num_people, n_comp, 1 + n_comp * i + j,
                                 xticks=[], yticks=[])
            ax.set_title("{0:01.3f}".format(E[j]))
            if j==0:
                ax.set_ylabel("No."+str(i), rotation=0, fontsize=10, labelpad=20)
            ax.imshow(components[j, :].reshape(64, 64))
     
    plt.tight_layout()
    plt.savefig("EigenFaces.png")
    plt.show()
    plt.close()
    """
    fig, axes = plt.subplots(len(data), 3, figsize=figsize)
    for i, axrow in enumerate(axes):
        if i == 0:
            axrow[0].set_title('train data')
            axrow[1].set_title("input data")
            axrow[2].set_title('output data')
        hn.plot_data(axrow[0], data[i])
        hn.plot_data(axrow[1], test[i])
        hn.plot_data(axrow[2], predicted[i])

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
    """
    Load Olivetti faces dataset
     
    oliv.images (400, 64, 64) : 400 face images
    oliv.data   (400, 4096)   : reshape images (4096=64x64)
    oliv.target (400,)        : 40 label of faces
    """

    oliv = fetch_olivetti_faces()
    num_people = 6
    
    print("Show dataset...")
    # Show samples
    fig = plt.figure(figsize=(6,6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in tqdm(range(10*num_people)):
      ax = fig.add_subplot(10, 10, i+1, xticks=[], yticks=[])
      ax.imshow(oliv.images[i], cmap=plt.cm.bone, interpolation='nearest')
    plt.savefig("fetch_olivetti_faces.png")
    plt.show()
    plt.close()
    
    # Marge data
    data_num_list = [i*10 for i in range(num_people)]
    data = [oliv.images[i] for i in data_num_list]
    
    # Preprocessing
    print("Start to data preprocessing...")
    data = [preprocessing(d) for d in data]
    
    # Create Hopfield Network Model
    hn = network.HopfieldNetwork()
    hn.train_weights(data)
    
    """
    test_num_list = [d+2 for d in data_num_list]
    test = [oliv.images[i] for i in test_num_list]
    test = [preprocessing(d) for d in test]
    """
    test = [get_corrupted_input(d, 0.1) for d in data]
    
    predicted = hn.predict(test, threshold=20)
    
    plot(hn, data, test, predicted, figsize=(5, 5))
    #hn.plot_weights()

main()