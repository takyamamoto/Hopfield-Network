# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 08:40:49 2018

@author: user
"""

import numpy as np
np.random.seed(1)
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from skimage import data
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.transform import resize
#from tqdm import tqdm
import network

# ユーティリティ関数

def get_corrupted_input(input, corruption_level):
    """
    入力にノイズを付与
    """
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted

def plot(hn, data, test, predicted, figsize=(5, 7)):
    """
    元データ、テストデータ、推測値を描画
    """
    fig, axes = plt.subplots(len(data), 3, figsize=figsize)
    for i, axrow in enumerate(axes):
        if i == 0:
            axrow[0].set_title('train data')
            axrow[1].set_title("input data")
            axrow[2].set_title('output data')
        hn.plot_data(axrow[0], data[i])
        hn.plot_data(axrow[1], test[i], with_energy=True)
        hn.plot_data(axrow[2], predicted[i], with_energy=True)

        for ax in axrow:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
    return fig, axes

def preprocessing(img, w=128, h=128):
    # Resize image
    img = resize(img, (w,h), mode='reflect')
    
    # Thresholding
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2*(binary*1)-1 # Boolian to int
    
    # Reshape
    flatten = np.reshape(shift, (w*h))
    return flatten

# Load data
camera = data.camera()
astronaut = rgb2gray(data.astronaut())
horse = data.horse()
coffee = rgb2gray(data.coffee())

# Marge data
data = [camera, astronaut, horse, coffee]

# Preprocessing
print("Start to data preprocessing")
data = [preprocessing(d) for d in data]

# Hopfield Network インスタンスの作成 & 学習
hn = network.HopfieldNetwork()
hn.train_weights(data)

# 画像に 10% のノイズを付与し、テストデータとする
test = [get_corrupted_input(d, 0.3) for d in data]
# Hopfield Network からの出力
#predicted = hn.predict(test)

#plot(hn, data, test, predicted, figsize=(5, 5))
#hn.plot_weight()
