# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 16:18:37 2018

@author: user
"""

from keras.datasets import mnist
(x_train, y_train) , (_, _ )= mnist.load_data()
data = []
for i in range(10):
    xi = x_train[y_train==i]
    data.append(xi[0])