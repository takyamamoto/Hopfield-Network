# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 08:40:49 2018

@author: user
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from lattice import Lattice2D
from typing import List

class HopfieldNetwork(object):      
    """ Run a Hopfield Neural Network

    ==== Attributes ===
        num_neron: the size of the flattened training data
        W: the weight matrix
        num_iter: the maximum number of iterations
        threshold: The convergence threshold
        asyn: ?
        exp1/2: Precomputed exponent values
    """
    num_neuron: int
    W: np.ndarray
    num_inter: int
    threshold: float
    asyn: bool
    exp1: np.ndarray
    exp1: np.ndarray

    # TODO: get rid of the asyn attribute
    # TODO: Add the precomputed exponetial values
    # here they are intializing the weights and stationary energies
    # this function is closer to an initializer than a training function
    # the training happens in the predict and run methods
    def train_weights(self, train_data):
        print("Start to train weights...")
        num_data =  train_data.size[0] * train_data.size[1]
        # num_data =  len(train_data)
        self.num_neuron = train_data.size[0]
        
        # initialize weights
        W = np.zeros((self.num_neuron, self.num_neuron))
        # this wont work since Lattice2D does not have an iterative method
        rho = np.sum([np.sum(t) for t in train_data]) / (num_data*self.num_neuron)
        
        # Hebb rule
        for i in tqdm(range(num_data)):
            t = train_data[i] - rho
            W += np.outer(t, t)
        
        # Make diagonal element of W into 0
        diagW = np.diag(np.diag(W))
        W = W - diagW
        W /= num_data
        
        self.W = W 
    
    def predict(self, data: List[Lattice2D], num_iter=20, threshold=0, asyn=True):
        # in the orginal code data is a list containing flattened arrays
        # changing it to a list of lattices
        print("Start to predict...")
        self.num_iter = num_iter
        self.threshold = threshold
        self.asyn = asyn  # not important for our use
        
        # list aliasing
        copied_data = np.copy(data)
        
        # Define predict list
        predicted = []
        # this for loops all the example images
        # copied_data[i] is a 1xn array
        for i in tqdm(range(len(data))):
            predicted.append(self._run(copied_data[i]))
        return predicted
    
    def _run(self, init_s):
        """ Asynchronous update

            === Parameters ===
            init_s: the input data, in the form of a row vector; will be
            changing this to a Lattice2D object
        """
        # Compute initial state energy
        s = init_s
        e = self.energy(s)
        
        # Iteration
        for i in range(self.num_iter):
            for j in range(100):
                # Select random neuron
                idx = np.random.randint(0, self.num_neuron) 
                # This line is the previous activation function
                # v = self.W[idx].T @ s - self.threshold
                # s[idx] = np.sign(v)

                # The new way for computing v and s
                update_lattice(self, s, idx), 
            
            # Compute new state energy
            e_new = self.energy(s)
            
            # s is converged
            if e == e_new:
                return s
            # Update energy
            e = e_new
        return s
    
    def _acceptance(self, original: Lattice2D, site: int):
        """Return the acceptance probability according to the metropolis
        algorithm w/ single spin dynamic."""
        # this looks weird for the delta energy term but check page 52 of the
        # monte carlo documents i linked as well as my personal notes in the
        # drive
        nearest_neigh, nn_idx = original.neighbours(site)
        w_ik = self.W[nn_idx, k]
        assert(w_ik.size() = 4)
        delta_energy = original.spins[site] * ( np.dot(w_ik, nearest_neigh) - 2)

        # TODO check on the exp values
        if delta_energy > 1:
            return self.exp2[site], delta_energy

        elif delta_energy > 0:
            return self.exp1[site], delta_energy
        else:
            return 1.0, delta_energy

    def _change_state(self, original: Lattice, site: int):
        """ Update the neurons by changing the spin value at <site>
        """
        original.spins[site] = -1 * original.spins[site]

    def update_lattice(self, original: Lattice, site: int) -> None:
        """ update the lattice according to the acceptance probability
        """
        number = r.uniform(0, 1)

        while number == 1:
            number = r.uniform(0, 1)

        accept = self._acceptance(original, site)

        if number < accept[0]:
            self._change_state(original, site, accept[1])
    
    def energy(self, s):
        return -0.5 * s @ self.W @ s + np.sum(s * self.threshold)

    def plot_weights(self):
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self.W, cmap=cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title("Network Weights")
        plt.tight_layout()
        plt.savefig("weights.png")
        plt.show()
