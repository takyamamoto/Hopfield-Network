""" lattice.py

File containg lattices of different dimenions for different purposes.
Each lattice tpe is represented by a class.
The different lattices contained are: 
    Ising: represents a 1D isling lattice
    Ising2D: represents a 2D isling lattice
    Potts: represents a 1D q-pott lattice
    Potts2D: represents a 2D q-pott lattice

last updated: May 31
@author: organicWinesOnly
"""
import numpy as np
from typing import *
import random as r
from helpers_mc import flip_coin


########################################################################
# Lattice2D
########################################################################
class Lattice2D:
    """Create a 2d lattice

     === Attributes ===
        size: amount of spins in the lattice
        energy: total energy in the lattice
        m: magnetization per spin
        temp: starting temperature
        beta: inverse of temp we want our system to come to at equilibrium
        spins: ndarray repersenting the spins in our lattice

        Repersentation Invariants:
        """
    size: Tuple
    total_energy: float
    m: float
    spins: np.ndarray

    def __init__(self, data: np.ndarray, run_at: float):
        """Initailize a 2d lattice

        === parameters ===
        data: flatted matrix -> row vector
        temp: the temp you want the sytem to start at, 0 or -1
        run_at: temperture you want the system to come to
        m: magnetization
        total_energy: energy of lattice
        """
        self.beta = 1 / run_at

        # ensure they array has been flattened
        if data.shape[0] != 1:
            data = data.flatten()

        self.spins = data
        self.size = data.size

        interaction_ = np.zeros(self.spins.shape)
        for i in range(self.size):
                nn_spins, _ = self.neighbours(i)
                interaction_[i] = np.sum(nn_spins) * self.spins[i]

    def neighbours(self, site: int) -> List[np.ndarray]:
        """Return neighbouring spin values alng with their indicies
        [<left>, <right>, <above> , <below>].

        Helliac Periodic Boundary conditions are applied
        """
        left = (site - 1) % self.size
        right = (site + 1) % self.size
        up = (site + self.spins.shape[0]) % self.size
        down = (site - self.spins.shape[0]) % self.size

        left_ = self.spins[left]
        right_ = self.spins[right]
        up_ = self.spins[up]
        down_ = self.spins[down]

        values = np.array(left_, right_, up_, down_)
        idx = np.array(left, right, up, down)

        return [values, idx]


