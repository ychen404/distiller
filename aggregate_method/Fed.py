#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
from math import pi
from math import cos
from math import floor

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

# cosine annealing learning rate schedule
def cosine_annealing(epoch, n_epochs, n_cycles, lrate_max):
	epochs_per_cycle = floor(n_epochs/n_cycles)
	cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
	return lrate_max/2 * (cos(cos_inner) + 1)


if __name__ == "__main__":
    n_epochs = 100
    n_cycles = 5
    lrate_max = 0.001
    
    series = [cosine_annealing(i, n_epochs, n_cycles, lrate_max) for i in range(n_epochs)]
    print(series)