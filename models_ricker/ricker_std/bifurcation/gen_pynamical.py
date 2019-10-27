#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 17:46:28 2018

Generate a bifurcation diagram of a discrete system.
Model: Seasonal Ricker model

@author: Thomas Bury
"""


from pynamical import bifurcation_plot, simulate

import numpy as np
from numba import jit





# Fixed model params
alpha = 0.01 # density dependent


# r values to vary between
rl = 0
rh = 3.5

#------------------------------
# Setup bifurcation diagram varying breeding growth rate


# Difference equation for breeding population size
@jit(nopython=True)
def ricker(x,r):
    '''
    Annual iteration for population size after breeding season.
    Inputs:
        x - breeding population size
        rb - growth rate during breeding season
    Ouptut:
        population size after the following breeding season.
    '''
    
    # Compute population size after non-breeding season based on x
    xnew = x * np.exp(r - alpha * x )

    # Output new population size
    return xnew






# Simulate to get bifurcation points
bif_data_x = simulate(model=ricker, num_gens=100, rate_min=0, rate_max=3.5, num_rates=1000, num_discard=100)


# Bifurcation plot
bifurcation_plot(bif_data_x, title='Non-breeding population size',
                 xmin=0, xmax=3.5, ymin=0, ymax=500, save=False,
                 xlabel='Breeding growth rate (rb)')





