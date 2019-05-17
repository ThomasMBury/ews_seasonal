#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 17:46:28 2018

Generate a bifurcation diagram of a discrete system.

Mdel 

@author: Thomas Bury
"""


from pynamical import bifurcation_plot, simulate
import numpy as np
from numba import jit



# Density dependence parameter
gamma = 1/200

# Ricker model
@jit(nopython=True)
def ricker(x, r):
    return x * np.exp(r-gamma * x)

# Simulate to get bifurcation points
pops = simulate(model=ricker, num_gens=100, rate_min=0.01, rate_max=4, num_rates=1000, num_discard=100)

# Make plot of bifurcation
bifurcation_plot(pops, title='Ricker Bifurcation Diagram', xmin=0, xmax=4, ymin=0, ymax=500, save=False)

## Export bifurcation points
#pops.to_csv('../data_export/bif_data_ricker.csv')