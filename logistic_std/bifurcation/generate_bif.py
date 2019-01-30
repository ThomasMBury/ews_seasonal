#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 17:46:28 2018

Generate a bifurcation diagram of a discrete system.
Model: : Discrete logistic model

@author: Thomas Bury
"""


from pynamical import bifurcation_plot, simulate
import numpy as np
from numba import jit



# Model parameters (from Betini et al. (2013))
kb = 224    # Carrying capacity for breeding period

# Logistic model
@jit(nopython=True)
def logistic(x, rb):
    # Output new population size
    return x * (1 + rb*(1-x/kb))


# Simulate to get bifurcation points
bif_data = simulate(model=logistic, num_gens=100, rate_min=0.5, rate_max=5, num_rates=1000, num_discard=100)


# Make plot of bifurcation
bifurcation_plot(bif_data, title='Ricker Bifurcation Diagram', xmin=0, xmax=5, ymin=0, ymax=500, save=False)




# Export bifurcation points
bif_data.to_csv('../data_export/bif_data.csv')




