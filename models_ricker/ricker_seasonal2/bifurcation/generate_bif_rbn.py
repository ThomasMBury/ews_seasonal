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





# Fixed system parameters
alpha_b = 1/250 # density dependent effects in breeding period
alpha_nb = 1/250 # density dependent effects in non-breeding period


#------------------------------
# Setup bifurcation diagram varying non-breeding growth rate


rb = 1 # growth rate in breeding period

# Difference equation for breeding population size
@jit(nopython=True)
def ricker_b(x, rnb):
    '''
    Annual iteration for population size after breeding season.
    Inputs:
        x - breeding population size
        rnb - growth rate during non-breeding season
    Ouptut:
        population size after the following breeding season.
    '''
    
    # Compute population size after non-breeding season based on x
    y = x * np.exp(rnb - alpha_nb * x )
    
    # Compute population after breeding season
    xnew = y * np.exp(rb - alpha_b * y )
    
    # Output new population size
    return xnew



# Difference equation for non-breeding population size
@jit(nopython=True)
def ricker_nb(y, rnb):
    '''
    Annual iteration for population size after non-breeding season.
    Inputs:
        y - current population size after non-breeding season
        rnb - growth rate during breeding season
    Ouptut:
        population size after the following non-breeding season.
    '''
    
    # Compute population size after breeding season based on y
    x = y * np.exp(rb - alpha_b * y )
    
    # Compute population after breeding seasone
    ynew = x * np.exp(rnb - alpha_nb * x )
    
    # Output new population size
    return ynew




# Simulate to get bifurcation points
bif_data_x = simulate(model=ricker_b, num_gens=100, rate_min=-1.25, rate_max=0, num_rates=1000, num_discard=100)
bif_data_y = simulate(model=ricker_nb, num_gens=100, rate_min=-1.25, rate_max=0, num_rates=1000, num_discard=100)


# Bifurcation plot
bifurcation_plot(bif_data_x, title='Non-breeding population size',
                 xmin=-1.25, xmax=0, ymin=0, ymax=500, save=False,
                 xlabel='Non-breeding growth rate (rnb)')
bifurcation_plot(bif_data_y, title='Breeding population size',
                 xmin=-1.25, xmax=0, ymin=0, ymax=500, save=False,
                 xlabel='Non-breeding growth rate (rnb)')



