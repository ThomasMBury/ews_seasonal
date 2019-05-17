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


# System parameters
alpha_b = 1/250 # density dependent effects in breeding period
alpha_nb = 1/250 # density dependent effects in non-breeding period
rnb = -0.01 # growth rate in non-breeding period


# Difference equation for breeding population size
@jit(nopython=True)
def ricker_b(x, rb):
    '''
    Annual iteration for population size after breeding season.
    Inputs:
        x - breeding population size
        rb - growth rate during breeding season
    Ouptut:
        population size after the following breeding season.
    '''
    
    # Compute population size after non-breeding season based on x
    y = x * np.exp(rnb - alpha_nb * x )
    
    # Compute population after breeding season
    xnew = y * np.exp(rb - alpha_b * y )
    
    # Output new population size
    return xnew



# Ricker model for population size after breeding season
@jit(nopython=True)
def ricker_nb(y, rb):
    '''
    Annual iteration for population size after non-breeding season.
    Inputs:
        y - current population size after non-breeding season
        rb - growth rate during breeding season
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
bif_data_x = simulate(model=ricker_b, num_gens=100, rate_min=-0.5, rate_max=5, num_rates=1000, num_discard=100)
bif_data_y = simulate(model=ricker_nb, num_gens=100, rate_min=-0.5, rate_max=5, num_rates=1000, num_discard=100)



# Make plot of bifurcation
bifurcation_plot(bif_data_x, title='Non-breeding population size', xmin=-0.5, xmax=5, ymin=0, ymax=500, save=False)
bifurcation_plot(bif_data_y, title='Breeding population size', xmin=-0.5, xmax=5, ymin=0, ymax=500, save=False)



## Export bifurcation points
#bif_data_x.to_csv('../data_export/bif_data_x.csv')
#bif_data_y.to_csv('../data_export/bif_data_y.csv')





