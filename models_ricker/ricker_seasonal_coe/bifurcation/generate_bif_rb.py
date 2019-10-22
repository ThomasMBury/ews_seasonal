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
alpha_b = 0.01 # density dependent effects in breeding period
alpha_nb = 0.000672 # density dependent effects in non-breeding period



#------------------------------
# Setup bifurcation diagram varying breeding growth rate


rnb = -0.0568 # growth rate in non-breeding period
rbEmp = 2.24 # empirically measured rb

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



# Difference equation for non-breeding population size
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
bif_data_x = simulate(model=ricker_b, num_gens=100, rate_min=-0.5, rate_max=rbEmp, num_rates=1000, num_discard=100)
bif_data_y = simulate(model=ricker_nb, num_gens=100, rate_min=-0.5, rate_max=rbEmp, num_rates=1000, num_discard=100)


# Bifurcation plot
bifurcation_plot(bif_data_x, title='Non-breeding population size',
                 xmin=-0.5, xmax=rbEmp, ymin=0, ymax=500, save=False,
                 xlabel='Breeding growth rate (rb)')
bifurcation_plot(bif_data_y, title='Breeding population size',
                 xmin=-0.5, xmax=rbEmp, ymin=0, ymax=500, save=False,
                 xlabel='Breeding growth rate (rb)')






# Export bifurcation points
bif_data_x.to_csv('../data_export/bif_data/bifdata_rb_x.csv')
bif_data_y.to_csv('../data_export/bif_data/bifdata_rb_y.csv')





