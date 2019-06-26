#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:12:16 2019

@author: Thomas Bury

Function to compute stable equilbria of seasonal Ricker model
(by running trajectories until they settle down)
"""


import numpy as np
import pandas as pd


# Set density-dep parameters 
alpha_b = 1/500
alpha_nb = 1/500



# Difference equation (no noise)
def de_fun(state, rb, rnb):
    '''
    Inputs:
        state: array of state variables [x,y]
        rb: breeding growth rate
        rnb: non-breeding growth rate
    Output:
        array for subsequent state
    '''        
    
    [x, y] = state   # x (y) population after breeding (non-breeding) period
    
    # Compute pop size after breeding period season t+1
    xnew = y * np.exp(rb - alpha_b * y )
    # Compute pop size after non-breeding period season t+1
    ynew =  xnew * np.exp(rnb - alpha_nb * xnew )
    # Ouput updated state        
    return np.array([xnew, ynew])




def find_equi(rb, rnb):
    '''
    Function to find the equilibrium of the system given the breeding
    and non-breeding growth rates.
    
    Inputs:
        rb: growth rate in the breeding period
        rng: growth rate in the non-breeding period
    
    Ouptut: One of
        array of equilibrium value
        string 'oscillations' if oscillations observed
    '''
    
    # Parameters
    s0 = [400,200] # Initial condition
    tmax = 2000     # Time steps to simulate
    eps = 0.1    # Error margin required for equilibria convergence
    
    # Set up
    tVals = np.arange(0,tmax+1,1)
    s = np.zeros([tmax+1,2])
    s[0] = s0
    
    # Run the system 
    for i in range(len(tVals)-1):
        s[i+1] = de_fun(s[i],rb,rnb)
        
    
    if abs(np.linalg.norm(s[-1]) - np.linalg.norm(s[-2])) < eps:
        if np.linalg.norm(s[-1]) < 0.01:
            out = np.array([0,0])
        else:
            out = s[-1]
        return out
    
    elif abs(np.linalg.norm(s[-1]) - np.linalg.norm(s[-3])) < eps:
        return ['Period-2 oscillations']*2
    
    elif abs(np.linalg.norm(s[-1]) - np.linalg.norm(s[-5])) < eps:
        return ['Period-4 oscillations']*2

    else:
        return ['No convergence or oscillations higher than period 4']*2
    
    
# Find equilibrium values over a sweep of growth parameters
      
# Growth parameters
rbVals = np.arange(0,3.05,0.05)
rnbVals = np.arange(-2,0.05,0.05)


# Create a DataFrame to store values
df_equiVals = pd.DataFrame([],columns=['rb','rnb','equi'])


# Create a list to store values
list_temp = []

for rb in rbVals:
    for rnb in rnbVals:
        # Make a list 
        equi = find_equi(rb,rnb)
        list_temp.append([rb,rnb,equi[0],equi[1]])
    print('Complete for rb = %.2f' %rb)


# Put into a DataFrame
df_equi = pd.DataFrame(list_temp, columns = ['rb','rnb','x_equi','y_equi'])

# Export as csv
df_equi.to_csv('data_export/equi_data/equi_data1.csv')









