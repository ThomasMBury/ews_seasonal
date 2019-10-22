#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:12:16 2019

@author: Thomas Bury

Script to generate points for bifurcation diagram of discrete-time model
Model: Ricker model, vary r
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Fixed model params
alpha = 0.01 # density dependent

# r values to vary between
rl = 0
rh = 3.5

# Difference equation (no noise)
def de_fun(state, r):
    '''
    Inputs:
        state: array of state variables [x]
        r: growth rate
    Output:
        array for subsequent state
    '''        
    
    [x] = state
    
    # Compute pop size after breeding period season t+1
    xnew = x * np.exp(r - alpha * x )
    # Ouput updated state        
    return np.array([xnew])




def run_model(r, tmax=1000, t_keep=100):
    '''
    Function to run model and keep final points
    
    Inputs:
        r: growth rate
        tmax: number of time steps to run for
        t_keep: number of time steps to keep for plotting
    
    Ouptut:
        array of state values
    '''
    
    # Parameters
    s0 = [r/alpha] # Initial condition
    
    # Set up
    tVals = np.arange(0,tmax+1,1)
    s = np.zeros([tmax+1,1])
    s[0] = s0
    
    # Run the system 
    for i in range(len(tVals)-1):
        s[i+1] = de_fun(s[i],r)
        
    s_out = s[-t_keep:]
    t_out = tVals[-t_keep:]
    dic_temp = {'Time':t_out,'Pop':s_out[:,0]}
    df_out = pd.DataFrame(dic_temp)
    return df_out
    
    
# Do a sweep over rbvals     
# Growth parameters
rVals = np.arange(rl,rh,0.01).round(2)


# Create a list to store dfs
list_df = []

for r in rVals:
    
    # Run model 
    df_temp = run_model(r,t_keep=400)
    df_temp['r'] = r
    list_df.append(df_temp)
    
    
# concatenate dfs
df_bif = pd.concat(list_df)

# Bifurcation plot
df_bif.plot(x='r',y='Pop',kind='scatter')


# Export data for plotting in mma
df_bif.to_csv('data/r_vary.csv')





