#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:12:16 2019

@author: Thomas Bury

Script to generate points for bifurcation diagram of discrete-time model
Model: Seasoanl Ricker model with COEs, vary rnb
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Fixed model params
alpha_b = 0.01 # density dependent effects in breeding period
alpha_nb = 0.000672 # density dependent effects in non-breeding period
a = 0 # Strenghth of COEs
rb = 2.24 # Growth rate in breeding period

# rnb value to vary over
rnb_l = -2.5
rnb_h = 1

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
    xnew = y * np.exp(rb - a*x - alpha_b * y )
    # Compute pop size after non-breeding period season t+1
    ynew =  xnew * np.exp(rnb - alpha_nb * xnew )
    # Ouput updated state        
    return np.array([xnew, ynew])




def run_model(rb, rnb, tmax=1000, t_keep=100):
    '''
    Function to run model and keep final points
    
    Inputs:
        rb: growth rate in the breeding period
        rnb: growth rate in the non-breeding period
        tmax: number of time steps to run for
        t_keep: number of time steps to keep for plotting
    
    Ouptut:
        array of state values
    '''
    
    # Parameters
    s0 = [rb/alpha_b,rb/alpha_b] # Initial condition
    
    # Set up
    tVals = np.arange(0,tmax+1,1)
    s = np.zeros([tmax+1,2])
    s[0] = s0
    
    # Run the system 
    for i in range(len(tVals)-1):
        s[i+1] = de_fun(s[i],rb,rnb)
    s_out = s[-t_keep:]
    t_out = tVals[-t_keep:]
    dic_temp = {'Time':t_out,'Non-breeding':s_out[:,0],'Breeding':s_out[:,1]}
    df_out = pd.DataFrame(dic_temp)
    return df_out
    
    
# Do a sweep over rbvals     
# Growth parameters
rnbVals = np.arange(rnb_l,rnb_h,0.01).round(2)


# Create a list to store dfs
list_df = []

for rnb in rnbVals:
    
    # Run model 
    df_temp = run_model(rb,rnb,t_keep=20)
    df_temp['rnb'] = rnb
    list_df.append(df_temp)
    
    
# concatenate dfs
df_bif = pd.concat(list_df)

# Bifurcation plot
df_bif.plot(x='rnb',y='Non-breeding',kind='scatter')
df_bif.plot(x='rnb',y='Breeding',kind='scatter')


# Export data for plotting in mma
df_bif.to_csv('data/rnb_vary_a0.csv')





