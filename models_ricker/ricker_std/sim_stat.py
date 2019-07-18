#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 18:04:51 2019


Code to simulate Ricker model (no seasonality)
Stationary simulations (fixed paramters)

@author: ThomasMBury
"""



#------------------------
# Import modules
#–----------------------

# Default python modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# EWS module
from ewstools import ewstools


#----------------------
# Useful functions
#-----------------------

# Apply operation to column of DataFrame in place
def apply_inplace(df, field, fun):
    """ Apply function to a column of a DataFrame in place."""
    return pd.concat([df.drop(field, axis=1), df[field].apply(fun)], axis=1)



#---------------------
# Directory for data output
#–----------------------

# Name of directory within data_export
dir_name = 'ews_stat_r'

if not os.path.exists('data_export/'+dir_name):
    os.makedirs('data_export/'+dir_name)




#--------------------------------
# Global parameters
#–-----------------------------


# Simulation parameters
dt = 1
t0 = 0
tmax = 4000 # make large (to get idealised statistics from stationary distribution)
tburn = 200 # burn-in period
seed = 3 # random number generation seed




# EWS parameters
dt2 = 1 # spacing between time-series for EWS computation
rw = 1 # rolling window (compute EWS using full time-series)
span = 1 # span (take the whole dataset as stationary)
lags = [1,2,3] # autocorrelation lag times
ews = ['var','ac','sd','cv','skew','kurt','smax','smax/var','smax/mean'] # EWS to compute
ham_length = 40 # number of data points in Hamming window
ham_offset = 0.5 # proportion of Hamming window to offset by upon each iteration
pspec_roll_offset = 10 # offset for rolling window when doing spectrum metrics




#----------------------------------
# Model
#----------------------------------



# Model parameters
    
gamma = 0.01   	# Strength of density dependent effects
amp_dem = 0.02    # Demographic noise amplitude
amp_env = 0.02	# Environmental noise amplitude


# Bifurcation parameter
rl = 0
rh = 2
rinc = 0.05


# Function dynamic - outputs the subsequent state
def de_fun(state, params, noise):
    '''
    Inputs:
        state: single value
        params: array of parameters [r]
        noise: array of noise values for env. and dem noise
    
    Output:
        array for subsequent state
    '''
        
       
    x = state   # x population size
    [r] = params
    
    # Compute pop size after breeding period season t+1
    xnew = x * np.exp(r - gamma * x) + amp_dem*np.sqrt(x)*noise[0] + amp_env*x*noise[1]
    
    # Ouput updated state        
    return xnew
    
 

# Control parameter values
rVals = np.arange(rl, rh, rinc).round(2)





#--------------------------------------------
# Simulate (stationary) realisations of model for each r value
#-------------------------------------------


## Implement Euler Maryuyama for stocahstic simulation


# Set seed
np.random.seed(seed)

# Initialise a list to collect trajectories
list_traj_append = []

# Time values
t = np.arange(t0,tmax,dt)

# Use the same noise values for each simulation
dW_burn = np.random.normal(loc=0, scale=1, size = [int(tburn),2])
dW = np.random.normal(loc=0, scale=1, size = [len(t),2])




# Loop over control parameter values
print('\nBegin simulations \n')
for r in rVals:


    # Set initial condition 
    s0 = r/gamma
        
    # Set parameter values
    params = [r]
    
    # Initialise array to store time-series data
    s = np.zeros(len(t)) # State array



    
    # Run burn-in period on initial condition
    for i in range(int(tburn)):
         # Iterate
         s0 = de_fun(s0, params, dW_burn[i])
         # Make sure that state variable remains >= 0 
         s0 = np.max([s0,0])
        

        
    # Initial condition post burn-in period
    s[0]=s0
        
    # Run simulation
    for i in range(len(t)-1):
        s[i+1] = de_fun(s[i], params, dW[i])
        # make sure that state variable remains >= 0 
        s[i+1] = np.max([s[i+1],0])
                
    # Store series data in a DataFrame
    data = {'r': r,
            'Time': t,
            'State variable': s}
    df_temp = pd.DataFrame(data)
    # Append to list
    list_traj_append.append(df_temp)
        
    print('Simulation with r=%.2f complete' %r)

#  Concatenate DataFrame from each realisation
df_traj = pd.concat(list_traj_append)
df_traj.set_index(['r','Time'], inplace=True)





#----------------------
## Execute ews_compute for each r value and each variable
#---------------------


# Set up a list to store output dataframes from ews_compute
# We will concatenate them at the end
appended_ews = []
appended_pspec = []


# loop through realisation number
print('\nBegin EWS computation\n')
for r in rVals:       
         
    
    # If time-series goes extinct at any point, do not compute EWS
    if any(df_traj.loc[r]['State variable'] == 0):
        continue
    ews_dic = ewstools.ews_compute(df_traj.loc[r]['State variable'], 
                      roll_window = rw,
                      smooth = 'Lowess',
                      span = span,
                      lag_times = lags, 
                      ews = ews,
                      ham_length = ham_length,
                      ham_offset = ham_offset,
                      sweep = False
                      )
    
    # The DataFrame of EWS
    df_ews_temp = ews_dic['EWS metrics']
    # The DataFrame of power spectra
    df_pspec_temp = ews_dic['Power spectrum']
    
    # Include a column in the DataFrames for r value and variable
    df_ews_temp['r'] = r    
    df_pspec_temp['r'] = r

    # Add DataFrames to list
    appended_ews.append(df_ews_temp)
    appended_pspec.append(df_pspec_temp)
        
    # Print status
    print('EWS for r = %.2f complete' %r)


# Concatenate EWS DataFrames
df_ews_full = pd.concat(appended_ews)
# Select ews at tmax (the rest should be Nan since using rw=1)
df_ews = df_ews_full[df_ews_full.index==tmax-1].reset_index(drop=True).set_index('r')


# Concatenate power spec DataFrames
df_pspec_full = pd.concat(appended_pspec)
# Select pspec at tmax (the rest should be Nan since using rw=1)
df_pspec = df_pspec_full.loc[tmax-1].reset_index().set_index(['r','Frequency'])


# Plot the power spectrum normalised by the variance
df_pspec_norm = df_pspec['Empirical']/df_ews['Variance']
#df_pspec_norm.unstack(level=0).plot()





#------------------------------------
## Export data for plotting elsewhere
#-----------------------------------


## Export EWS data
df_ews.to_csv('data_export/'+dir_name+'/ews.csv')

## Export power spectrum data
df_pspec.to_csv('data_export/'+dir_name+'/pspec.csv')
df_pspec_norm.to_csv('data_export/'+dir_name+'/pspec_norm.csv', header='Empirical')








