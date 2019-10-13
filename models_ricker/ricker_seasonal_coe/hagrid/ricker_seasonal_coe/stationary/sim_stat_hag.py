#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 18:04:51 2019


Code to simulate seasonal Ricker model with COEs
Stationary simulations (fixed paramters)

@author: ThomasMBury
"""



#------------------------
# Import modules
#–----------------------

print("Start importing modules")

# Default python modules
import numpy as np
import pandas as pd
import os
import sys

# EWS module
import ewstools
print("Modules imported successfully")


#----------------------
# Useful functions
#-----------------------

# Apply operation to column of DataFrame in place
def apply_inplace(df, field, fun):
    """ Apply function to a column of a DataFrame in place."""
    return pd.concat([df.drop(field, axis=1), df[field].apply(fun)], axis=1)


# Get parameter a from external input
a = float(sys.argv[1])
print("Running simulation for a={}".format(a))

#---------------------
# Directory for data output
#–----------------------


#--------------------------------
# Global parameters
#–-----------------------------


# Simulation parameters
dt = 1
t0 = 0
tmax = 1000 # make large (to get idealised statistics from stationary distribution)
tburn = 2000 # burn-in period
seed = 0 # random number generation seed


# EWS parameters
dt2 = 1 # spacing between time-series for EWS computation
rw = 1 # rolling window (compute EWS using full time-series)
bw = 1 # bandwidth (take the whole dataset as stationary)
span=1
lags = [1,2,3] # autocorrelation lag times
ews = ['var','ac','sd','cv','skew','kurt','smax','smax/var'] # EWS to compute
ham_length = 40 # number of data points in Hamming window
ham_offset = 0.5 # proportion of Hamming window to offset by upon each iteration
pspec_roll_offset = 20 # offset for rolling window when doing spectrum metrics




#----------------------------------
# Model
#----------------------------------


# Noise parameters
amp_dem_x = 0.01 # amplitude of demographic noise
amp_dem_y = 0.01
amp_env_x= 0.01 # amplitude of environmental noise
amp_env_y= 0.01


# Model parameters
alpha_b = 0.01 # density dependent effects in breeding period
alpha_nb = 0.000672 # density dependent effects in non-breeding period


# Function dynamic - outputs the subsequent state
def de_fun(state, params, noise):
    '''
    Inputs:
        state: array of state variables [x,y]
        params: array of parameters [rb, rnb]
        noise: array of noise values for env. and dem noise
    
    Output:
        array for subsequent state
    '''
        
    
    [x, y] = state   # x: non-breeding pop size, y: breeding pop size
    [rb, rnb] = params
    
    # Compute pop size after breeding period season t+1
    xnew = y * np.exp(rb - a*x - alpha_b * y ) + amp_dem_x*np.sqrt(y)*noise[0] + amp_env_x*y*noise[1]
    # Compute pop size after non-breeding period season t+1
    ynew =  xnew * np.exp(rnb - alpha_nb * xnew ) + amp_dem_y*np.sqrt(abs(xnew))*noise[2] + amp_env_y*xnew*noise[3]
    
    # Ouput updated state        
    return np.array([xnew, ynew])
    


# Growth parameters
rbVals = np.arange(0, 3.05, 0.1).round(2)
rnbVals = np.arange(-3, 0.05, 0.1).round(2)




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
dW_burn = np.random.normal(loc=0, scale=1, size = [int(tburn),4])
dW = np.random.normal(loc=0, scale=1, size = [len(t),4])


# Loop over control parameter values
print('\nBegin simulations \n')
for rb in rbVals:
    for rnb in rnbVals:
        
        # Set initial condition 
        s0 = np.array([rb/alpha_b,rb/alpha_b])
        
        # Set parameter values
        params = [rb, rnb]
    
        # Initialise array to store time-series data
        s = np.zeros([len(t), 2]) # State array

    
        # Run burn-in period on initial condition
        for i in range(int(tburn)):
            # Iterate
            s0 = de_fun(s0, params, dW_burn[i])
            # Make sure that state variable remains >= 0 
            s0 = [np.max([k,0]) for k in s0]
        
        
        # Initial condition post burn-in period
        s[0]=s0
        
        # Run simulation
        for i in range(len(t)-1):
            s[i+1] = de_fun(s[i], params, dW[i])
            # make sure that state variable remains >= 0 
            s[i+1] = [np.max([k,0]) for k in s[i+1]]
                
        # Store series data in a DataFrame
        data = {'rb': rb,
                'rnb': rnb,
                'Time': t,
                'Non-breeding': s[:,0],
                'Breeding': s[:,1]}
        df_temp = pd.DataFrame(data)
        # Append to list
        list_traj_append.append(df_temp)
        
    print('Simulation with rb=%.2f complete' %rb)

#  Concatenate DataFrame from each realisation
df_traj = pd.concat(list_traj_append)
df_traj.set_index(['rb','rnb','Time'], inplace=True)






#----------------------
## Execute ews_compute for each parameter combination
#---------------------


# Set up a list to store output dataframes from ews_compute
# We will concatenate them at the end
appended_ews = []
appended_pspec = []


print('\nBegin EWS computation\n')

# Loop through growth rates
for rb in rbVals:
    for rnb in rnbVals:
            
        # Loop through state variable
        for var in ['Non-breeding', 'Breeding']:
            
            
            # If time-series goes extinct at any point, do not compute EWS
            if any(df_traj.loc[(rb, rnb)]['Non-breeding'] == 0):
                break
            ews_dic = ewstools.core.ews_compute(df_traj.loc[(rb,rnb)][var], 
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
            df_ews_temp['rb'] = rb
            df_ews_temp['rnb'] = rnb
            df_ews_temp['Variable'] = var
            
            df_pspec_temp['rb'] = rb
            df_pspec_temp['rnb'] = rnb
            df_pspec_temp['Variable'] = var
                    
            # Add DataFrames to list
            appended_ews.append(df_ews_temp)
            appended_pspec.append(df_pspec_temp)
            
        # Print status
        print('EWS for (rb, rnb) = (%.2f, %.2f) complete' %(rb,rnb))
    

# Concatenate EWS DataFrames
df_ews_full = pd.concat(appended_ews)
# Select ews at tmax (the rest should be Nan since using rw=1)
df_ews = df_ews_full[df_ews_full.index==tmax-1].reset_index(drop=True).set_index(['Variable','rb','rnb'])


# Concatenate power spec DataFrames
df_pspec_full = pd.concat(appended_pspec)
# Select pspec at tmax (the rest should be Nan since using rw=1)
df_pspec = df_pspec_full[df_pspec_full.index==tmax-1].reset_index().set_index(['Variable','rb', 'rnb','Frequency'])





#------------------------------------
## Export data for plotting elsewhere
#-----------------------------------

## Export EWS data
df_ews.to_csv('ews.csv')

## Export pspec data
df_pspec.to_csv('pspec.csv')




