#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 18:04:51 2019


Code to simulate seasonal Logistic model with COE
Stationary simulations (fixed paramters)

@author: ThomasMBury
"""



#------------------------
# Import modules
#–----------------------

# Default python modules
import numpy as np
import pandas as pd
import os

# Import Python packages
from ewstools import ewstools

# Import from local directory
from cross_corr import cross_corr

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
dir_name = 'ews_stat'

if not os.path.exists('data_export/'+dir_name):
    os.makedirs('data_export/'+dir_name)




#--------------------------------
# Global parameters
#–-----------------------------


# Simulation parameters
dt = 1
t0 = 0
tmax = 1000 # make large (to get idealised statistics from stationary distribution)
tburn = 500 # burn-in period
seed = 0 # random number generation seed
rbif = 3.968 # flip bifurcation (from MMA bif file)
rl = 0 # low r value
rh = 5 # high r value
rinc = 0.01 # amount to increment r by



# EWS parameters
dt2 = 1 # spacing between time-series for EWS computation
rw = 1 # rolling window (compute EWS using full time-series)
bw = 1 # bandwidth (take the whole dataset as stationary)
lags = [1,2,3] # autocorrelation lag times
ews = ['var','ac','sd','cv','skew','kurt','smax','aic','cf'] # EWS to compute
ham_length = 40 # number of data points in Hamming window
ham_offset = 0.5 # proportion of Hamming window to offset by upon each iteration
pspec_roll_offset = 20 # offset for rolling window when doing spectrum metrics




#----------------------------------
# Model
#----------------------------------


# Function dynamic - outputs the subsequent state
def de_fun(state, control, params):
    '''
    Inputs:
        state: array of state variables [x,y]
        control: control parameter that is to be varied
        params: list of parameter values [kb, knb, rnb]
    Output:
        array for subsequent state
    '''
        
    
    [x, y] = state   # x (y) population after breeding (non-breeding) period
    [kb, knb, rnb, a] = params
    rb = control
    
    # Compute pop size after breeding period season t+1
    xnew = y * (1 + (rb - a*x) * (1-y/kb) )
    # Compute pop size after non-breeding period season t+1
    ynew =  xnew * (1 + rnb * (1-xnew/knb) )
    
    # Ouput updated state        
    return np.array([xnew, ynew])
    
    
   
# System parameters
    
kb = 224 # carrying capacity in breeding period
knb = -84.52 # carrying capacity in non-breeding period
rnb = -0.0568 # growth rate in non-breeding period
a = 0.0031 # regulates the strenght of COEs


# Parameter list
params = [kb, knb, rnb, a]

# Control parameter values
rVals = np.arange(rl, rh, rinc)


# Noise parameters
sigma_x = 0.1 # amplitude for x
sigma_y = 0.1 # amplitude for y

# Initial conditions
x0 = kb
y0 = x0 * np.exp(rnb * (1-x0/knb))




#--------------------------------------------
# Simulate (stationary) realisations of model for each r value
#-------------------------------------------



## Implement Euler Maryuyama for stocahstic simulation


# Set seed
np.random.seed(seed)

# Initialise a list to collect trajectories
list_traj_append = []

# Loop over control parameter values
print('\nBegin simulations \n')
for r in rVals:
    
    # Initialise array to store time-series data
    t = np.arange(t0,tmax,dt) # Time array
    s = np.zeros([len(t), 2]) # State array

    
    # Create brownian increments (s.d. sqrt(dt))
    dW_x_burn = np.random.normal(loc=0, scale=sigma_x*np.sqrt(dt), size = int(tburn/dt))
    dW_x = np.random.normal(loc=0, scale=sigma_x*np.sqrt(dt), size = len(t)) 
    
    dW_y_burn = np.random.normal(loc=0, scale=sigma_y*np.sqrt(dt), size = int(tburn/dt))
    dW_y = np.random.normal(loc=0, scale=sigma_y*np.sqrt(dt), size = len(t))
  
    
    # Noise vectors
    dW_burn = np.array([dW_x_burn, dW_y_burn]).transpose()
    dW = np.array([dW_x, dW_y]).transpose()
 
    # IC as a state vector
    s0 = np.array([x0 , y0])
    
    # Run burn-in period on initial condition
    for i in range(int(tburn/dt)):
        # Iterate
        s0 = de_fun(s0, r, params) + dW_burn[i]
        # Make sure that state variable remains >= 0 
        s0 = [np.max([k,0]) for k in s0]
        
        
    # Initial condition post burn-in period
    s[0]=s0
    
    # Run simulation
    for i in range(len(t)-1):
        s[i+1] = de_fun(s[i], r, params) + dW[i]
        # make sure that state variable remains >= 0 
        s[i+1] = [np.max([k,0]) for k in s[i+1]]
            
    # Store series data in a DataFrame
    data = {'Growth rate': r,
                'Time': t,
                'Post-breeding pop': s[:,0],
                'Post-non-breeding pop': s[:,1]}
    df_temp = pd.DataFrame(data)
    # Append to list
    list_traj_append.append(df_temp)
    
    print('Simulation with r='+str(r)+' complete')

#  Concatenate DataFrame from each realisation
df_traj = pd.concat(list_traj_append)
df_traj.set_index(['Growth rate','Time'], inplace=True)


# Coarsen time-series to have spacing dt2 (for EWS computation)
df_traj_filt = df_traj.loc[::int(dt2/dt)]


# Normalise each population size by breeding-carrying capacity

df_traj_filt['x/K'] = df_traj_filt['Post-breeding pop']/kb
df_traj_filt['y/K'] = df_traj_filt['Post-non-breeding pop']/kb




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
    # loop through sate variable
    for var in ['Post-breeding pop', 'Post-non-breeding pop']:
        
        ews_dic = ewstools.ews_compute(df_traj_filt.loc[r][var], 
                          roll_window = rw, 
                          band_width = bw,
                          lag_times = lags, 
                          ews = ews,
                          ham_length = ham_length,
                          ham_offset = ham_offset,
                          pspec_roll_offset = pspec_roll_offset,
                          sweep=True
                          )
        
        # The DataFrame of EWS
        df_ews_temp = ews_dic['EWS metrics']
        # The DataFrame of power spectra
        df_pspec_temp = ews_dic['Power spectrum']
        
        # Compute cross-correlation
        df_cross_corr = cross_corr(df_traj_filt.loc[r][['Post-breeding pop','Post-non-breeding pop']],
                                   roll_window = rw,
                                   span = 1)
        series_cross_corr = df_cross_corr['EWS metrics']['Cross correlation']
        


        # Include a column in the DataFrames for r value and variable
        df_ews_temp['Growth rate'] = r
        df_ews_temp['Variable'] = var
        df_ews_temp['Cross correlation'] = series_cross_corr
        
        
        df_pspec_temp['Growth rate'] = r
        df_pspec_temp['Variable'] = var
                
        # Add DataFrames to list
        appended_ews.append(df_ews_temp)
        appended_pspec.append(df_pspec_temp)
        
    # Print status every realisation
    print('EWS for r =  '+str(r)+' complete')


# Concatenate EWS DataFrames. Index [Growth rate, Variable, Time]
df_ews_full = pd.concat(appended_ews).reset_index().set_index(['Growth rate','Variable','Time'])
# Concatenate power spectrum DataFrames. Index [Realisation number, Variable, Time, Frequency]
df_pspec = pd.concat(appended_pspec).reset_index().set_index(['Growth rate','Variable','Time','Frequency'])


# Refine DataFrame to just have EWS data (no time dependence)
df_ews = df_ews_full.dropna().reset_index(level=2, drop=True).reorder_levels(['Variable', 'Growth rate'])
df_pspec = df_pspec.reset_index(level=2, drop=True).reorder_levels(['Variable', 'Growth rate','Frequency'])



#------------------------------------
## Export data / figures
#-----------------------------------


## Export EWS data

# Post-breeding season EWS DataFrame
df_ews_x = df_ews.loc['Post-breeding pop']
df_ews_x.to_csv('data_export/'+dir_name+'/ews_x.csv')

# Post non-breeding season EWS DataFrame
df_ews_y = df_ews.loc['Post-non-breeding pop']
df_ews_y.to_csv('data_export/'+dir_name+'/ews_y.csv')




