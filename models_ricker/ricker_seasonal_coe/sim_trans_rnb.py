#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:41:47 2018

@author: Thomas Bury

Simulate transient simulations of the seasonal Ricker model with COEs
varyting rnb

"""

# import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 

# import ewstools
import ewstools


# import cross correlation function
from cross_corr import cross_corr


#---------------------
# Directory for data output
#–----------------------

# Name of directory within data_export 
dir_name = 'trans_rnb'

if not os.path.exists('data_export/'+dir_name):
    os.makedirs('data_export/'+dir_name)


#--------------------------------
# Global parameters
#–-----------------------------


# Simulation parameters
dt = 1 # time-step (must be 1 since discrete-time system)
t0 = 0
tmax = 1000
tburn = 200 # burn-in periods
numSims = 3
seed = 10 # random number generation seed


# EWS parameters
dt2 = 1 # spacing between time-series for EWS computation
rw = 0.4 # rolling window
span = 0.2 # Lowess span
lags = [1,2,3] # autocorrelation lag times
ews = ['var','ac','sd','cv','skew','kurt','smax','smax/mean','smax/var'] # EWS to compute
ham_length = 40 # number of data points in Hamming window
ham_offset = 0.5 # proportion of Hamming window to offset by upon each iteration
pspec_roll_offset = 20 # offset for rolling window when doing spectrum metrics


#----------------------------------
# Simulate many (transient) realisations
#----------------------------------


# Empirical param values
rb_emp = 2.24
rnb_emp = -0.0568


# Model parameters    
rb = rb_emp   # Growth rate for breeding period
alpha_b = 0.01 # density dependent effects in breeding period
alpha_nb = 0.000672 # density dependent effects in non-breeding period
a = 0 # Strength of carry-over effects

# Noise parameters
amp_dem_b = 0.02 # amplitude of demographic noise
amp_dem_nb = 0.02
amp_env_b = 0.02 # amplitude of environmental noise
amp_env_nb = 0.02



# Bifurcation parameter
rnb_l = -2.5
rnb_h = rnb_emp
rnb_crit = -2.3


# Function dynamic - outputs the subsequent state
def de_fun(state, control, params, noise):
    '''
    Inputs:
        state: array of state variables [x,y]
        control: control parameter that is to be varied
        params: list of parameter values [kb, knb, rnb]
    Output:
        array for subsequent state
    '''
        
    
    [x, y] = state   # x (y) population after breeding (non-breeding) period
    [alpha_b, alpha_nb, rb] = params
    rnb = control
    
    # Compute pop size after breeding period season t+1
    xnew = y * np.exp(rb - a*x - alpha_b * y ) + amp_dem_b*np.sqrt(abs(y))*noise[0] + amp_env_b*y*noise[1]
    # Compute pop size after non-breeding period season t+1
    ynew =  xnew * np.exp(rnb - alpha_nb * xnew ) + amp_dem_nb*np.sqrt(abs(xnew))*noise[2] + amp_env_nb*xnew*noise[3]
    
    # Ouput updated state        
    return np.array([xnew, ynew])
    

# Parameter list
params = [alpha_b, alpha_nb, rb]
 

# Initialise arrays to store time-series data
t = np.arange(t0,tmax,dt)
s = np.zeros([len(t),2])
   
# Set bifurcation parameter b, that increases decreases linearly from bh to bl
b = pd.Series(np.linspace(rnb_h, rnb_l ,len(t)),index=t)
# Time at which bifurcation occurs
tcrit = b[b < rnb_crit].index[1]


# Initial conditions
x0 = rb/alpha_b
y0 = x0 * np.exp(rnb_h - alpha_nb * 0)
s0 = [x0, y0]


## Implement Euler Maryuyama for stocahstic simulation

# Set seed
np.random.seed(seed)

# Initialise a list to collect trajectories
list_traj_append = []

# loop over simulations
print('\nBegin simulations \n')
for j in range(numSims):
    
    
    # Create noise increments (s.d. sqrt(dt))
    dW_burn = np.random.normal(loc=0, scale=np.sqrt(dt), size = [int(tburn/dt),4])
    dW = np.random.normal(loc=0, scale=np.sqrt(dt), size = [len(t),4])
    
    # Run burn-in period on s0
    for i in range(int(tburn/dt)):
        s0 = de_fun(s0, rnb_h, params, dW_burn[i])
        # make sure that state variable remains >= 0 
        s0 = [np.max([k,0]) for k in s0]
    # Initial condition post burn-in period
    s[0] = s0
    
    # Run simulation
    for i in range(len(t)-1):
        s[i+1] = de_fun(s[i], b.iloc[i], params, dW[i])
        # make sure that state variable remains >= 0 
        s[i+1] = [np.max([k,0]) for k in s[i+1]]
            
    # Store series data in a temporary DataFrame- include column for total population count
    data = {'Realisation number': (j+1)*np.ones(len(t)),
                'Time': t,
                'Non-breeding': s[:,0],
                'Breeding': s[:,1],
                'Total': s[:,0] + s[:,1]}
    df_temp = pd.DataFrame(data)
    # Append to list
    list_traj_append.append(df_temp)
    
    print('Simulation '+str(j+1)+' complete')

#  Concatenate DataFrame from each realisation
df_traj = pd.concat(list_traj_append)
df_traj.set_index(['Realisation number','Time'], inplace=True)






# ----------------------
# Execute ews_compute for each realisation
# ---------------------


# Filter time-series to have time-spacing dt2
df_traj_filt = df_traj.loc[::int(dt2/dt)]

# set up a list to store output dataframes from ews_compute- we will concatenate them at the end
appended_ews = []
appended_pspec = []
appended_ktau = []

# loop through realisation number
print('\nBegin EWS computation\n')
for i in range(numSims):
    # loop through variable
    for var in ['Non-breeding','Breeding']:
        
        ews_dic = ewstools.core.ews_compute(df_traj_filt.loc[i+1][var], 
                          roll_window = rw,
                          smooth='Lowess',
                          span=span,
                          lag_times = lags, 
                          ews = ews,
                          ham_length = ham_length,
                          ham_offset = ham_offset,
                          pspec_roll_offset = pspec_roll_offset,
                          upto=tcrit,
                          sweep=False)
        
        # The DataFrame of EWS
        df_ews_temp = ews_dic['EWS metrics']
        # The DataFrame of power spectra
        df_pspec_temp = ews_dic['Power spectrum']
        # The DataFrame of kendall tau values
        df_ktau_temp = ews_dic['Kendall tau']
        
        # Compute cross-correlation
        df_cross_corr = cross_corr(df_traj_filt.loc[i+1][['Non-breeding','Breeding']],
                                   roll_window = rw,
                                   span = span,
                                   upto=tcrit)
        series_cross_corr = df_cross_corr['EWS metrics']['Cross correlation']
        
        
        # Include a column in the DataFrames for realisation number and variable
        df_ews_temp['Realisation number'] = i+1
        df_ews_temp['Variable'] = var
        df_ews_temp['Cross correlation'] = series_cross_corr

        
        df_pspec_temp['Realisation number'] = i+1
        df_pspec_temp['Variable'] = var


        df_ktau_temp['Realisation number'] = i+1
        df_ktau_temp['Variable'] = var
        df_ktau_temp['Cross correlation'] = df_cross_corr['Kendall tau'].iloc[0,0]
                
        
        # Add DataFrames to list
        appended_ews.append(df_ews_temp)
        appended_pspec.append(df_pspec_temp)
        appended_ktau.append(df_ktau_temp)


        
    # Print status every realisation
    if np.remainder(i+1,1)==0:
        print('EWS for realisation '+str(i+1)+' complete')


# Concatenate EWS DataFrames. Index [Realisation number, Variable, Time]
df_ews = pd.concat(appended_ews).reset_index().set_index(['Realisation number','Variable','Time'])

# Concatenate power spectrum DataFrames. Index [Realisation number, Variable, Time, Frequency]
df_pspec = pd.concat(appended_pspec).reset_index().set_index(['Realisation number','Variable','Time','Frequency'])

# Concatenate kendall tau DataFrames. Index [Realisation number, Variable]
df_ktau = pd.concat(appended_ktau).reset_index().set_index(['Realisation number','Variable'])


# Compute ensemble statistics of EWS over all realisations (mean, pm1 s.d.)
df_ews_means = df_ews.mean(level=('Variable','Time'))
df_ews_deviations = df_ews.std(level=('Variable','Time'))



#-------------------------
# Plots to visualise EWS
#-------------------------


# Realisation number to plot
plot_num = 1
## Plot of trajectory, smoothing and EWS of var (x or y)
fig1, axes = plt.subplots(nrows=7, ncols=1, sharex=True, figsize=(4,8))
df_ews.loc[plot_num]['State variable'].unstack(level=0).plot(ax=axes[0],
          title='Early warning signals for a single realisation')
df_ews.loc[plot_num]['Coefficient of variation'].unstack(level=0).plot(ax=axes[1],legend=False)
df_ews.loc[plot_num]['Lag-1 AC'].unstack(level=0).plot(ax=axes[2], legend=False)
df_ews.loc[plot_num]['Lag-2 AC'].unstack(level=0).plot(ax=axes[3], legend=False)
df_ews.loc[plot_num,'Smax/Var'].dropna().unstack(level=0).plot(ax=axes[4], legend=False)
df_ews.loc[plot_num]['Skewness'].unstack(level=0).plot(ax=axes[5], legend=False, xlim=(0,tmax))
df_ews.loc[plot_num]['Kurtosis'].unstack(level=0).plot(ax=axes[6], legend=False, xlim=(0,tmax))

axes[0].set_ylabel('Population')
axes[0].legend(title=None)
axes[1].set_ylabel('CoV')
axes[2].set_ylabel('Lag-1 AC')
axes[3].set_ylabel('Lag-2 AC')
axes[4].set_ylabel('Smax/Var')
axes[5].set_ylabel('Skewness')
axes[6].set_ylabel('Kurtosis')



# Realisation number to plot
plot_num = 2
## Plot of trajectory, smoothing and EWS of var (x or y)
fig1, axes = plt.subplots(nrows=7, ncols=1, sharex=True, figsize=(4,8))
df_ews.loc[plot_num]['State variable'].unstack(level=0).plot(ax=axes[0],
          title='Early warning signals for a single realisation')
df_ews.loc[plot_num]['Coefficient of variation'].unstack(level=0).plot(ax=axes[1],legend=False)
df_ews.loc[plot_num]['Lag-1 AC'].unstack(level=0).plot(ax=axes[2], legend=False)
df_ews.loc[plot_num]['Lag-2 AC'].unstack(level=0).plot(ax=axes[3], legend=False)
df_ews.loc[plot_num,'Smax/Var'].dropna().unstack(level=0).plot(ax=axes[4], legend=False)
df_ews.loc[plot_num]['Skewness'].unstack(level=0).plot(ax=axes[5], legend=False, xlim=(0,tmax))
df_ews.loc[plot_num]['Kurtosis'].unstack(level=0).plot(ax=axes[6], legend=False, xlim=(0,tmax))

axes[0].set_ylabel('Population')
axes[0].legend(title=None)
axes[1].set_ylabel('CoV')
axes[2].set_ylabel('Lag-1 AC')
axes[3].set_ylabel('Lag-2 AC')
axes[4].set_ylabel('Smax/Var')
axes[5].set_ylabel('Skewness')
axes[6].set_ylabel('Kurtosis')


# Realisation number to plot
plot_num = 3
## Plot of trajectory, smoothing and EWS of var (x or y)
fig1, axes = plt.subplots(nrows=7, ncols=1, sharex=True, figsize=(4,8))
df_ews.loc[plot_num]['State variable'].unstack(level=0).plot(ax=axes[0],
          title='Early warning signals for a single realisation')
df_ews.loc[plot_num]['Coefficient of variation'].unstack(level=0).plot(ax=axes[1],legend=False)
df_ews.loc[plot_num]['Lag-1 AC'].unstack(level=0).plot(ax=axes[2], legend=False)
df_ews.loc[plot_num]['Lag-2 AC'].unstack(level=0).plot(ax=axes[3], legend=False)
df_ews.loc[plot_num,'Smax/Var'].dropna().unstack(level=0).plot(ax=axes[4], legend=False)
df_ews.loc[plot_num]['Skewness'].unstack(level=0).plot(ax=axes[5], legend=False, xlim=(0,tmax))
df_ews.loc[plot_num]['Kurtosis'].unstack(level=0).plot(ax=axes[6], legend=False, xlim=(0,tmax))

axes[0].set_ylabel('Population')
axes[0].legend(title=None)
axes[1].set_ylabel('CoV')
axes[2].set_ylabel('Lag-1 AC')
axes[3].set_ylabel('Lag-2 AC')
axes[4].set_ylabel('Smax/Var')
axes[5].set_ylabel('Skewness')
axes[6].set_ylabel('Kurtosis')



# Box plot to visualise kendall tau values
#df_ktau[['Coefficient of variation','Lag-1 AC','Skewness','Cross correlation']].boxplot()



#
#
##------------------------------------
### Export data 
##-----------------------------------
#
## Export 5 single realisations EWS DataFrame
#df_ews.loc[1:5].to_csv('data_export/'+dir_name+'/ews_singles.csv')
#
## Export power spectra of first 5 realisations
#df_pspec.loc[1:5].to_csv('data_export/'+dir_name+'/pspec.csv')
#
## Export aggregates
#df_ews_means.to_csv('data_export/'+dir_name+'/ews_means.csv')
#df_ews_deviations.to_csv('data_export/'+dir_name+'/ews_deviations.csv')
#
## Export Kendall tau values
#df_ktau.to_csv('data_export/'+dir_name+'/ktau.csv')


