#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:41:47 2018

@author: Thomas Bury

Simulate transient simulations of the seasonal Ricker model undergoing the
Flip bifurcation as rb is increased.

"""

# import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# import ewstools
from ewstools import ewstools

# import cross correlation function
from cross_corr import cross_corr


#---------------------
# Directory for data output
#–----------------------

# Name of directory within data_export
dir_name = 'ricker_trans_rnb'

if not os.path.exists('data_export/'+dir_name):
    os.makedirs('data_export/'+dir_name)


#--------------------------------
# Global parameters
#–-----------------------------


# Simulation parameters
dt = 1 # time-step (must be 1 since discrete-time system)
t0 = 0
tmax = 200
tburn = 100 # burn-in period
numSims = 1
seed = 0 # random number generation seed


# EWS parameters
dt2 = 1 # spacing between time-series for EWS computation
rw = 0.4 # rolling window
span = 0.5 # Lowess span
lags = [1,2,3] # autocorrelation lag times
ews = ['var','ac','sd','cv','skew','kurt'] # EWS to compute
ham_length = 80 # number of data points in Hamming window
ham_offset = 0.5 # proportion of Hamming window to offset by upon each iteration
pspec_roll_offset = 20 # offset for rolling window when doing spectrum metrics


#----------------------------------
# Simulate many (transient) realisations
#----------------------------------



# Model parameters
    
rb = 1     # Growth rate for breeding period
alpha_b = 1/500 # density dependent effects in breeding period
alpha_nb = 1/500 # density dependent effects in non-breeding period
a = 0.001     # Effect of non-breeding density on breeding output (COE)

# Noise parameters
amp_dem_b = 0 # amplitude of demographic noise
amp_dem_nb = 0
amp_env_b = 0.1 # amplitude of environmental noise
amp_env_nb = 0.1



# Bifurcation parameter
rnb_l = -1.2
rnb_h = 0
rnb_crit = -1


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
    xnew = y * np.exp(rb - alpha_b * y ) + amp_dem_b*np.sqrt(y)*noise[0] + amp_env_b*y*noise[1]
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
        
    # Initial condition post burn-in period
    s[0] = s0
    
    # Run simulation
    for i in range(len(t)-1):
        s[i+1] = de_fun(s[i], b.iloc[i], params, dW[i])
        # make sure that state variable remains >= 0 
        s[i+1] = [np.max([k,0]) for k in s[i+1]]
            
    # Store series data in a temporary DataFrame
    data = {'Realisation number': (j+1)*np.ones(len(t)),
                'Time': t,
                'Non-breeding': s[:,0],
                'Breeding': s[:,1]}
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
appended_ktau = []

# loop through realisation number
print('\nBegin EWS computation\n')
for i in range(numSims):
    # loop through variable
    for var in ['Non-breeding','Breeding']:
        
        ews_dic = ewstools.ews_compute(df_traj_filt.loc[i+1][var], 
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


        df_ktau_temp['Realisation number'] = i+1
        df_ktau_temp['Variable'] = var
        df_ktau_temp['Cross correlation'] = df_cross_corr['Kendall tau'].iloc[0,0]
                
        
        # Add DataFrames to list
        appended_ews.append(df_ews_temp)
        appended_ktau.append(df_ktau_temp)


        
    # Print status every realisation
    if np.remainder(i+1,1)==0:
        print('EWS for realisation '+str(i+1)+' complete')


# Concatenate EWS DataFrames. Index [Realisation number, Variable, Time]
df_ews = pd.concat(appended_ews).reset_index().set_index(['Realisation number','Variable','Time'])
# Concatenate kendall tau DataFrames. Index [Realisation number, Variable]
df_ktau = pd.concat(appended_ktau).reset_index().set_index(['Realisation number','Variable'])


## Compute ensemble statistics of EWS over all realisations (mean, pm1 s.d.)
#ews_names = ['Variance', 'Lag-1 AC', 'Lag-2 AC', 'Lag-4 AC', 'AIC fold', 'AIC hopf', 'AIC null', 'Coherence factor']

#df_ews_means = df_ews[ews_names].mean(level='Time')
#df_ews_deviations = df_ews[ews_names].std(level='Time')



#-------------------------
# Plots to visualise EWS
#-------------------------

# Realisation number to plot
plot_num = 1
## Plot of trajectory, smoothing and EWS of var (x or y)
fig1, axes = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(6,8))
df_ews.loc[plot_num]['State variable'].unstack(level=0).plot(ax=axes[0],
          title='Early warning signals for a single realisation')
df_ews.loc[plot_num]['Coefficient of variation'].unstack(level=0).plot(ax=axes[1],legend=False)
df_ews.loc[plot_num]['Lag-1 AC'].unstack(level=0).plot(ax=axes[2], legend=False)
df_ews.loc[plot_num]['Skewness'].dropna().unstack(level=0).plot(ax=axes[3], legend=False)
df_ews.loc[plot_num,'Breeding']['Cross correlation'].plot(ax=axes[4], legend=False)

axes[0].set_ylabel('Population')
axes[0].legend(title=None)
axes[1].set_ylabel('CoV')
axes[2].set_ylabel('Lag-1 AC')
axes[3].set_ylabel('Skewness')
axes[4].set_ylabel('Cross correlation')


## Box plot to visualise kendall tau values
#df_ktau[['Coefficient of variation','Lag-1 AC','Skewness','Cross correlation']].boxplot()
#




#
#
##------------------------------------
### Export data 
##-----------------------------------
#
### Export power spectrum evolution (grid plot)
##plot_pspec.savefig('figures/pspec_evol.png', dpi=200)
#
#
#
### Export the first 5 realisations to see individual behaviour
## EWS DataFrame (includes trajectories)
#df_ews.loc[:5].to_csv('data_export/'+dir_name+'/ews_singles.csv')
## Power spectrum DataFrame (only empirical values)
#df_pspec.loc[:5,'Empirical'].dropna().to_csv('data_export/'+dir_name+'/pspecs.csv',
#            header=True)
#
#
## Export kendall tau values
#df_ktau.to_csv('data_export/'+dir_name+'/ktau.csv')
#
#
#    






