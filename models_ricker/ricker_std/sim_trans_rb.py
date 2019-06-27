#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:41:47 2018

@author: Thomas Bury

Simulate transient simulations of the Ricker model (with dem and env noise) going to 
extinction as the growth rate goes below 1.

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
dir_name = 'ricker_trans_rext'

if not os.path.exists('data_export/'+dir_name):
    os.makedirs('data_export/'+dir_name)


#--------------------------------
# Global parameters
#–-----------------------------


# Simulation parameters
dt = 1 # time-step (must be 1 since discrete-time system)
t0 = 0
tmax = 400
tburn = 100 # burn-in period
numSims = 5
seed = 1 # random number generation seed


# EWS parameters
dt2 = 1 # spacing between time-series for EWS computation
rw = 0.4 # rolling window
span = 0.5 # Lowess span
lags = [1,2,3] # autocorrelation lag times
ews = ['var','ac','sd','cv','skew','kurt','smax','smax/var','smax/mean'] # EWS to compute
ham_length = 80 # number of data points in Hamming window
ham_offset = 0.5 # proportion of Hamming window to offset by upon each iteration
pspec_roll_offset = 10 # offset for rolling window when doing spectrum metrics


#----------------------------------
# Simulate many (transient) realisations
#----------------------------------



# Model parameters
    
gamma = 0.01   	# Strength of density dependent effects
amp_dem = 0.1    # Demographic noise amplitude
amp_env = 0.1	# Environmental noise amplitude


# Bifurcation parameter
rl = -0.2
rh = 1.2
rcrit = 0

# Function dynamic - outputs the subsequent state
def de_fun(state, control, noise):
    '''
    Inputs:
        state: array of state variables [x,y]
        control: control parameter that is to be varied
        params: list of parameter values [kb, knb, rnb, a, eps1, eps2]
    Output:
        array for subsequent state
    '''
        
    
    x = state   # x population size
    [eps_dem, eps_env] = noise
    r = control
    
    # Compute pop size after breeding period season t+1
    xnew = x * np.exp(r - gamma * x) + eps_dem*np.sqrt(x) + eps_env*x
    
    # Ouput updated state        
    return xnew
    
 

# Initialise arrays to store time-series data
t = np.arange(t0,tmax,dt)
s = np.zeros(len(t))
   
# Set bifurcation parameter b, that increases linearly in time from bl to bh
b = pd.Series(np.linspace(rh, rl, len(t)),index=t)
# Time at which bifurcation occurs
tcrit = b[b < rcrit].index[1]


# Initial conditions
x0 = rh/gamma
s0 = x0


## Implement Euler Maryuyama for stocahstic simulation

# Set seed
np.random.seed(seed)

# Initialise a list to collect trajectories
list_traj_append = []

# loop over simulations
print('\nBegin simulations \n')
for j in range(numSims):
    
    
    # Create brownian increments (s.d. sigma*sqrt(dt) )
    dW_burn = np.random.normal(loc=0, scale=np.sqrt(dt), size = (int(tburn/dt),2))*np.array([amp_dem,amp_env])
    dW = np.random.normal(loc=0, scale=np.sqrt(dt), size = (len(t),2))*np.array([amp_dem,amp_env])
    
    # Run burn-in period on s0
    for i in range(int(tburn/dt)):
        s0 = de_fun(s0, rh, dW_burn[i])
        
    # Initial condition post burn-in period
    s[0] = s0
    
    # Run simulation
    for i in range(len(t)-1):
        s[i+1] = de_fun(s[i], b.iloc[i], dW[i])
        # make sure that state variable remains >= 0 
        s[i+1] = np.max([s[i+1],0]) 
            
    # Store series data in a temporary DataFrame
    data = {'Realisation number': (j+1)*np.ones(len(t)),
                'Time': t,
                'x': s}
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
    for var in ['x']:
        
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
        # The DataFrame of power spectra
        df_pspec_temp = ews_dic['Power spectrum']
        # The DataFrame of kendall tau values
        df_ktau_temp = ews_dic['Kendall tau']
       
        
        # Include a column in the DataFrames for realisation number and variable
        df_ews_temp['Realisation number'] = i+1
        df_ews_temp['Variable'] = var

        
        df_pspec_temp['Realisation number'] = i+1
        df_pspec_temp['Variable'] = var


        df_ktau_temp['Realisation number'] = i+1
        df_ktau_temp['Variable'] = var
                
        
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


## Compute ensemble statistics of EWS over all realisations (mean, pm1 s.d.)
#ews_names = ['Variance', 'Lag-1 AC', 'Lag-2 AC', 'Lag-4 AC', 'AIC fold', 'AIC hopf', 'AIC null', 'Coherence factor']

#df_ews_means = df_ews[ews_names].mean(level='Time')
#df_ews_deviations = df_ews[ews_names].std(level='Time')



#-------------------------
# Plots to visualise EWS
#-------------------------

# Realisation number to plot
plot_num = 1
var = 'x'
## Plot of trajectory, smoothing and EWS of var (x or y)
fig1, axes = plt.subplots(nrows=6, ncols=1, sharex=True, figsize=(6,8))
df_ews.loc[plot_num,var][['State variable','Smoothing']].plot(ax=axes[0],
          title='Early warning signals for a single realisation')
df_ews.loc[plot_num,var][['Variance']].plot(ax=axes[1], legend=True)
df_ews.loc[plot_num,var]['Coefficient of variation'].plot(ax=axes[2],legend=True)
df_ews.loc[plot_num,var][['Lag-1 AC']].plot(ax=axes[2], secondary_y=True,legend=True)
df_ews.loc[plot_num,var]['Smax/Var'].dropna().plot(ax=axes[3],legend=True)
df_ews.loc[plot_num,var]['Smax/Mean'].dropna().plot(ax=axes[4],legend=True)
df_ews.loc[plot_num,var]['Skewness'].plot(ax=axes[5],legend=True)

#plt.savefig('figures/ews_trans_r_ext.png')




### Define function to make grid plot for evolution of the power spectrum in time
#def plot_pspec_grid(tVals, plot_num, var):
#    
#    g = sns.FacetGrid(df_pspec.loc[plot_num,var].loc[t_display].reset_index(), 
#                  col='Time',
#                  col_wrap=3,
#                  sharey=False,
#                  aspect=1.5,
#                  height=1.8
#                  )
#
#    g.map(plt.plot, 'Frequency', 'Empirical', color='k', linewidth=2)
#    g.map(plt.plot, 'Frequency', 'Fit fold', color='b', linestyle='dashed', linewidth=1)
#    g.map(plt.plot, 'Frequency', 'Fit hopf', color='r', linestyle='dashed', linewidth=1)
#    g.map(plt.plot, 'Frequency', 'Fit null', color='g', linestyle='dashed', linewidth=1)
#    # Axes properties
#    axes = g.axes
#    # Set y labels
#    for ax in axes[::3]:
#        ax.set_ylabel('Power')
#        # Set y limit as max power over all time
#        for ax in axes:
#            ax.set_ylim(top=1.05*max(df_pspec.loc[plot_num,var]['Empirical']), bottom=0)
#       
#    return g

##  Choose time values at which to display power spectrum
#t_display = df_pspec.index.levels[2][::3].values
#
#plot_pspec = plot_pspec_grid(t_display, plot_num, 'x')




## Box plot to visualise kendall tau values
#df_ktau[['Variance','Lag-1 AC','Lag-2 AC','Smax','Cross correlation']].boxplot()
#
#



#
#
#------------------------------------
## Export data 
#-----------------------------------



# Export EWS DataFrame
df_ews.to_csv('data_export/'+dir_name+'/ews.csv')

# Export power spectra
df_pspec.to_csv('data_export/'+dir_name+'/pspec.csv')



    






