#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 18:04:51 2019


Code to simulate seasonal Ricker model at different values of rb
Stationary simulations (fixed paramters).

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
dir_name = 'ews_stat'

if not os.path.exists('data_export/'+dir_name):
    os.makedirs('data_export/'+dir_name)




#--------------------------------
# Global parameters
#–-----------------------------


# Simulation parameters
dt = 1
t0 = 0
tmax = 400 # make large (to get idealised statistics from stationary distribution)
tburn = 100 # burn-in period
seed = 0 # random number generation seed
rbif = 2.39 # flip bifurcation (from MMA bif file)
rbl = 0 # low r value
rbh = 2 # high r value
rinc = 0.05 # amount to increment r by



# EWS parameters
dt2 = 1 # spacing between time-series for EWS computation
rw = 1 # rolling window (compute EWS using full time-series)
span = 1 # span (take the whole dataset as stationary)
lags = [1,2,3] # autocorrelation lag times
ews = ['var','ac','sd','cv','skew','kurt'] # EWS to compute
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
    [kb, knb, rnb] = params
    rb = control
    
    # Compute pop size after breeding period season t+1
    xnew = y * np.exp(rb - alpha_b * y ) + amp_dem_x*np.sqrt(y)*noise[0] + amp_env_x*y*noise[1]
    # Compute pop size after non-breeding period season t+1
    ynew =  xnew * np.exp(rnb - alpha_nb * xnew ) + amp_dem_y*np.sqrt(abs(xnew))*noise[2] + amp_env_y*xnew*noise[3]
    
    # Ouput updated state        
    return np.array([xnew, ynew])
    
    
   
# System parameters
    
alpha_b = 1/224 # carrying capacity in breeding period
alpha_nb = 1/84.52 # carrying capacity in non-breeding period
rnb = -0.0568 # growth rate in non-breeding period


# Parameter list
params = [alpha_b, alpha_nb, rnb]

# Control parameter values
rVals = np.arange(rbl, rbh, rinc)





# Initial conditions
x0 = rbh/alpha_b
y0 = x0 * np.exp(rnb - alpha_b*x0)




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

    
    # Create noise increments (s.d. sqrt(dt))
    dW_burn = np.random.normal(loc=0, scale=np.sqrt(dt), size = [int(tburn/dt),4])
    dW = np.random.normal(loc=0, scale=np.sqrt(dt), size = [len(t),4])

 
    # IC as a state vector
    s0 = np.array([x0 , y0])
    
    # Run burn-in period on initial condition
    for i in range(int(tburn/dt)):
        # Iterate
        s0 = de_fun(s0, r, params, dW_burn[i])
        # Make sure that state variable remains >= 0 
        s0 = [np.max([k,0]) for k in s0]
        
        
    # Initial condition post burn-in period
    s[0]=s0
    
    # Run simulation
    for i in range(len(t)-1):
        s[i+1] = de_fun(s[i], r, params, dW[i])
        # make sure that state variable remains >= 0 
        s[i+1] = [np.max([k,0]) for k in s[i+1]]
            
    # Store series data in a DataFrame
    data = {'Growth rate': r,
                'Time': t,
                'Non-breeding pop': s[:,0],
                'Breeding pop': s[:,1]}
    df_temp = pd.DataFrame(data)
    # Append to list
    list_traj_append.append(df_temp)
    
    print('Simulation with r='+str(r)+' complete')

#  Concatenate DataFrame from each realisation
df_traj = pd.concat(list_traj_append)
df_traj.set_index(['Growth rate','Time'], inplace=True)


# Coarsen time-series to have spacing dt2 (for EWS computation)
df_traj_filt = df_traj.loc[::int(dt2/dt)]




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
    for var in ['Non-breeding pop', 'Breeding pop']:
        
        ews_dic = ewstools.ews_compute(df_traj_filt.loc[r][var], 
                          roll_window = rw,
                          smooth = 'Lowess',
                          span = span,
                          lag_times = lags, 
                          ews = ews,
                          ham_length = ham_length,
                          ham_offset = ham_offset
                          )
        
        # The DataFrame of EWS
        df_ews_temp = ews_dic['EWS metrics']
#        # The DataFrame of power spectra
#        df_pspec_temp = ews_dic['Power spectrum']
        
        # Include a column in the DataFrames for r value and variable
        df_ews_temp['Growth rate'] = r
        df_ews_temp['Variable'] = var
        
#        df_pspec_temp['Growth rate'] = r
#        df_pspec_temp['Variable'] = var
                
        # Add DataFrames to list
        appended_ews.append(df_ews_temp)
#        appended_pspec.append(df_pspec_temp)
        
    # Print status every realisation
    print('EWS for r =  '+str(r)+' complete')


# Concatenate EWS DataFrames. Index [Growth rate, Variable, Time]
df_ews_full = pd.concat(appended_ews).reset_index().set_index(['Growth rate','Variable','Time'])
# Concatenate power spectrum DataFrames. Index [Realisation number, Variable, Time, Frequency]
#df_pspec = pd.concat(appended_pspec).reset_index().set_index(['Growth rate','Variable','Time','Frequency'])


# Refine DataFrame to just have EWS data (no time dependence)
df_ews = df_ews_full.dropna().reset_index(level=2, drop=True).reorder_levels(['Variable', 'Growth rate'])
#df_pspec = df_pspec.reset_index(level=2, drop=True).reorder_levels(['Variable', 'Growth rate','Frequency'])



#--------------------------
## Grid plot of some trajectories
#--------------------------

# Set up frame and axes
g = sns.FacetGrid(df_ews_full.loc[rVals[0:-1:(int(len(rVals)/8))]].reset_index(), 
                  col='Growth rate',
                  hue='Variable',
                  palette='Set1',
                  col_wrap=2,
                  sharey=False,
                  aspect=1.5,
                  height=1.8
                  )
# Set plot title size
plt.rc('axes', titlesize=10)
# Plot state variable
g.map(plt.plot, 'Time', 'State variable', linewidth=1)
# Plot smoothing
g.map(plt.plot, 'Time', 'Smoothing', color='tab:orange', linewidth=1)
# Axes properties
axes = g.axes
# Assign plot label
plot_traj = g

## Axes properties
#axes = g.axes
#for i in range(len(axes)):
#    ax=axes[i]
#    r=rVals[i]
#    ax.set_ylim(bottom=0, top=60)




#----------------
## Plots of EWS against r value
#----------------

# Plot of EWS metrics
fig1, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(6,6))
df_ews.loc['Non-breeding pop'][['Variance']].plot(ax=axes[0],title='Early warning signals')
df_ews.loc['Breeding pop'][['Variance']].plot(ax=axes[0],secondary_y=True)
df_ews.loc['Non-breeding pop'][['Coefficient of variation']].plot(ax=axes[1])
df_ews.loc['Breeding pop'][['Coefficient of variation']].plot(ax=axes[1],secondary_y=True)
df_ews.loc['Non-breeding pop'][['Lag-1 AC']].plot(ax=axes[2])
df_ews.loc['Breeding pop'][['Lag-1 AC']].plot(ax=axes[2],secondary_y=True)
df_ews.loc['Non-breeding pop'][['Skewness']].plot(ax=axes[3])
df_ews.loc['Breeding pop'][['Skewness']].plot(ax=axes[3],secondary_y=True)






#
#
##---------------------------------
### Power spectra visualisation
##--------------------------------
#
## Limits for x-axis
#xmin = -np.pi
#xmax = np.pi
#
### Post-breeding population
#var = 'Post-breeding pop'
#g = sns.FacetGrid(df_pspec.loc[var].loc[rVals[0:-1:(int(len(rVals)/4))]].reset_index(level=['Growth rate','Frequency']), 
#                  col='Growth rate',
#                  col_wrap=3,
#                  sharey=False,
#                  aspect=1.5,
#                  height=1.8
#                  )
## Plots
#plt.rc('axes', titlesize=10) 
#g.map(plt.plot, 'Frequency', 'Empirical', color='k', linewidth=1)
#g.map(plt.plot, 'Frequency', 'Fit null', color='g', linestyle='dashed', linewidth=1)
#g.map(plt.plot, 'Frequency', 'Fit fold', color='b', linestyle='dashed', linewidth=1)
#g.map(plt.plot, 'Frequency', 'Fit hopf', color='r', linestyle='dashed', linewidth=1)
#
## Axes properties
#axes = g.axes
## Global axes properties
#for i in range(len(axes)):
#    ax=axes[i]
#    r=rVals[i]
##    ax.set_ylim(bottom=0, top=1.1*max(df_pspec.loc[var,d]['Empirical'].loc[xmin:xmax].dropna()))
#    ax.set_xlim(left=xmin, right=xmax)
#    ax.set_xticks([-3,-2,-1,0,1,2,3])
#    ax.set_title('r = %.2f' % rVals[i])
#    # AIC weights
#    xpos=0.7
#    ypos=0.9
#    ax.text(xpos,ypos,
#            '$w_f$ = %.1f' % df_ews.loc[var,r]['AIC fold'],
#            fontsize=9,
#            color='b',
#            transform=ax.transAxes)  
#    ax.text(xpos,ypos-0.12,
#            '$w_h$ = %.1f' % df_ews.loc[var,r]['AIC hopf'],
#            fontsize=9,
#            color='r',
#            transform=ax.transAxes)
#    ax.text(xpos,ypos-2*0.12,
#            '$w_n$ = %.1f' % df_ews.loc[var,r]['AIC null'],
#            fontsize=9,
#            color='g',
#            transform=ax.transAxes)
## Y labels
#for ax in axes[::3]:
#    ax.set_ylabel('Power')
#    
### Specific Y limits
##for ax in axes[:4]:
##    ax.set_ylim(top=0.004)
##for ax in axes[6:9]:
##    ax.set_ylim(top=0.25)
## Assign to plot label
#pspec_plot_breeding=g
#
#
#
## Limits for x-axis
#xmin = -np.pi
#xmax = np.pi
#
### Post-non-breeding population
#var = 'Post-non-breeding pop'
#g = sns.FacetGrid(df_pspec.loc[var].loc[rVals[0:-1:(int(len(rVals)/4))]].reset_index(level=['Growth rate','Frequency']), 
#                  col='Growth rate',
#                  col_wrap=3,
#                  sharey=False,
#                  aspect=1.5,
#                  height=1.8
#                  )
## Plots
#plt.rc('axes', titlesize=10) 
#g.map(plt.plot, 'Frequency', 'Empirical', color='k', linewidth=1)
#g.map(plt.plot, 'Frequency', 'Fit null', color='g', linestyle='dashed', linewidth=1)
#g.map(plt.plot, 'Frequency', 'Fit fold', color='b', linestyle='dashed', linewidth=1)
#g.map(plt.plot, 'Frequency', 'Fit hopf', color='r', linestyle='dashed', linewidth=1)
#
## Axes properties
#axes = g.axes
## Global axes properties
#for i in range(len(axes)):
#    ax=axes[i]
#    r=rVals[i]
##    ax.set_ylim(bottom=0, top=1.1*max(df_pspec.loc[var,d]['Empirical'].loc[xmin:xmax].dropna()))
#    ax.set_xlim(left=xmin, right=xmax)
#    ax.set_xticks([-3,-2,-1,0,1,2,3])
#    ax.set_title('r = %.2f' % rVals[i])
#    # AIC weights
#    xpos=0.7
#    ypos=0.9
#    ax.text(xpos,ypos,
#            '$w_f$ = %.1f' % df_ews.loc[var,r]['AIC fold'],
#            fontsize=9,
#            color='b',
#            transform=ax.transAxes)  
#    ax.text(xpos,ypos-0.12,
#            '$w_h$ = %.1f' % df_ews.loc[var,r]['AIC hopf'],
#            fontsize=9,
#            color='r',
#            transform=ax.transAxes)
#    ax.text(xpos,ypos-2*0.12,
#            '$w_n$ = %.1f' % df_ews.loc[var,r]['AIC null'],
#            fontsize=9,
#            color='g',
#            transform=ax.transAxes)
## Y labels
#for ax in axes[::3]:
#    ax.set_ylabel('Power')
#    
### Specific Y limits
##for ax in axes[:4]:
##    ax.set_ylim(top=0.004)
##for ax in axes[6:9]:
##    ax.set_ylim(top=0.25)
## Assign to plot label
#pspec_plot_nonbreeding=g
#
#
#
##------------------------------------
### Export data / figures
##-----------------------------------
#
#
### Export EWS data
#
## Post-breeding season EWS DataFrame
#df_ews_x = df_ews.loc['Post-breeding pop']
#df_ews_x.to_csv('data_export/'+dir_name+'/ews_x.csv')
#
## Post non-breeding season EWS DataFrame
#df_ews_y = df_ews.loc['Post-non-breeding pop']
#df_ews_y.to_csv('data_export/'+dir_name+'/ews_y.csv')
#
#
#### Export power spectrum (empirical data)
##
### Chlorella pspecs
##df_pspec_chlor = df_pspec.loc['Chlorella','Empirical'].dropna()
##df_pspec_chlor.to_csv('data_export/'+dir_name+'/pspec_chlor.csv')
##
##
### Brachionus pspecs
##df_pspec_brach = df_pspec.loc['Brachionus', 'Empirical'].dropna()
##df_pspec_brach.to_csv('data_export/'+dir_name+'/pspec_brach.csv')
#
#


