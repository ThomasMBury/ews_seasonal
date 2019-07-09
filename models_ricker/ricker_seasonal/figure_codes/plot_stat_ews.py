#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 18:04:51 2019


Plot stationary EWS in (rb, rnb) of seasonal Ricker model

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




# Import data
df_ews = pd.read_csv('../data_export/ews_stat_rvals/ews.csv', index_col='Variable')
df_equi = pd.read_csv('../data_export/equi_data/equi_data1.csv')



# Join data into single DataFrames for breeding and non-breeding population

df_temp = df_equi.merge(df_ews.loc['Breeding'], on=('rb', 'rnb'))







#-------------------
# Plot equilibrium values
#–-------------------

# Remove non-numeric entries (text like "period-2 oscillations")
df_equi['x_num'] = pd.to_numeric(df_equi['x'], errors='coerce')
df_equi['y_num'] = pd.to_numeric(df_equi['y'], errors='coerce')


df_plot_x = df_equi.pivot(index='rb', columns='rnb', values='x_num').iloc[::-1]
df_plot_y = df_equi.pivot(index='rb', columns='rnb', values='y_num').iloc[::-1]




# Plot for breeding population
plt.figure(figsize=(3,3))
ax = plt.axes()
sns.heatmap(df_plot_x, cmap='RdYlGn', vmin=0, vmax=500, ax=ax, 
            xticklabels=5,
            yticklabels=5)
ax.set_title('Non-breedinge population: size')
plt.show()

# Plot for non-breeding population
plt.figure(figsize=(3,3))
ax = plt.axes()
sns.heatmap(df_plot_y, cmap='RdYlGn', vmin=0, ax=ax,
            xticklabels=5,
            yticklabels=5)
ax.set_title('Breeding population: size')
plt.show()




#----------------
# Plot EWS
#---------------

## Variance

# Breeding population variance
df_plot = df_ews.loc['Breeding'].reset_index().round({'rb':2, 'rnb':2}).pivot(index='rb', columns='rnb', values='Variance').iloc[::-1]
plt.figure(figsize=(3,3))
ax = plt.axes()
sns.heatmap(df_plot, cmap='RdYlGn', vmin=0, vmax=200, ax=ax, 
            xticklabels=1,
            yticklabels=1)
ax.set_title('Breeding population: Variance')
plt.show()

# Non-breeding population variance
df_plot = df_ews.loc['Non-breeding'].reset_index().round({'rb':2, 'rnb':2}).pivot(index='rb', columns='rnb', values='Variance').iloc[::-1]
plt.figure(figsize=(3,3))
ax = plt.axes()
sns.heatmap(df_plot, cmap='RdYlGn', vmin=0, vmax=200, ax=ax, 
            xticklabels=1,
            yticklabels=1)
ax.set_title('Non-breeding population: Variance')
plt.show()




## Coefficient of variation


# Breeding population
df_plot = df_ews.loc['Breeding'].reset_index().round({'rb':2, 'rnb':2}).pivot(index='rb', columns='rnb', values='Coefficient of variation').iloc[::-1]
plt.figure(figsize=(3,3))
ax = plt.axes()
sns.heatmap(df_plot, cmap='RdYlGn', ax=ax, 
            xticklabels=1,
            yticklabels=1)
ax.set_title('Breeding population: Coefficient of variation')
plt.show()

# Non-breeding population
df_plot = df_ews.loc['Non-breeding'].reset_index().round({'rb':2, 'rnb':2}).pivot(index='rb', columns='rnb', values='Coefficient of variation').iloc[::-1]
plt.figure(figsize=(3,3))
ax = plt.axes()
sns.heatmap(df_plot, cmap='RdYlGn', ax=ax, 
            xticklabels=1,
            yticklabels=1)
ax.set_title('Non-breeding population: Coefficient of variation')
plt.show()




## Smax/Var


# Breeding population
df_plot = df_ews.loc['Breeding'].reset_index().round({'rb':2, 'rnb':2}).pivot(index='rb', columns='rnb', values='Smax/Var').iloc[::-1]
plt.figure(figsize=(3,3))
ax = plt.axes()
sns.heatmap(df_plot, cmap='RdYlGn', ax=ax, vmax=0.6,
            xticklabels=1,
            yticklabels=1)
ax.set_title('Breeding population: Smax/Var')
plt.show()

# Non-breeding population
df_plot = df_ews.loc['Non-breeding'].reset_index().round({'rb':2, 'rnb':2}).pivot(index='rb', columns='rnb', values='Smax/Var').iloc[::-1]
plt.figure(figsize=(3,3))
ax = plt.axes()
sns.heatmap(df_plot, cmap='RdYlGn', ax=ax, 
            xticklabels=1,
            yticklabels=1)
ax.set_title('Non-breeding population: Smax/Var')
plt.show()



## Lag-1 AC




# Breeding population
df_plot = df_ews.loc['Breeding'].reset_index().round({'rb':2, 'rnb':2}).pivot(index='rb', columns='rnb', values='Lag-1 AC').iloc[::-1]
plt.figure(figsize=(3,3))
ax = plt.axes()
sns.heatmap(df_plot, cmap='RdYlGn', ax=ax, 
            xticklabels=1,
            yticklabels=1)
ax.set_title('Breeding population: Lag-1 AC')
plt.show()

# Non-breeding population
df_plot = df_ews.loc['Non-breeding'].reset_index().round({'rb':2, 'rnb':2}).pivot(index='rb', columns='rnb', values='Lag-1 AC').iloc[::-1]
plt.figure(figsize=(3,3))
ax = plt.axes()
sns.heatmap(df_plot, cmap='RdYlGn', ax=ax, 
            xticklabels=1,
            yticklabels=1)
ax.set_title('Non-breeding population: Lag-1 AC')
plt.show()









