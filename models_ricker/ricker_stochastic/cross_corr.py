#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:07:21 2019

Function to compute cross correlation between two variables over a rolling window

@author: Thomas Bury
"""



#---------------------------------
# Import relevant packages
#--------------------------------

# For numeric computation and DataFrames
import numpy as np
import pandas as pd



# For detrending time-series
from statsmodels.nonparametric.smoothers_lowess import lowess


# Test cross_corr

# Create a DataFrame of two time-series
tVals = np.arange(0,10,0.1)
xVals = 5 + np.random.normal(0,1,(len(tVals),2))
df_series = pd.DataFrame(xVals, index=tVals)



def cross_corr(df_series,
            roll_window=0.4,
            span=0.1,
            upto='Full'):
    '''
    Compute cross correlation between two time-series
    	
    Args
    ----
    df_series: pd.DataFrame
        Time-series data to analyse. Indexed by time. Two columns
    roll_window: float
        Rolling window size as a proportion of the length of the time-series 
        data.
    span: float
        Span of time-series data used for Lowess filtering. Taken as a 
        proportion of time-series length if in (0,1), otherwise taken as 
        absolute.
    upto: int or 'Full'
        Time up to which EWS are computed. Enter 'Full' to use
        the entire time-series. Otherwise enter a time value.
    
    Returns
    --------
    dict of pd.DataFrames:
        A dictionary with the following entries.
        **'EWS metrics':** A DataFrame indexed by time with columns corresopnding 
        to each EWS.
        **'Kendall tau':** A DataFrame of the Kendall tau values for each EWS metric.
    '''
    
    
    
    # Initialise a DataFrame to store EWS data
    df_ews = pd.DataFrame()
    df_ews['State 1'] = df_series.iloc[:,0]
    df_ews['State 2'] = df_series.iloc[:,1]
    df_ews.index = df_series.index
    df_ews.index.rename('Time', inplace=True)
    
    
    
    # Portion of time-series for EWS computation
    if upto == 'Full':
        df_short_series = df_ews
    else: df_short_series = df_ews.loc[:upto]


    #------Data detrending--------
       
    # Compute the Lowess span as a proportion if given as absolute
    if not 0 < span <= 1:
        span = span/df_short_series.shape[0]
    else:
        span = span
    
    
    # Smooth time-series and compute residuals
    for var in [1,2]:
        smooth_data = lowess(df_short_series['State '+str(var)].values, 
                             df_short_series.index.values, frac=span)[:,1]
        smooth_series = pd.Series(smooth_data, index=df_short_series.index)
        
        residuals = df_short_series['State '+str(var)].values - smooth_data
        resid_series = pd.Series(residuals, index=df_short_series.index)
        
        # Add smoothed data and residuals to the EWS DataFrame
        df_ews['Trend '+str(var)] = smooth_series
        df_ews['Residuals '+str(var)] = resid_series
        
    
    # Compute the rolling window size (integer value)
    rw_size=int(np.floor(roll_window * df_series.shape[0]))
    
    
    # Compute cross correlation between residual time-series
    cross_cor = df_ews[['Residuals 1']].rolling(window=rw_size).corr(df_ews['Residuals 2'])['Residuals 1']
    
    # Add to EWS DataFrame
    df_ews['Cross correlation'] = cross_cor
    
    
        
        
    
    #------------Compute Kendall tau coefficients----------------#
    
    ''' In this section we compute the kendall correlation coefficients for each EWS
        with respect to time. Values close to one indicate high correlation (i.e. EWS
        increasing with time), values close to zero indicate no significant correlation,
        and values close to negative one indicate high negative correlation (i.e. EWS
        decreasing with time).'''
        
                                                                             
    # Put time values as their own series for correlation computation
    time_vals = pd.Series(df_ews.index, index=df_ews.index)

    
    # Find Kendall tau for each EWS and store in a DataFrame
    dic_ktau = {x:df_ews[x].corr(time_vals, method='kendall') for x in ['Cross correlation']} # temporary dictionary
    df_ktau = pd.DataFrame(dic_ktau, index=[0]) # DataFrame (easier for concatenation purposes)
                                                                             
                                                                             
 
    #-------------Organise final output and return--------------#
       
    # Ouptut a dictionary containing EWS DataFrame, power spectra DataFrame, and Kendall tau values
    output_dic = {'EWS metrics': df_ews, 'Kendall tau': df_ktau}
        
    return output_dic









