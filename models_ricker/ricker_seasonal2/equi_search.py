#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:12:16 2019

@author: Thomas Bury

Script to compute stable equilbria of seasonal Ricker model
"""


import numpy as np
from scipy.optimize import fsolve



# Set density-dep parameters 
ab = 1/500
anb = 1/500

# Set list of values for rb and rnb to loop over
rbVals = np.arange(0,1,0.1)
rnbVals = np.arange(-1,0,0.1)

# temp
rb=1
rnb=-0.5



# Equations to solve for equilibrium
def equations(p):
    x, y = p
    return (x-y*np.exp(rb-ab*y), y-x*np.exp(rnb-anb*x))

# Initial guess
(x0,y0) = (400,400)

# Solve equations
x, y =  fsolve(equations, (x0,y0))



print(equations((x, y)))

