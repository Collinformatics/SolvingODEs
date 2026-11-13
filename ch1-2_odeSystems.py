"""
Chapter 1.2: Systems of ODEs
"""

from functions import ODE
import pandas as pd
import numpy as np
import sys


ode = ODE()


"""
Many interesting processes can be described by systems of ODEs, which consist of 
multiple ODEs where the right-hand side of one equation depends on the solution of the 
others. 

Such equation systems are also referred to as vector ODEs. An example is:

            u' = v,     u(0) = 1
            v' = -u,    v(0) = 0

In this example, the system is u = cos t, v = sin t, which can be
easily verified by inserting the solution into the equations and the initial
conditions

"""

# Input: Curve Params
inInitValue = np.array([0.1, 0.125])
inTime = 3 # x-max
inTimeStep = 0.01 # Timestep for actual curve
inNumDatapoints = [300, 100, 50, 25] # Euler timesteps, Δt = inTime/inNumDatapoints[i]

# Input: Figure Params
inColors = ['black', 'limegreen', 'deepskyblue', 'magenta', 'darkviolet']
inOverwriteFigs = True
inDirFigure = 'CH1'
inPlotFig = True


def fn(t, u):
    return 2*t + 1.5

euler = ode.forwardEuler(fPrime=fn, u0=inInitValue)
