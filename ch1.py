"""
Chapter 1: Solving ODEs with the Forward Euler
"""

from functions import ODE
import pandas as pd
import numpy as np
import sys




ode = ODE()


"""
1.1 General-purpose ODE Solver

We will use the Euler Method as a numerical method to solve a DE.

Pass a derivative of a function to ode.forwardEuler to numerically solve an ODE.
Try experimenting with a different number of datapoints to see their effect.
Notice that the initial value (inInitValue) will also be very important for the solution.
"""
# Input: Curve Params
inInitValue = 0.1
inTime = 3 # x-max
inTimeStep = 0.05 # Timestep for actual curve
inNumDatapoints = [160, 80, 40, 20] # Used to set Euler timesteps

# Input: Figure Params
inColors = ['black', 'limegreen', 'deepskyblue', 'magenta', 'darkviolet']
inOverwriteFigs = True
inDirFigure = 'CH1'
inPlotFig = False

# Input: Function
def fn(t, u=5):
    return t**2


# ========================================================================================
# Set params
ode.overwriteFigs = inOverwriteFigs

# Generate exponential curve
time = np.arange(0, stop=inTime+inTimeStep, step=inTimeStep)
data = pd.DataFrame(fn(time), index=time, columns=['Actual'])

# Differentiate
df_dt = ode.derivative(fn, idxDiffArg=0)

# Estimate the curve
lines = [data]
curves = ode.forwardEuler(fPrime=df_dt, u0=inInitValue, time=inTime,
                          nDatapoints=inNumDatapoints, pHeader='Euler Method')
lines += curves

# Plot the approximations
if inPlotFig:
    ode.plotLines(
        data=lines, colors=inColors, title='Solving an ODE: Forward Euler',
        figName=f'eulerSolution_nCurves-{len(inNumDatapoints)}', labelX='Time',
        labelY='y(t)', figDir=inDirFigure,
    )

# ========================================================================================
# Input: Curve Params
inInitValue = 0.01
inTime = 75 # x-max
inTimeStep = 0.05 # Timestep for actual curve
inNumDatapoints = [300, 150, 75, 25] # Used to set Euler timesteps

# Input: Figure Params
inColors = ['limegreen', 'deepskyblue', 'magenta', 'darkviolet']
inOverwriteFigs = True
inDirFigure = 'CH1'
inPlotFig = True


def logistic(t, u=inInitValue, alpha=0.2, R=24):
    return alpha * u * (1 - u/R)

# Estimate the curve
lines = []
curves = ode.forwardEuler(fPrime=logistic, u0=inInitValue, time=inTime,
                          nDatapoints=inNumDatapoints, pHeader='Euler Method')
lines += curves

# Plot the approximations
if inPlotFig:
    ode.plotLines(
        data=lines, colors=inColors, title='Solving a Logistic ODE: Forward Euler',
        figName=f'eulerSolution_logistic_nCurves-{len(inNumDatapoints)}', labelX='Time',
        labelY='y(t)', figDir=inDirFigure,
    )
