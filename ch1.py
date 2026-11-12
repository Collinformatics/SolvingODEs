"""
Chapter 1:

"""

from functions import ODE
import pandas as pd
import numpy as np
from sympy import exp, latex, symbols, pretty
import sys




ode = ODE()


"""
1.1 General-purpose ODE Solver

We will use the Euler Method as a numerical method to solve a DE.

Pass a derivative of a function to ode.forwardEuler to numerically solve an ODE.
Try experimenting with a different number of datapoints to see their effect.
    Notice that the initial value (inInitValue) will be very important for the solution

eˣ
"""
# Input: Figure Params
inA = 0.5
inInitValue = 0
inColors = ['black', 'limegreen', 'deepskyblue', 'magenta', 'darkviolet']
inTime = 3
inTimeStep = 0.05
inNumDatapoints = [160, 80, 40, 20]

# Input: Save Params
inOverwriteFigs = True
inDirFigure = 'CH1'
inPlotFig = True

# Input: Function
def fn(t):
    return t**2/3 - 9*np.sin(t)


# ========================================================================================
# Set params
ode.overwriteFigs = inOverwriteFigs

# Generate exponential curve
time = np.arange(0, stop=inTime+inTimeStep, step=inTimeStep)
y = fn(time)
data = pd.DataFrame(y, index=time, columns=['Actual'])

# Differentiate
df_dt = ode.derivative(fn, idxDiffArg=0)

# Estimate the curve
lines = [data]
curves = ode.forwardEuler(fPrime=df_dt, y0=inInitValue, time=inTime,
                          nDatapoints=inNumDatapoints, pHeader='Euler Method')
lines += curves

# Plot the approximations
if inPlotFig:
    ode.plotLines(
        data=lines, colors=inColors, title='Solving an ODE: Forward Euler',
        figName=f'eulerSolution_nCurves-{len(inNumDatapoints)}', labelX='Time',
        labelY='y(t)', figDir=inDirFigure,
    )
