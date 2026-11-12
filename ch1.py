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

Try experimenting with a different number of datapoints to see their effect

eˣ
"""
# Input: Figure Params
inA = 0.5
inInitU = 6
inColors = ['black', 'limegreen', 'deepskyblue', 'magenta', 'darkviolet']
inTime = 3
inTimeStep = 0.05
inNumDatapoints = [160, 80, 40, 20]

# Input: Save Params
inOverwriteFigs = True
inDirFigure = 'CH1'
inPlotFig = True

# Input: Function
def fn(t, u=inInitU,):
    return u*t/3 - 9*np.sin(t) + u


# ========================================================================================
# Generate exponential curve
time = np.arange(0, stop=inTime+inTimeStep, step=inTimeStep)
y = fn(time)
data = pd.DataFrame(y, index=time, columns=['Actual'])

# Estimate the curve
lines = [data]
curves = ode.forwardEuler(
    f=fn, y0=inInitU, time=inTime, nDatapoints=inNumDatapoints,
    pHeader='Euler Method', figDir=inDirFigure
)
lines += curves

# Set params
ode.overwriteFigs = inOverwriteFigs

# Plot the approximations
if inPlotFig:
    ode.plotLines(
        data=lines, colors=inColors, title='Solving an ODE: Forward Euler',
        figName=f'eulerSolution_nCurves-{len(inNumDatapoints)}', labelX='Time',
        labelY='y(t)', figDir=inDirFigure,
    )
