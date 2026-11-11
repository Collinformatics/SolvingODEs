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


"""
# Input: Figure Params
inInitU = 1
inColors = ['black', 'green', 'purple']
inTime = 4
inTimeStep = 0.05
inNumDatapoints = [40, 20]

# Input: Save Params
inDirFigure = 'CH1'
inPlotFig = True

# Generate exponential curve
e = pretty(exp(symbols('x')))
time = np.arange(0, inTime + 0.05, 0.05)
y = [np.exp(x) for x in time]
data = pd.DataFrame(y, index=time, columns=[e])

# Estimate the curve
lines = [data]
for n in inNumDatapoints:
    lines.append(
        ode.forwardEuler(
            y0=inInitU, time=inTime, nDatapoints=n,
            pHeader='Euler Method', figDir=inDirFigure
        )
    )

# Plot the approximations
if inPlotFig:
    ode.plotLines(
        data=lines, colors=inColors, title='Solving an ODE: Forward Euler',
        figName='eulerSolution', labelX='Time', labelY='y(t)', figDir=inDirFigure,
    )
