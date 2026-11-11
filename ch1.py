"""
Chapter 1:

"""

from functions import ODE

ode = ODE()


"""
1.1 General-purpose ODE Solver

We will use the Euler Method as a numerical method to solve a DE.


"""
inPlotFig = True
data = ode.forwardEuler(
    y0=1, time=4, nDatapoints=20,
    plotFig=inPlotFig, pHeader='Euler Method', figDir='CH1'
)

