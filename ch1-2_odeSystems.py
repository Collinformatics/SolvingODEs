"""
Chapter 1.2: Systems of ODEs
"""

from functions import ODE
import pandas as pd
from math import sin
import numpy as np
import sys


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

# Input: Pendulum
inLength = 1 # meters
inInitValue = np.array([np.pi/4, 0])
inTimespan = [0, 6] # x-max
inTimeStep = 0.01 # Timestep for actual curve
inNumDatapoints = 300 # Euler timesteps, Δt = time/inNumDatapoints[i]

# Input: Figure Params
inColors = ['limegreen', 'darkviolet']
inOverwriteFigs = True
inDirFigure = 'CH1'
inPlotFig = True


# ========================================================================================
class Pendulum:
    def __init__(self, L, g=9.81):
        """
        An ODE Model for a Pendulum
        :param L: length of the pendulum
        :param g: gravitational constant (m/s²)
        """
        self.L = L
        self.g = g

    def __call__(self, t, u):
        theta, omega = u
        dTheta = omega
        dOmega = -self.g * sin(theta) / self.L
        return [dTheta, dOmega]


# Initialize classes
ode = ODE(overwriteFigs=inOverwriteFigs)
fn = Pendulum(L=inLength)

# Evaluate
lines = ode.forwardEuler(fPrime=fn, u0=inInitValue, nDatapoints=inNumDatapoints,
                         timespan=inTimespan)[0]

# Plot the approximations
if inPlotFig:
    ode.plotLines(
        data=lines, colors=inColors, title='Forward Euler Solution of a Pendulum',
        figName=f'eulerSolution_pendulum-nCurves-{len(lines.columns)}', labelX='Time',
        labelY='y(t)', figDir=inDirFigure,
    )