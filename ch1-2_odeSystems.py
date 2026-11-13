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


class forwardEuler:
    def __init__(self, fPrime, nDatapoints=[10], timespan=[0,1]):
        """
        A numerical solution to an ODE
        :param fPrime: derivative of a function, format: f(t, u)
        :param timespan: [int or float] duration, or maximum value of the x-axis
        :param nDatapoints: [list] Number of datapoint for a curve
        :param pHeader: [bool] print bar, [str] print bar with label
        :return: [list] pd.Dataframes of each solution found at different dt values
        """
        self.fPrime = lambda t, u: np.asarray(fPrime(t, u), float)
        self.nDatapoints = nDatapoints
        self.timespan = timespan
        self.neq = None


    def solve(self, u0):
        """
        A solve an ODE or a system of ODEs
        :param u0: initial value
        """
        if np.isscalar(u0):  # scalar ODE
            self.neq = 1  # no of equations
            u0 = float(u0[0])
        else: 
            if len(u0) > 1:
                self.neq = u0.size  # no of equations
                u0 = np.asarray(u0)
            else: # system of ODEs
                self.neq = 1
                u0 = u0[0]
    
        datasets = []
        for N in self.nDatapoints:
            t0, time = self.timespan
            dt = (time - t0) / N
            t = np.zeros(N + 1)
            if self.neq == 1:
                u = np.zeros(N + 1)
            else:
                u = np.zeros((N + 1, self.neq))
            print(f'Shape: {u.shape}')
            print(f'Timestep: Δt={dt}')
    
            # Evaluate
            u[0] = u0
            for n in range(N):
                t[n + 1] = t[n] + dt
                u[n + 1] = u[n] + dt * self.fPrime(t[n], u[n])
            cols = [f'Vect{i}' for i in range(self.neq)]
            d = pd.DataFrame(u, index=t, columns=cols)
            # datasets.append(d)
            print(f'{d}\n\n')

        return datasets

euler = forwardEuler(fPrime=fn)
euler.solve(u0=np.array(inInitValue))
