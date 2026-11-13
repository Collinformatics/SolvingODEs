"""
Chapter 1.1: Solving ODEs with the Forward Euler


We will use the Euler Method as a numerical method to solve an ODE.
An ODE can be though of as a function of time "t", and the parent function "u(t)". 
So we can write a given ODE "u'(t)" in the following form:

            u′(t) = f(t, u(t))                                                       (1.1)

Because the DE shows the rate of change over a time interval "Δt", we can use Eqt 1.1 
to predict the next output of u(tₙ₊₁) with the following Eqt, the Forward Euler:

            uₙ₊₁ = uₙ + Δt * f(tₙ, uₙ)                                           (1.2)

For this we simply need the initial value "u₀" at t₀ where u(t₀) = u₀, then we can 
predict the next value in the curve "uₙ₊₁" by the rate of change we expect at each "Δt".


The following code will pass a derivative of a function to ode.forwardEuler so that we 
can solve the numerically solve a given ODE.

Try experimenting with a different number of datapoints to see their effect.
Notice that the initial value (inInitValue) will also be very important for the solution.
"""

from functions import ODE
import pandas as pd
import numpy as np


"""
In the first example we'll create a function, and then compute df/dt, with this ODE we 
can use the Forward Euler to solve it.

By experimenting with different equations we can see how closely the Euler can come to 
solving the DE, and what effect different timesteps have on the solution.
"""

# Input: Curve Params
inInitValue = 9
inTimespan = [0, 3] # x-max
inTimeStep = 0.01 # Timestep for actual curve
inNumDatapoints = [300, 100, 50, 25] # Euler timesteps, Δt = time/inNumDatapoints[i]

# Input: Figure Params
inColors = ['black', 'limegreen', 'deepskyblue', 'magenta', 'darkviolet']
inOverwriteFigs = True
inDirFigure = 'CH1'
inPlotFig = True

# Input: Function
def fn(t, u=0.5):
    return t**3 - 7*np.sin(t) + 10


# # Execute Code
# Initialize class
ode = ODE(overwriteFigs=inOverwriteFigs)

# Generate exponential curve
time = np.arange(0, stop=inTimespan[-1]+inTimeStep, step=inTimeStep)
data = pd.DataFrame(fn(time), index=time, columns=['Actual'])

# Differentiate
df_dt = ode.derivative(fn, idxDiffArg=0)

# Estimate the curve
lines = [data]
curves, nCurves = ode.forwardEuler(fPrime=df_dt, u0=inInitValue, timespan=inTimespan,
                                   nDatapoints=inNumDatapoints, pHeader='Euler Method')
lines += curves

# Plot the approximations
if inPlotFig:
    ode.plotLines(
        data=lines, colors=inColors, title='Solving an ODE: Forward Euler',
        figName=f'eulerSolution_nCurves-{nCurves+1}', labelX='Time',
        labelY='y(t)', figDir=inDirFigure,
    )


# ========================================================================================
"""
Now instead of creating a function, well start with an ODE, the logistic curve, 
and use the Forward Euler to solve it.
"""

# Input: Curve Params
inInitValue = 0.01
inTimespan = [0, 60]
inTimeStep = 0.05 # Timestep for actual curve
inNumDatapoints = [300, 150, 75, 25] # Euler timesteps, Δt = time/inNumDatapoints[i]

# Input: Figure Params
inColors = ['limegreen', 'deepskyblue', 'magenta', 'darkviolet']
inPlotFig = True


def logistic(t, u=inInitValue, alpha=0.2, R=5):
    return alpha * u * (1 - u/R)


# # Execute Code
# Estimate the curve
lines = []
curves, nCurves = ode.forwardEuler(fPrime=logistic, u0=inInitValue, timespan=inTimespan,
                                   nDatapoints=inNumDatapoints, pHeader='Euler Method')
lines += curves

# Plot the approximations
if inPlotFig:
    ode.plotLines(
        data=lines, colors=inColors, title='Solving a Logistic ODE: Forward Euler',
        figName=f'eulerSolution_logistic_nCurves-{nCurves}', labelX='Time',
        labelY='y(t)', figDir=inDirFigure,
    )
