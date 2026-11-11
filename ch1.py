"""
Chapter 1:

"""


from functions import ODE

ode = ODE()


"""
1.1 General-purpose ODE Solver

Our goal is to implement numerical solvers that can be applied to any ODE, without being 
limited to a single example or a specific class of equations.
To achieve this, we need a general abstract notation for an arbitrary ODE. 
We will write the ODEs on the following form:

            u′(t) = f (t, u(t))                                                      (1.1)

The ODE is fully specified by the definition of the right-hand side function f (t, u).
Examples of this function may be:

            f(t, u) = αu, exponential growth
            f(t, u) = (1 − (u / R)), logistic growth
            f(t, u) = −b|u|u + g, falling body in a fluid

- For the sake of generality, we write all the right-hand sides of the ODEs as functions 
of both "t" and "u", even though the mathematical formulations only involve "u".
- This is highly convenient when we start programming and want to use the same solver for 
a diverse range of ODE models.

Now, our objective is to write functions and classes that accept the function "f" as input
and solve the corresponding ODE to generate the output u.
"""
ode.header(txt='1.1 General-Purpose ODE Solver')

