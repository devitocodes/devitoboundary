"""
A selection of symbols used for specification of immersed boundaries and
processes which this entails.
"""
import sympy as sp

x_a = sp.IndexedBase('x_a')  # Arbitrary values of x
u_x_a = sp.IndexedBase('u_x_a')  # Respective values of the function

a = sp.IndexedBase('a')  # Polynomial coefficients

# Polynomial iterator, Maximum polynomial order
n, n_max = sp.symbols('n, n_max')

# Generic boundary position, right boundary position, left boundary position
x_b, x_r, x_l = sp.symbols('x_b, x_r, x_l')

x_c = sp.symbols('x_c')  # Continuous x
f = sp.IndexedBase('f')  # Function values at particular points
h_x = sp.symbols('h_x')  # Grid spacing
# Distance to boundary in grid increments
eta_l, eta_r = sp.symbols('eta_l, eta_r')
