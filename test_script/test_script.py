import numpy as np

from opesciboundary import Boundary, plot_boundary
from devito import (Grid, Function, TimeFunction, SubDomainSet, Eq, solve, Operator,
                    Coefficient, Substitutions, Dimension)
from devito.tools import as_list, as_tuple

import matplotlib
import matplotlib.pyplot as plt

# Parameters
pi = np.pi
eps = np.finfo(float).eps

# Boundary function definitions
def boundary_func(*args):
    for arg in args:
        x = arg
    y = 0.9-(x-0.5)**2
    return y

def iboundary_func(*args):
    for arg in args:
        y = arg
    x0 = 0.5+np.sqrt(0.9-y)
    x1 = 0.5-np.sqrt(0.9-y)
    return x0, x1

def most_frequent(List): 
    dict = {} 
    count, itm = 0, '' 
    for item in reversed(List): 
        dict[item] = dict.get(item, 0) + 1
        if dict[item] >= count : 
            count, itm = dict[item], item 
    return count, itm

# Set up the Devito grid
Lx = 1.
Ly = 1.

Nx = 101
Ny = Nx

dx = Lx/(Nx-1)
dy = dx

grid = Grid(shape=(Nx,Ny), extent=(Lx,Ly))
time = grid.time_dim
t = grid.stepping_dim
x,y = grid.dimensions

# Create our immersed boundar
boundary_obj = Boundary(grid, boundary_func, iboundary_func)

stencil_data = boundary_obj.stencil()

stencil_data = stencil_data.sort_values(by=['Node'])

x_set = [i[0] for i in stencil_data['Node'].values]
y_set = [i[1] for i in stencil_data['Node'].values]

n_domains = len(set(x_set))

class IB(SubDomainSet):
    name = 'ib'

bounds_xm = np.zeros((n_domains,), dtype=np.int32)
bounds_xM = np.zeros((n_domains,), dtype=np.int32)
bounds_ym = np.zeros((n_domains,), dtype=np.int32)
bounds_yM = np.zeros((n_domains,), dtype=np.int32)

for j in range(0, n_domains):
    indices = [i for i, x in enumerate(x_set) if x == j]
    bounds_xm[j] = j
    bounds_xM[j] = n_domains-1-j
    bounds_ym[j] = y_set[min(indices)]
    bounds_yM[j] = n_domains-1-y_set[max(indices)]

bounds = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)

ib = IB(N=n_domains, bounds=bounds)

grid = Grid(shape=(Nx,Ny), extent=(Lx,Ly), subdomains=(ib, ))
x, y = grid.dimensions

so = 4
f = TimeFunction(name='f', grid=grid, space_order=5, coefficients='symbolic')

""" ********************* TESTING ******************* """

s = Dimension(name='s')
ncoeffs = so+1

wshape = as_list(n_domains)
wshape.extend([1, most_frequent(x_set)[0]])
wshape.append(ncoeffs)
wshape = as_tuple(wshape)

wdims = as_list(grid.subdomains['ib'].implicit_dimension)
wdims.extend(as_list(grid.subdomains['ib'].dimensions))
wdims.append(s)
wdims = as_tuple(wdims)

w = Function(name='w', dimensions=wdims, shape=wshape)

# Fill out the weights here ...
# ...

#f_xx_coeffs = Coefficient(2, f, x, wx)
#f_yy_coeffs = Coefficient(2, f, y, wy)

#subs = Substitutions(f_xx_coeffs, f_yy_coeffs)



""" ************************************************* """

#stencil = Eq(f.forward, solve(Eq(f.dt, 1, coefficients=subs), f.forward),
             #subdomain=grid.subdomains['ib'])

#op = Operator(stencil)
#op(time_m=0, time_M=9, dt=1)

from IPython import embed; embed()

## Plot of 'nodes' where the stencil is modified
## relative to the boundary
#xg = np.linspace(0,Lx,Nx)
#yg = xg

#nodes = stencil_data["Node"].values
#nnodes = len(stencil_data.index)

#xn = np.zeros(nnodes)
#yn = np.zeros(nnodes)

#for j in range(0,nnodes):
    #b = np.asarray(nodes[j])*[dx,dy]
    #xn[j] = b[0]
    #yn[j] = b[1]

#yb = boundary_func(xg)

#plot_boundary(xg, yb, xn, yn, Lx, Ly)
