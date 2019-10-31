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

max_oc = most_frequent(x_set)[0]

n_domains = len(set(x_set))

class IB(SubDomainSet):
    name = 'ib'

bounds_xm = np.zeros((n_domains,), dtype=np.int32)
bounds_xM = np.zeros((n_domains,), dtype=np.int32)
bounds_ym = np.zeros((n_domains,), dtype=np.int32)
bounds_yM = np.zeros((n_domains,), dtype=np.int32)

pad = []
maxy = []
for j in range(0, n_domains):

    indices = [i for i, x in enumerate(x_set) if x == j]

    bounds_xm[j] = j
    bounds_xM[j] = n_domains-1-j

    d_size = y_set[max(indices)] - y_set[min(indices)] + 1
    pad_size = max_oc-d_size
    pad.append(pad_size)

    bounds_ym[j] = y_set[min(indices)]-pad_size
    bounds_yM[j] = n_domains-1-y_set[max(indices)]

    maxy.append(y_set[max(indices)])

bounds = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)

ib = IB(N=n_domains, bounds=bounds)

#grid = Grid(shape=(Nx,Ny), extent=(Lx,Ly), subdomains=(ib, ))
#x, y = grid.dimensions

so = 4
f = TimeFunction(name='f', grid=grid, space_order=so, coefficients='symbolic')

""" ********************* TESTING ******************* """

s = Dimension(name='s')
ncoeffs = so+1

wshape = list(grid.shape)
wshape.append(ncoeffs)
wshape = as_tuple(wshape)

wdims = list(grid.dimensions)
wdims.append(s)
wdims = as_tuple(wdims)

wx = Function(name='wx', dimensions=wdims, shape=wshape)
wy = Function(name='wy', dimensions=wdims, shape=wshape)

fill_ws = np.array([-1./12., 4./3., -5./2., 4./3., -1./12.])
fill_w0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

## Fill out the weights here ...
for i in range(grid.shape[0]):
    j_max = maxy[i]
    for j in range(grid.shape[0]):
        if j < j_max:
            wx.data[i,j,:] = fill_ws[:]/dx**2
            wy.data[i,j,:] = fill_ws[:]/dy**2
        else:
            wx.data[i,j,:] = fill_w0[:]/dx**2
            wy.data[i,j,:] = fill_w0[:]/dy**2

for index, row in stencil_data.iterrows():
    node = row[0]
    d_xx = row[1]
    d_yy = row[2]
    xx = node[0]
    yy = node[1]
    wx.data[xx,yy,:] = d_xx[:]/dx**2
    wy.data[xx,yy,:] = d_yy[:]/dy**2

f_xx_coeffs = Coefficient(2, f, x, wx)
f_yy_coeffs = Coefficient(2, f, y, wy)

subs = Substitutions(f_xx_coeffs, f_yy_coeffs)

""" ************************************************* """

eq = Eq(f.dt + f.dx2, 1, coefficients=subs)

stencil = solve(eq.evaluate, f.forward)

op = Operator(Eq(f.forward, stencil))

op(time_m=0, time_M=9, dt=0.01)

from IPython import embed; embed()

##############################################################
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
