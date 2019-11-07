# Testbed file for understanding functionality of topography.py
import numpy as np
import pandas as pd
from opesciboundary import Boundary
from devito import Grid

grid_1 = Grid(extent=(1),shape=(11))
grid_2 = Grid(extent=(1,1),shape=(11,11))

def boundary_func_1(*args):
    return 0.51

def boundary_func_2(*args):
    for arg in args:
        x = arg
    #y = -(x-0.5)**2+0.5
    y = -(1.5*x-0.75)**2+1
    return y

def inv_boundary_func_2(*args):
    for arg in args:
        y = arg
    x = 0.5 + np.sqrt(0.5 - y)
    return x

boundary_obj_1 = Boundary(grid_1, boundary_func_1)
boundary_obj_2 = Boundary(grid_2, boundary_func_2, inv_boundary_func_2)
