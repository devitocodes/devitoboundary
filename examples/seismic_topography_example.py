import matplotlib.pyplot as plt

from devito import Grid, TimeFunction, Eq
from devitoboundary import ImmersedBoundary

VP = 1.2

extent = (10800., 10800., 5400.)
shape = (109, 109, 55)
origin = (0., 0., -3900.)
grid = Grid(shape=shape, extent=extent, origin=origin)

u = TimeFunction(name='u', grid=grid,
                 space_order=4, time_order=2,
                 coefficients='symbolic')

# Surface configuration
infile = 'topography/crater_lake.ply'
# Create the immersed boundary surface
surface = ImmersedBoundary(infile, u)

# Zero even derivatives on the boundary
bc_0 = Eq(surface.u(u, surface.x_b(u)), 0)
bc_2 = Eq(surface.u(u, surface.x_b(u), 2), 0)
bcs = [bc_0, bc_2]

surface.add_bcs(u, bcs)

# Configure derivative needed
deriv = ('u.d2',)

surface.subs(deriv)
