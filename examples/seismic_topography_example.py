from devito import Grid, TimeFunction, Eq, solve, Operator
from devitoboundary import ImmersedBoundary
from examples.seismic import TimeAxis, RickerSource

C = 0.1  # Courant number
VP = 1.5  # P wave velocity

# Grid configuration
# 10.8 x 10.8 x 5.4 km
# 50m spacing
extent = (10800., 10800., 5400.)
shape = (217, 217, 109)  # (109, 109, 55)
origin = (0., 0., -3900.)
grid = Grid(shape=shape, extent=extent, origin=origin)

# Time series configuration
t0 = 0.  # Simulation starts at t=0
tn = 500.  # Simulation length in ms
dt = C*grid.spacing[0]/(VP)

steps = int((t0+tn)/dt)+2

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

# Add the boundary conditions to the surface
surface.add_bcs(u, bcs)

"""
# Configure the source
time_range = TimeAxis(start=t0, stop=tn, step=dt)
f0 = 0.006  # 6Hz
src = RickerSource(name='src', grid=grid, f0=f0,
                   npoint=1, time_range=time_range)

# First, position source centrally in x and y dimensions, then set depth
src.coordinates.data[0, :-1] = 5400.  # Centered
src.coordinates.data[0, -1] = -500  # 500m below sea level
"""

# Configure derivative needed
deriv = ('u.d2',)

# We can now write the PDE
pde = VP*u.dt2 - u.laplace
eq = Eq(pde, 0, coefficients=surface.subs(deriv))

"""
# And set up the update
stencil = solve(eq.evaluate, u.forward)

# Our injection term
src_term = src.inject(field=u.forward, expr=src*dt**2/VP)

# Now create our operator
op = Operator([Eq(u.forward, stencil)] + src_term)
"""
