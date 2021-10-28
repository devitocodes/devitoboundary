import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap

cmap = np.loadtxt('cmap.dat', dtype=np.int)
devito_seismic = ListedColormap(cmap/255.0)

infile = '../data/seismic_topography_wavefield.npy'
dat = np.load(infile)

insurface = '../topography/crater_lake.ply'
topography = pv.read(insurface)

# Create the spatial reference
grid = pv.UniformGrid()

# Set the grid dimensions: shape + 1 because we want to inject our values on
#   the CELL data
grid.dimensions = np.array(dat.shape[1:]) + 1

# Edit the spatial reference
grid.origin = (0., 0., -3900.)  # The bottom left corner of the data set
grid.spacing = (50, 50, 50)  # These are the cell sizes along each axis

# Add the data values to the cell data
grid.cell_arrays["values"] = dat[-2].flatten(order="F")  # Flatten the array!

cake_slice = pv.Cylinder(center=(10800., 10800., -1200.), direction=(0., 0., 1.),
                         radius=7637., height=6000.)

p = pv.Plotter()

"""
clipped = grid.clip_surface(cake_slice, invert=False)

clipped_topography = topography.clip_surface(cake_slice, invert=False)

double_clipped = clipped.clip_surface(clipped_topography, invert=False)
p.add_mesh(clipped_topography, opacity=1, specular=0.5, specular_power=15)
p.add_mesh(clipped, opacity=0.75, cmap='seismic', show_scalar_bar=False)
# p.add_mesh(double_clipped, opacity=1, cmap=devito_seismic, show_scalar_bar=False)
"""
# p.add_mesh(topography, opacity=1, specular=0.5, specular_power=15)
p.add_mesh(grid, opacity=0.75, cmap='seismic', show_scalar_bar=False)

source = pv.Sphere(radius=300, center=(5400, 5400, -500))
p.add_mesh(source, color='r')

outfile = 'render_6.png'
p.show(screenshot=outfile)

# Modify this to make a movie and run all day and night
