import numpy as np
import matplotlib.pyplot as plt

from devito import Grid, TimeFunction, Eq, div, Operator

grid = Grid(shape=(101, 101), extent=(10, 10))

u = TimeFunction(name='u', grid=grid, space_order=4)

u.data[:, 50:] = 1

eq = Eq(u.forward, u.div)

plt.imshow(u.data[1])
plt.colorbar()
plt.show()

op = Operator([eq])
op.apply(time_M=1)

plt.imshow(u.data[1])
plt.colorbar()
plt.show()
