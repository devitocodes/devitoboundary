from devito import Grid
from devitoboundary import PolyTree

data = [[1, 2, 3], [2, 1, 0], [5, 1, 2],
        [5, 3, 1], [1, 3, 5], [1, 2, 8],
        [2, 5, 7], [4, 1, 2], [8, 5, 2],
        [17, 15, 1], [13, 15, 2], [13, 14, 3]]

grid = Grid(extent=(20, 20, 20), shape=(11, 11, 11))

mesh = PolyTree(data, grid)
