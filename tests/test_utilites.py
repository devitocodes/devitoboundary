import pytest

from devito import Grid, Function, TimeFunction, Dimension
from devitoboundary.stencils.stencil_utils import get_grid_offset

class TestUtilities:
    @pytest.mark.parametrize('stagger', ['none', 'x', 'y', 'z', 'xyz'])
    @pytest.mark.parametrize('negative', [True, False])
    @pytest.mark.parametrize('axis', [0, 1, 2])
    @pytest.mark.parametrize('is_TimeFunction', [False, True])
    def test_get_grid_offset(self, stagger, negative, axis, is_TimeFunction):
        """
        Check that the grid offsets are being correctly detected
        """
        grid = Grid(shape=(11, 11, 11), extent=(10., 10., 10.))
        x, y, z = grid.dimensions

        if stagger == 'none':
            offset = None
        elif stagger == 'x':
            offset = x
        elif stagger == 'y':
            offset = y
        elif stagger == 'z':
            offset = z
        elif stagger == 'xyz':
            offset = (x, y, z)
        else:
            raise ValueError("Invalid stagger")

        if negative:
            if isinstance(offset, tuple):
                offset = tuple(-i for i in offset)
            elif offset is not None:
                offset = -offset

        if is_TimeFunction:
            f = TimeFunction(name='f', grid=grid, staggered=offset)
        else:
            f = Function(name='f', grid=grid, staggered=offset)

        result = get_grid_offset(f, axis)
        if isinstance(offset, Dimension):
            if grid.dimensions[axis] == offset:
                assert result == 0.5
            elif -grid.dimensions[axis] == offset:
                assert result == -0.5
            else:
                assert result == 0
        elif offset is None:
            assert result == 0
        elif isinstance(offset, tuple):
            if grid.dimensions[axis] in offset:
                assert result == 0.5
            elif -grid.dimensions[axis] in offset:
                assert result == -0.5
            else:
                assert result == 0
        
