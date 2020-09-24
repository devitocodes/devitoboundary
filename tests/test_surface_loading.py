import pytest

import numpy as np

from devito import Grid
from devitoboundary import PolyReader, NormalCalculator, SDFGenerator


class TestPipeline:
    """
    A class to test the functionality of the components of the processing
    pipeline used to generate signed distance functions
    """

    trial_surfaces_ply = ['tests/trial_surfaces/x_neg_slope.ply',
                          'tests/trial_surfaces/x_pos_slope.ply',
                          'tests/trial_surfaces/x_step.ply',
                          'tests/trial_surfaces/y_neg_slope.ply',
                          'tests/trial_surfaces/y_pos_slope.ply',
                          'tests/trial_surfaces/y_step.ply',
                          'tests/trial_surfaces/z_flat.ply']

    toggle_normals = [True, False]

    @pytest.mark.parametrize('infile', trial_surfaces_ply)
    def test_reader_ply(self, infile):
        """Test the reader with .ply files and check it updates"""
        read = PolyReader(infile)
        try:
            read.Update()
        except Exception:
            raise RuntimeError("Reader update failed.")

    @pytest.mark.parametrize('infile', trial_surfaces_ply)
    @pytest.mark.parametrize('toggle_normals', toggle_normals)
    def test_normal_calculation(self, infile, toggle_normals):
        """Test the normal calculation to ensure it updates"""
        read = PolyReader(infile)
        norms = NormalCalculator(read, toggle_normals, 20)
        try:
            norms.Update()
        except Exception:
            raise RuntimeError("Normal calculator update failed.")

    @pytest.mark.parametrize('infile', trial_surfaces_ply)
    @pytest.mark.parametrize('toggle_normals', toggle_normals)
    def test_sdf_calculation(self, infile, toggle_normals):
        """Test the SDF generation to ensure it updates sucessfully"""
        # Grid config
        extent = (100., 100., 100.)
        shape = (101, 101, 101)
        grid = Grid(extent=extent, shape=shape)
        try:
            _ = SDFGenerator(infile, grid, toggle_normals=toggle_normals)
        except Exception:
            raise RuntimeError("SDF generation failed.")


class TestNormals:
    """
    A class to check that toggling the normals has no effect on distances
    measured.
    """

    trial_surfaces_ply = ['tests/trial_surfaces/x_neg_slope.ply',
                          'tests/trial_surfaces/x_pos_slope.ply',
                          'tests/trial_surfaces/x_step.ply',
                          'tests/trial_surfaces/y_neg_slope.ply',
                          'tests/trial_surfaces/y_pos_slope.ply',
                          'tests/trial_surfaces/y_step.ply',
                          'tests/trial_surfaces/z_flat.ply']

    radii = [2, 4, 6]

    @pytest.mark.parametrize('infile', trial_surfaces_ply)
    @pytest.mark.parametrize('radius', radii)
    def test_normal_toggle(self, infile, radius):
        """Test to check that toggling normals preserves calculated distances"""
        # Grid config
        extent = (100., 100., 100.)
        shape = (101, 101, 101)
        grid = Grid(extent=extent, shape=shape)

        # Toggle normals
        sdf_t = SDFGenerator(infile, grid, radius=radius, toggle_normals=True)
        # Don't toggle normals
        sdf_f = SDFGenerator(infile, grid, radius=radius, toggle_normals=False)

        diff = sdf_t.array + sdf_f.array
        min_diff = 0
        max_diff = -2*radius*grid.spacing[0]

        if not np.all(np.logical_or(diff == min_diff, diff == max_diff)):
            raise ValueError("Toggling normals caused distance change")


class TestOrientation:
    """
    A class to test the orientation and position of generated signed distance
    functions, and ensure that they are correctly mapped to the grid.
    """

    trial_surfaces_orientation = ['tests/trial_surfaces/x_neg_slope.ply',
                                  'tests/trial_surfaces/x_pos_slope.ply',
                                  'tests/trial_surfaces/y_neg_slope.ply',
                                  'tests/trial_surfaces/y_pos_slope.ply',
                                  'tests/trial_surfaces/z_flat.ply']

    trial_surfaces_position = ['tests/trial_surfaces/x_step.ply',
                               'tests/trial_surfaces/y_step.ply']

    x_offset = (10., 20., 30.)
    y_offset = (10., 20., 30.)

    @pytest.mark.parametrize('infile', trial_surfaces_orientation)
    def test_orientation(self, infile):
        """A test to assert that no rotation occurs during reading"""
        # Grid config
        extent = (100., 100., 100.)
        shape = (101, 101, 101)
        grid = Grid(extent=extent, shape=shape)

        sdf = SDFGenerator(infile, grid, toggle_normals=True)

        if infile == 'trial_surfaces/x_pos_slope.ply':
            diag = np.diagonal(sdf.array, axis1=0, axis2=2)
            if np.any(np.absolute(diag) > 1e-6):
                raise ValueError("Boundary region out of sdf scope")
        elif infile == 'trial_surfaces/x_neg_slope.ply':
            diag = np.diagonal(sdf.array[::-1, :, :], offset=-1, axis1=0, axis2=2)
            if np.any(np.absolute(diag) > 1e-6):
                raise ValueError("Boundary region out of sdf scope")
        elif infile == 'trial_surfaces/y_pos_slope.ply':
            diag = np.diagonal(sdf.array, axis1=1, axis2=2)
            if np.any(np.absolute(diag) > 1e-6):
                raise ValueError("Boundary region out of sdf scope")
        elif infile == 'trial_surfaces/y_neg_slope.ply':
            diag = np.diagonal(sdf.array[:, ::-1, :], offset=-1, axis1=1, axis2=2)
            if np.any(np.absolute(diag) > 1e-6):
                raise ValueError("Boundary region out of sdf scope")
        elif infile == 'trial_surfaces/z_flat.ply':
            if np.any(sdf.array[:, :, int(0.1*grid.shape[2])] > 1e-6):
                raise ValueError("Boundary region out of sdf scope")

    @pytest.mark.parametrize('infile', trial_surfaces_position)
    def test_position(self, infile):
        """A test to assert that surfaces are not translated during reading"""
        # Grid config
        extent = (100., 100., 100.)
        shape = (101, 101, 101)
        grid = Grid(extent=extent, shape=shape)

        sdf = SDFGenerator(infile, grid, toggle_normals=True)

        if infile == 'trial_surfaces/x_step.ply':
            lo_half = sdf.array[:int(0.5*grid.shape[0]), :, int(0.2*grid.shape[2])]
            hi_half = sdf.array[int(0.5*grid.shape[0]):, :, int(0.5*grid.shape[2])]

            lo_aligned = np.all(np.absolute(lo_half) < 1e-6)
            hi_aligned = np.all(np.absolute(hi_half) < 1e-6)
            if not lo_aligned or not hi_aligned:
                raise ValueError("Boundary region has been shifted")

        if infile == 'trial_surfaces/y_step.ply':
            lo_half = sdf.array[:, :int(0.5*grid.shape[1]), int(0.2*grid.shape[2])]
            hi_half = sdf.array[:, int(0.5*grid.shape[1]):, int(0.5*grid.shape[2])]

            lo_aligned = np.all(np.absolute(lo_half) < 1e-6)
            hi_aligned = np.all(np.absolute(hi_half) < 1e-6)
            if not lo_aligned or not hi_aligned:
                raise ValueError("Boundary region has been shifted")

    @pytest.mark.parametrize('infile', trial_surfaces_position)
    @pytest.mark.parametrize('x_offset', x_offset)
    @pytest.mark.parametrize('y_offset', y_offset)
    def test_origin(self, infile, x_offset, y_offset):
        """A test to assert that shifting the origin moves the surface correctly"""
        # Grid config
        extent = (50., 50., 50.)
        shape = (51, 51, 51)
        origin = (x_offset, y_offset, 0.)
        grid = Grid(extent=extent, shape=shape, origin=origin)

        sdf = SDFGenerator(infile, grid, toggle_normals=True)

        x_index_shift = -int(x_offset/grid.spacing[0])
        y_index_shift = -int(y_offset/grid.spacing[1])

        if infile == 'trial_surfaces/x_step.ply':
            lo_half = sdf.array[:int(grid.shape[0] + x_index_shift),
                                :,
                                int(0.4*grid.shape[2])]
            hi_half = sdf.array[int(grid.shape[0] + x_index_shift):,
                                :,
                                int(grid.shape[2] - 1)]

            lo_aligned = np.all(np.absolute(lo_half) < 1e-6)
            hi_aligned = np.all(np.absolute(hi_half) < 1e-6)
            if not lo_aligned or not hi_aligned:
                raise ValueError("Boundary region has been shifted")
        elif infile == 'trial_surfaces/y_step.ply':
            lo_half = sdf.array[:,
                                :int(grid.shape[0] + y_index_shift),
                                int(0.4*grid.shape[2])]
            hi_half = sdf.array[:,
                                int(grid.shape[0] + y_index_shift):,
                                int(grid.shape[2] - 1)]

            lo_aligned = np.all(np.absolute(lo_half) < 1e-6)
            hi_aligned = np.all(np.absolute(hi_half) < 1e-6)
            if not lo_aligned or not hi_aligned:
                raise ValueError("Boundary region has been shifted")
