"""
A module for reading and processing point cloud and polygon data with the
ultimate goal of obtaining axial distances in positive and negative directions.
Contains various geometric utilities.
"""

import os
import vtk

import numpy as np

from vtk.numpy_interface import dataset_adapter as dsa


__all__ = ['PolyReader', 'NormalCalculator', 'SDFGenerator']


class PolyReader:
    """
    A reader for common polygon and point cloud data formats. Automatically
    detects file type from suffix. Supported formats include .ply (support for
    additional file types is planned).

    Parameters
    ----------
    infile : str
        The file to be read.

    Methods
    -------
    filename(self)
        Returns name of input file

    GetOutputPort(self)
        Return the VTK output port of the reader
    """
    def __init__(self, infile):
        self._file = str(infile)

        _, ext = os.path.splitext(self._file)

        if ext == '.ply':
            self._reader = vtk.vtkPLYReader()
        else:
            read_err = "Files of type {} are currently unsupported"
            raise NotImplementedError(read_err.format(ext))

        self._reader.SetFileName(self._file)

    @property
    def filename(self):
        """
        Return the file which is being read
        """
        return self._file

    def GetOutputPort(self):
        """
        Returns the output port of the VTK file reader.
        """
        return self._reader.GetOutputPort()

    def Update(self):
        """
        Update the state of the reader.
        """
        return self._reader.Update()


class NormalCalculator:
    """
    Uses VTK to estimate normals via principal component analysis. Based on
    VTK's vtkPCANormalEstimation function.

    Parameters
    ----------
    input : PolyReader
        The input reader for the normal calculator
    toggle_normals : bool
        Flip the direction of the estimated point normals. This has the effect
        of reversing the side of the surface which is considered to be the
        interior (positively valued).
    sample : int
        The number of sample points used for the PCA normal estimation.

    Methods
    -------
    GetOutputPort(self)
        Return the VTK output port of the reader
    """
    def __init__(self, input, toggle_normals, sample):
        self._norms = vtk.vtkPCANormalEstimation()
        self._norms.SetInputConnection(input.GetOutputPort())
        self._norms.SetSampleSize(sample)

        if toggle_normals:
            self._norms.FlipNormalsOn()
        else:
            self._norms.FlipNormalsOff()

        self._norms.SetNormalOrientationToGraphTraversal()

    def GetOutputPort(self):
        """
        Returns the output port of the normal calculation
        """
        return self._norms.GetOutputPort()

    def Update(self):
        """
        Update the state of the normal calculator.
        """
        return self._norms.Update()


class SDFGenerator:
    """
    Uses VTK to generate a signed distance function from a supplied point
    cloud or polygon file. This function is discretized on the specified
    Devito grid. Based on VTK's vtkSignedDistance function.

    Parameters
    ----------
    infile : str
        The point cloud or polygon file to be read.
    grid : Devito Grid
        The grid onto which the SDF should be discretized
    radius : int
        The radius from the boundary over which the SDF should be calculated,
        specified in grid increments (dx). Warning, setting large values here
        may significantly slow SDF calculation. Default is 2.
    toggle_normals : bool
        Flip the direction of the estimated point normals. This has the effect
        of reversing the side of the surface which is considered to be the
        interior (positively valued). Default is False.
    sample : int
        The number of sample points used for the PCA normal estimation. Default
        is 20.
    """
    def __init__(self, infile, grid, radius=2, toggle_normals=False, sample=20):
        # Catch non-3D grids
        if np.shape(grid.dimensions) != (3,):
            dim_count = np.shape(grid.dimensions)[0]
            dim_err = "The specified grid has {:n} dimensions, 3 are required"
            raise ValueError(dim_err.format(dim_count))

        self._grid = grid
        self._reader = PolyReader(infile)
        self._norms = NormalCalculator(self._reader, toggle_normals, sample)

        self._dist = vtk.vtkSignedDistance()
        self._dist.SetInputConnection(self._norms.GetOutputPort())

        self._dist.SetRadius(radius*grid.spacing[0])

        self._dist.SetBounds(self._get_bounds())
        self._dist.SetDimensions(self._grid.shape)
        self._dist.Update()

        # Get the SDF as an array
        sdf_array = dsa.WrapDataObject(self._dist.GetOutput())

        # Reshape the array to the grid
        sdf_cube = np.reshape(sdf_array.PointData['ImageScalars'],
                              self._grid.shape[::-1])

        # Swap axis to get [x, y, z] order
        self._array = np.swapaxes(sdf_cube, 0, 2)

    @property
    def grid(self):
        """
        The grid on which the SDF is discretized
        """
        return self._grid

    @property
    def array(self):
        """
        The SDF as a 3D numpy array. Dimensions of x, y, z.
        """
        return self._array

    def _get_bounds(self):
        """
        Gets the bounds of the SDF from the Devito grid.
        """
        # Replace this once 1461 goes through
        # Lower bounds are the origin
        min_bound = np.array([i.data for i in self._grid.origin])
        max_bound = min_bound + np.array(self._grid.extent)

        return min_bound[0], max_bound[0], \
            min_bound[1], max_bound[1], \
            min_bound[2], max_bound[2]
