"""
A module for segmenting interior and exterior regions from one another.
Taking a user-defined interior point, the domain will be segmented into
exterior and interior regions.
"""
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import flood_fill

_feps = np.finfo(np.float32).eps  # Get the eps


def get_point_index(i_point, spacing, origin):
    """
    Convert absolute position of a point into a grid index
    """
    i_pos = tuple([i_point[i] - origin[i] for i in range(len(i_point))])
    i_ind = tuple([int(i_pos[i]/spacing[i]) for i in range(len(i_pos))])
    return i_ind


def get_interior(sdf, i_point, qc=False):
    """
    A function to identify interior points given an interior point specified
    by the user.

    Parameters
    ----------
    sdf : Function
        The signed-distance function for the surface
    i_point : tuple of float
        The physical position of the interior point
    qc : bool
        Display a slice of the segmentation of quality checking purposes
    """

    point_index = get_point_index(i_point, sdf.grid.spacing, sdf.grid.origin)

    flooded = flood_fill(sdf.data, point_index, -np.amin(sdf.data))

    # FIXME: This wants tolerance
    # segmented = np.sign(flooded)
    segmented = flooded > -_feps

    # Show a slice of the segmentation for qc
    if qc:
        center = segmented.shape[1]//2
        plt.imshow(segmented[:, center].T, origin='lower')
        plt.colorbar()
        plt.show()

    return np.array(segmented)
