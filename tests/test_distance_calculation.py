import numpy as np

from devito import Grid
from devitoboundary import PolySurface

sup = np.testing.suppress_warnings()
sup.filter(category=RuntimeWarning)  # Prevents warnings from sqrt

SUBSAMPLE = 0.5  # Subsampling of the surface relative to the grid
PRECISION = 0.1  # Maximum allowable error in grid spacings


def z_sphere(grid, x, y):
    z_vals = 4*grid.extent[2]/3 - np.sqrt(grid.extent[2]**2 - (x - grid.extent[0]/2)**2 - (y - grid.extent[1]/2)**2)
    return z_vals


@sup
def y_pos_sphere(grid, x, z):
    y_vals = grid.extent[1]/2 + np.sqrt(grid.extent[2]**2 - (x - grid.extent[0]/2)**2 - (z - 4*grid.extent[2]/3)**2)
    return y_vals


@sup
def y_neg_sphere(grid, x, z):
    y_vals = grid.extent[1]/2 - np.sqrt(grid.extent[2]**2 - (x - grid.extent[0]/2)**2 - (z - 4*grid.extent[2]/3)**2)
    return y_vals


@sup
def x_pos_sphere(grid, y, z):
    x_vals = grid.extent[0]/2 + np.sqrt(grid.extent[2]**2 - (y - grid.extent[1]/2)**2 - (z - 4*grid.extent[2]/3)**2)
    return x_vals


@sup
def x_neg_sphere(grid, y, z):
    x_vals = grid.extent[0]/2 - np.sqrt(grid.extent[2]**2 - (y - grid.extent[1]/2)**2 - (z - 4*grid.extent[2]/3)**2)
    return x_vals


class TestDistances():
    """
    A class for testing distances calculated by querying the PolySurface object.
    """

    def test_surface_construction(self):
        """
        Test that the surface construction is working as intended.
        """
        grid = Grid(extent=(10, 10, 10), shape=(11, 11, 11))
        # Generate a data set from some analytical function which I can easily calculate distance to
        boundary_x = np.linspace(0, grid.extent[0],
                                 int(grid.shape[0]/SUBSAMPLE))

        boundary_y = np.linspace(0, grid.extent[1],
                                 int(grid.shape[1]/SUBSAMPLE))

        surface_x, surface_y = np.meshgrid(boundary_x, boundary_y)
        surface_z = z_sphere(grid, surface_x, surface_y)

        # Create the surface
        surface_data = np.vstack((surface_x.flatten(), surface_y.flatten(), surface_z.flatten())).T
        surface = PolySurface(surface_data, grid)

        # Check number of polygons generated is consistent for a given set of points
        assert surface.simplices.shape[0] == 882

    def test_measured_distances(self):
        """
        Test that the distances returned by the query are consistent with the
        analytical distances to the original surface function.
        """
        grid = Grid(extent=(10, 10, 10), shape=(11, 11, 11))
        # Generate a data set from some analytical function which I can easily calculate distance to
        boundary_x = np.linspace(0, grid.extent[0],
                                 int(grid.shape[0]/SUBSAMPLE))

        boundary_y = np.linspace(0, grid.extent[1],
                                 int(grid.shape[1]/SUBSAMPLE))

        surface_x, surface_y = np.meshgrid(boundary_x, boundary_y)
        surface_z = z_sphere(grid, surface_x, surface_y)

        # Create the surface
        surface_data = np.vstack((surface_x.flatten(), surface_y.flatten(), surface_z.flatten())).T
        surface = PolySurface(surface_data, grid)

        # Generate a set of 12 test points
        test_points = np.array([[1, 2, 3], [0.1, 2.1, 0.5], [5, 4, 0.2], [7, 2, 6],
                                [5, 2, 7], [3, 1, 4], [5.3, 2.4, 1.3], [3.7, 4.2, 5.1],
                                [1.5, 9.2, 4.3], [2.3, 6.5, 2.2], [9.5, 0.5, 3.3], [4.2, 9.2, 0.5]])

        # Query the surface with these points
        query_distances = surface.query(test_points)
        z_distances = test_points[:, 2] - z_sphere(grid, test_points[:, 0], test_points[:, 1])
        y_intersects_neg = y_neg_sphere(grid, test_points[:, 0], test_points[:, 2])
        y_intersects_pos = y_pos_sphere(grid, test_points[:, 0], test_points[:, 2])
        x_intersects_neg = y_neg_sphere(grid, test_points[:, 1], test_points[:, 2])
        x_intersects_pos = y_pos_sphere(grid, test_points[:, 1], test_points[:, 2])
        y_distances_neg = test_points[:, 1] - y_intersects_neg
        y_distances_pos = test_points[:, 1] - y_intersects_pos
        x_distances_neg = test_points[:, 0] - x_intersects_neg
        x_distances_pos = test_points[:, 0] - x_intersects_pos

        assert np.all(np.absolute(query_distances[0] - z_distances) <= PRECISION)

        measured_y_pos = query_distances[1][np.logical_not(np.isnan(query_distances[1]))]
        assert np.all(np.minimum(np.absolute(measured_y_pos
                                             - y_distances_neg[np.logical_not(np.isnan(query_distances[1]))]),
                                 np.absolute(measured_y_pos
                                             - y_distances_pos[np.logical_not(np.isnan(query_distances[1]))])) <= PRECISION)

        measured_y_neg = query_distances[2][np.logical_not(np.isnan(query_distances[2]))]
        assert np.all(np.minimum(np.absolute(measured_y_neg
                                             - y_distances_neg[np.logical_not(np.isnan(query_distances[2]))]),
                                 np.absolute(measured_y_neg
                                             - y_distances_pos[np.logical_not(np.isnan(query_distances[2]))])) <= PRECISION)

        measured_x_pos = query_distances[3][np.logical_not(np.isnan(query_distances[3]))]
        assert np.all(np.minimum(np.absolute(measured_x_pos
                                             - x_distances_neg[np.logical_not(np.isnan(query_distances[3]))]),
                                 np.absolute(measured_x_pos
                                             - x_distances_pos[np.logical_not(np.isnan(query_distances[3]))])) <= PRECISION)

        measured_x_neg = query_distances[4][np.logical_not(np.isnan(query_distances[4]))]
        assert np.all(np.minimum(np.absolute(measured_x_neg
                                             - x_distances_neg[np.logical_not(np.isnan(query_distances[4]))]),
                                 np.absolute(measured_x_neg
                                             - x_distances_pos[np.logical_not(np.isnan(query_distances[4]))])) <= PRECISION)

    def test_query_consistency(self):
        """
        Test that two successive queries of the same surface with the same set of
        points return the same distances.
        """
        grid = Grid(extent=(10, 10, 10), shape=(11, 11, 11))
        # Generate a data set from some analytical function which I can easily calculate distance to
        boundary_x = np.linspace(0, grid.extent[0],
                                 int(grid.shape[0]/SUBSAMPLE))

        boundary_y = np.linspace(0, grid.extent[1],
                                 int(grid.shape[1]/SUBSAMPLE))

        surface_x, surface_y = np.meshgrid(boundary_x, boundary_y)
        surface_z = z_sphere(grid, surface_x, surface_y)

        # Create the surface
        surface_data = np.vstack((surface_x.flatten(), surface_y.flatten(), surface_z.flatten())).T
        surface = PolySurface(surface_data, grid)

        # Generate a set of 12 test points
        test_points = np.array([[1, 2, 3], [0.1, 2.1, 0.5], [5, 4, 0.2], [7, 2, 6],
                                [5, 2, 7], [3, 1, 4], [5.3, 2.4, 1.3], [3.7, 4.2, 5.1],
                                [1.5, 9.2, 4.3], [2.3, 6.5, 2.2], [9.5, 0.5, 3.3], [4.2, 9.2, 0.5]])

        # Query the surface with these points
        query_distances = surface.query(test_points)

        # Check that two successive queries return the same distances
        new_query_distances = surface.query(test_points)
        assert np.all(np.isclose(query_distances[0], new_query_distances[0]))
