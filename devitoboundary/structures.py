"""
Data structures to be used in the implementation of immersed boundaries of any
sort.
"""

import numpy as np

from scipy.spatial import Delaunay

__all__ = ['PolyMesh']


class BSP_Node:
    """
    A node for a binary spatial partition (BSP) tree.

    Parameters
    ----------
    index : int
        The index of the simplex on whose plane this node splits
    index_list : ndarray
        The indices of simplices that are children of this node
    parent : BSP_Node
        The parent node of a node
    """
    def __init__(self, index, index_list, parent=None):
        self.parent = parent  # Parent node
        self.leaf = True
        self.pos = None  # Positive branch
        self.neg = None  # Negative branch
        self.index = index  # Index of the simplex/plane used for splitting
        self.index_list = index_list  # Indices of all the simplices at the node
        self.index_list.remove(self.index)  # Remove the index from child list

    def set_children(self, pos_list=None, neg_list=None):
        """Set up child tree nodes"""
        self.leaf = False
        if pos_list is not None:
            # Set up a new node using a random plane from the subset
            self.pos = BSP_Node(pos_list[np.random.randint(0, len(pos_list))],
                                pos_list, parent=self)
        if neg_list is not None:
            self.neg = BSP_Node(neg_list[np.random.randint(0, len(neg_list))],
                                neg_list, parent=self)
        del self.index_list


class BSP_Tree:
    """
    A generic BSP tree implementation.

    Parameters
    ----------
    vertices : ndarray
        List of vertex coordinates grouped as [[x, y, z], [x, y, z], ...]
    simplices : ndarray
        Indices of the vertices of each simplex
    equations : ndarray
        Coefficients of the plane equations in form x, y, z
    values : ndarray
        Constants of the plane equations
    """
    def __init__(self, vertices, simplices, equations, values):
        assert isinstance(vertices, np.ndarray), \
            "Vertices must be given as an array."
        assert isinstance(simplices, np.ndarray), \
            "Simplices must be given as an array."
        assert isinstance(equations, np.ndarray), \
            "Equations must be given as an array."
        assert isinstance(values, np.ndarray), \
            "Values must be given as an array."

        # These all need to be made into lists as may need to append polygons
        self._vertices = vertices.tolist()
        self._equations = equations.tolist()
        self._values = values.tolist()
        self._simplices = simplices.tolist()

        # Set up root node of tree
        self._root = BSP_Node(np.random.randint(0, len(simplices)),
                              list(range(len(simplices))))

        self._construct()

    @property
    def root(self):
        """The root node of the tree"""
        return self._root

    @property
    def vertices(self):
        """The vertices of the surface mesh"""
        return self._vertices

    @property
    def simplices(self):
        """The indices of the vertices of the simplices"""
        return self._simplices

    @property
    def equations(self):
        """The coefficents of the plane equation"""
        return self._equations

    @property
    def values(self):
        """The constant of the plane equation"""
        return self._values

    def _construct(self):
        """Construct the BSP tree"""
        self._split(self._root)

    def _split(self, node):
        """Split the remaining polygons using current node selection"""
        # Check each point in each simplex
        node_simplices = np.array(self._simplices)[np.array(node.index_list)]
        # Get every unique vertex in list
        node_vertices = np.array(self._vertices)[np.unique(node_simplices)]

        node_equation = np.array(self._equations)[np.array(node.index)]
        node_value = np.array(self._values)[np.array(node.index)]
        node_results = node_equation[0]*node_vertices[:, 0] \
            + node_equation[1]*node_vertices[:, 1] \
            + node_equation[2]*node_vertices[:, 2] \
            - node_value

        node_sides = np.sign(node_results[np.searchsorted(np.unique(node_simplices), node_simplices)])
        is_all_positive = np.all(node_sides <= 0, axis=1)
        is_all_negative = np.all(node_sides >= 0, axis=1)

        # If points in a simplex straddle a plane then split the simplex
        straddle = np.logical_not(np.logical_or(is_all_negative, is_all_positive))

        # Get the local indices of the simplices which straddle the plane
        simplices_to_split = straddle.nonzero()[0]

        # FIXME: redo with numpy rather than loops
        for simplex in simplices_to_split:
            print(simplex)
            # If one is on the plane
            if np.any(node_sides[simplex] == 0):
                # One of the vertices is located on the plane
                # Only need to find one intersection point in this case
                print('This one is on the plane', node_sides[simplex])
                # Indices within the simplex of the vertices which don't lie on the plane
                vertex_positions = node_sides[simplex].nonzero()[0]

                # Find the vertives which are not on the plane
                non_planar_vertices = np.array(self._vertices)[node_simplices[simplex][vertex_positions]]

                # Gradient of line
                line_m = non_planar_vertices[0] - non_planar_vertices[1]
                # Parameter value where line meets plane
                line_param = (node_value - np.dot(node_equation, non_planar_vertices[1])) \
                    / np.dot(node_equation, line_m)

                # The vertex that needs adding to split the polygon
                intersection = line_m*line_param + non_planar_vertices[1]
                # Figure out index of new vertex
                new_vertex_index = len(self._vertices)
                # Append the new vertex to the master vertex list
                self._vertices.append(intersection)
                # Figure out the indices of the two new simplices
                new_simplices_indices = [len(self._simplices), len(self._simplices)+1]
                # Add the two new simplices to the master simplex list
                positive_vertex = int(node_simplices[simplex][node_sides[simplex] == 1])
                plane_vertex = int(node_simplices[simplex][node_sides[simplex] == 0])
                negative_vertex = int(node_simplices[simplex][node_sides[simplex] == -1])

                new_simplex_1 = [new_vertex_index, positive_vertex, plane_vertex]
                new_simplex_2 = [new_vertex_index, negative_vertex, plane_vertex]

                self._simplices.extend([new_simplex_1, new_simplex_2])
                # Add the new simplices to the list on the node
                node.index_list.extend(new_simplices_indices)
                # Remove the index for the simplex that was split from the list on the node
                node.index_list.remove(simplex)

            else:
                # None of the nodes are located on the plane
                # There will be two intersection points
                print('This one is not on the plane', node_sides[simplex])
                # Find the node on its own

        print('Simplices at the node: ', node.index_list)
        # If all points in a simplex > zero, add to pos list
        # If all points in a simplex < zero, then add to neg list
        # If all points in a simplex == zero, then append to node


class PolyMesh:
    """
    A polygonal mesh of points. Used to express a non-concave (2.5D) surface
    in 3D space.

    Parameters
    ----------
    data : array_like
        Array of surface coordinates grouped as [[x, y, z], [x, y, z], ...]
    grid : Devito Grid object
        The grid against which the boundary surface is to be defined
    setup : bool, optional
        Set up the Delaunay triangulation and parameterize surface immediately.
        Set to 'False' if you want to combine several topography datasets.

    Attributes
    ----------
    points : ndarray
        The points which make up the vertices of the polygonal surface
    simplices : ndarray
        The indicies of the simplices of each polygon
    neighbors : ndarray
        The indices of neighboring polygons of a particular polygon
    equations : ndarray
        The coefficients of the plane equations
    values : ndarray
        The constants of the plane equations
    """

    def __init__(self, data, grid, setup=True):
        assert len(grid.dimensions) == 3, "PolyMesh is for 3D grids only"
        assert len(np.shape(data)) == 2, "Coordinates for each point should be in the form [x, y, z]"

        self._grid = grid
        self._points = np.array(data, dtype=np.float32)
        self._points /= self._grid.spacing  # Convert to grid-index space

        self._setup_bool = False
        if setup:
            self.setup()

    @property
    def points(self):
        """Datapoints which make up the topography point cloud"""
        return self._points

    @property
    def simplices(self):
        """Indices of the simplices of each polygon"""
        return self._simplices

    @property
    def neighbors(self):
        """The indices of neighboring polygons of each polygon"""
        return self._neighbors

    @property
    def polycount(self):
        """Number of polgons contained within mesh"""
        return self._polycount

    @property
    def equations(self):
        """The coefficents of the plane equation"""
        return self._equations

    @property
    def values(self):
        """The constant of the plane equation"""
        return self._values

    def add_data(self, data):
        """
        Add additional topography data to the mesh prior to setup.

        Parameters
        ----------
        data : array_like
            Array of surface coordinates grouped as [[x, y, z], [x, y, z], ...]
        """
        assert self._setup_bool is False, "Topography cannot be added after mesh setup."
        add_points = np.array(data, dtype=np.float32)
        add_points /= self._grid.spacing  # Convert to grid-index space
        self._points = np.concatenate(self._points, add_points)

    def setup(self):
        """Set up the mesh. The topography dataset cannot be modified once called."""
        assert self._setup_bool is False, "Mesh setup has already taken place."
        self._setup_bool = True  # Prevents repeated mesh setups
        # Initialise properties
        self._core_properties()
        # Characterize planes
        self._characterise_planes()
        # Set up bsp tree
        self._tree_setup()

    def _core_properties(self):
        """Set up all the core properties of the mesh"""
        mesh = Delaunay(self._points[:, :2], qhull_options="QJ")
        self._simplices = mesh.simplices
        self._neighbors = mesh.neighbors
        self._polycount = len(self._simplices)

    def _characterise_planes(self):
        """
        Find the equations for each plane.
        """
        plane_vector_1 = self._points[self._simplices][:, 0] \
            - self._points[self._simplices][:, 1]
        plane_vector_2 = self._points[self._simplices][:, 0] \
            - self._points[self._simplices][:, 2]
        # Equations of the plane
        self._equations = np.cross(plane_vector_1, plane_vector_2)
        self._values = np.sum(self.equations*self.points[self.simplices][:, 0], axis=1)

    def _tree_setup(self):
        """
        Initialise the BSP tree used to find nearest polygon to a node. This is
        used to accelerate node-finding.
        """
        self._tree = BSP_Tree(self._points, self._simplices,
                              self._equations, self._values)

    def query_polygons(self, x, k, distance_upper_bound=np.inf):
        """
        Traverse the tree of polygons to find polygons from nearest to furthest.

        Parameters
        ----------
        x : array_like
            Coordinate of point to query in [x, y, z] form.

        Returns
        -------
        """
        assert self._setup_bool is True, "The mesh has not been set up."
