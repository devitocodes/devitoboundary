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
        self.pos = None  # Positive branch
        self.neg = None  # Negative branch
        self.index = index  # Index of the simplex/plane used for splitting
        self.plane_indices = []  # Indices of any other simplices that lie in the plane  # NUMPYISE
        self.index_list = index_list  # Indices of all the simplices at the node  # NUMPYISE
        self.index_list.remove(self.index)  # Remove the index from child list  # NUMPYISE

    def set_children(self, pos_list, neg_list):
        """Set up child tree nodes"""
        if len(pos_list) != 0:
            # Set up a new node using a random plane from the subset
            self.pos = BSP_Node(pos_list[np.random.randint(0, len(pos_list))],  # NUMPYISE
                                pos_list, parent=self)
        if len(neg_list) != 0:
            self.neg = BSP_Node(neg_list[np.random.randint(0, len(neg_list))],  # NUMPYISE
                                neg_list, parent=self)
        self.index_list = []  # DITCH


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
    leafsize : positve int
        Number of polygons at each leaf node (default is one)
    """
    def __init__(self, vertices, simplices, equations, values, leafsize=1):
        assert isinstance(vertices, np.ndarray), \
            "Vertices must be given as an array."
        assert isinstance(simplices, np.ndarray), \
            "Simplices must be given as an array."
        assert isinstance(equations, np.ndarray), \
            "Equations must be given as an array."
        assert isinstance(values, np.ndarray), \
            "Values must be given as an array."

        # These all need to be made into lists as may need to append polygons
        self._vertices = vertices.tolist()  # NUMPYISE
        self._equations = equations.tolist()  # NUMPYISE
        self._values = values.tolist()  # NUMPYISE
        self._simplices = simplices.tolist()  # NUMPYISE

        # Set up root node of tree
        self._root = BSP_Node(np.random.randint(0, len(simplices)),
                              list(range(len(simplices))))  # NUMPYISE

        self.construct(leafsize-1)  # Only one plane at a leaf
        print(simplices.shape)

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

    def construct(self, leafsize):
        """Construct the BSP tree"""
        self._construct(self._root, leafsize)

    def _construct(self, node, leafsize):
        """The recursive tree constructor"""
        if len(node.index_list) > leafsize:  # Lets mess around with leaf size
            print('Index list is ', node.index_list)
            self._split(node)
            if node.pos is not None:
                print(node.index, 'Constructing a new subtree on the positive branch.')
                self._construct(node.pos, leafsize)  # Wooooo recursion!!!1!
            if node.neg is not None:
                print(node.index, 'Constructing a new subtree on the negative branch.')
                self._construct(node.neg, leafsize)  # Wooooo recursion!!!1!

    def _split(self, node):
        """Split the remaining polygons using current node selection"""
        # Check each point in each simplex
        node_simplices = np.array(self._simplices)[np.array(node.index_list)]  # NUMPYISE
        # Get every unique vertex in list
        node_vertices = np.array(self._vertices)[np.unique(node_simplices)]  # NUMPYISE

        node_equation = np.array(self._equations)[np.array(node.index)]  # NUMPYISE
        node_value = np.array(self._values)[np.array(node.index)]  # NUMPYISE
        node_results = node_equation[0]*node_vertices[:, 0] \
            + node_equation[1]*node_vertices[:, 1] \
            + node_equation[2]*node_vertices[:, 2] \
            - node_value

        node_sides = np.sign(node_results[np.searchsorted(np.unique(node_simplices), node_simplices)])

        # If all points in a simplex == zero, then append to node
        polygon_in_plane = np.all(node_sides == 0, axis=1)
        node.plane_indices = np.array(node.index_list)[polygon_in_plane].tolist()  # NUMPYISE

        # If any points in a simplex > zero, add to pos list
        polygon_positive = np.any(node_sides == 1, axis=1)
        pos_list = np.array(node.index_list)[polygon_positive].tolist()  # NUMPYISE
        # If any points in a simplex < zero, then add to neg list
        polygon_negative = np.any(node_sides == -1, axis=1)
        neg_list = np.array(node.index_list)[polygon_negative].tolist()  # NUMPYISE

        # Make child nodes
        node.set_children(pos_list, neg_list)


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
