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
        self.plane_indices = []  # Indices of any other simplices that lie in the plane
        self.index_list = index_list  # Indices of all the simplices at the node
        self.index_list.remove(self.index)  # Remove the index from child list

    def set_children(self, pos_list, neg_list):
        """Set up child tree nodes"""
        if len(pos_list) != 0:
            # Set up a new node using a random plane from the subset
            self.pos = BSP_Node(pos_list[np.random.randint(0, len(pos_list))],
                                pos_list, parent=self)
        if len(neg_list) != 0:
            self.neg = BSP_Node(neg_list[np.random.randint(0, len(neg_list))],
                                neg_list, parent=self)
        # del self.index_list  # Maybe do something else here like replace with None
        self.index_list = []


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

        self.construct(0)  # Only one plane at a leaf

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
        # if len(node.index_list) != 0:
        if len(node.index_list) > 10:  # Lets mess around with leaf size
            self._split(node)
            if node.pos is not None:
                print(node.index, 'Constructing a new subtree on the positive branch.')
                self._construct(node.pos)  # Wooooo recursion!!!1!
            if node.neg is not None:
                print(node.index, 'Constructing a new subtree on the negative branch.')
                self._construct(node.neg)  # Wooooo recursion!!!1!

    def _split(self, node):
        """Split the remaining polygons using current node selection"""
        # Initialise pos and neg lists for the split
        pos_list = []
        neg_list = []

        # Check each point in each simplex
        node_simplices = np.array(self._simplices)[np.array(node.index_list)]
        # Get every unique vertex in list
        node_vertices = np.array(self._vertices)[np.unique(node_simplices)]

        # FIXME: These could probably be stored somehow
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
        not_straddle = np.logical_or(is_all_negative, is_all_positive)  # Useful later
        straddle = np.logical_not(not_straddle)

        # Get the positions of the local indices of the simplices which straddle the plane
        simplices_to_split = straddle.nonzero()[0]
        # FIXME: redo with numpy rather than loops
        for simplex in simplices_to_split:
            # If one is on the plane
            if np.any(node_sides[simplex] == 0):
                # One of the vertices is located on the plane
                # Only need to find one intersection point in this case
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
                # Add the new simplices to respective pos and neg lists
                # node.index_list.extend(new_simplices_indices)
                pos_list.append(new_simplices_indices[0])  # See order of simplex assembly
                neg_list.append(new_simplices_indices[1])  # See order of simplex assembly

                # Need to append new equations and values for my new simplices
                new_simplices_equation = self._equations[np.array(node.index_list)[simplex]]
                self._equations.extend([new_simplices_equation, new_simplices_equation])
                new_simplices_value = self._values[np.array(node.index_list)[simplex]]
                self._values.extend([new_simplices_value, new_simplices_value])

            else:
                # None of the nodes are located on the plane
                # There will be two intersection points
                # Find the node on its own
                _, vertex_positions, counts = np.unique(node_sides[simplex],
                                                        return_index=True,
                                                        return_counts=True)
                # Index within the simplex of the node on its own
                lonely_pos = vertex_positions[np.where(counts == 1)][0]
                # Indices within the simplex of the paired nodes
                popular_pos = np.setdiff1d([0, 1, 2], lonely_pos)

                # The physical positions of these vertices
                lonely_vertex = np.array(self._vertices)[node_simplices[simplex][lonely_pos]]
                popular_vertices = np.array(self._vertices)[node_simplices[simplex][popular_pos]]

                # Note that line_m has a slightly different form in this case
                line_m = popular_vertices - lonely_vertex
                # Parameter value where line meets plane
                line_params = (node_value - np.dot(node_equation, lonely_vertex)) \
                    / np.dot(node_equation, line_m.T)

                # The new vertices that need adding to split the polygon
                intersection_A = line_m[0]*line_params[0] + lonely_vertex
                intersection_B = line_m[1]*line_params[1] + lonely_vertex
                # Figure out indices of new vertices
                new_vertex_indices = [len(self._vertices), len(self._vertices)+1]
                # Append the new vertices to the master vertex list
                self._vertices.extend([intersection_A, intersection_B])
                # Figure out the indices of the three new simplices
                new_simplices_indices = [len(self._simplices), len(self._simplices)+1, len(self._simplices)+2]
                # Add the three new simplices to the master simplex list
                # The simplex on the "lonely side"
                new_simplex_1 = [node_simplices[simplex][lonely_pos]] + new_vertex_indices
                # The other two simplices
                new_simplex_2 = node_simplices[simplex][popular_pos].tolist() + [new_vertex_indices[0]]
                new_simplex_3 = [node_simplices[simplex][popular_pos[1]]] + new_vertex_indices
                self._simplices.extend([new_simplex_1, new_simplex_2, new_simplex_3])
                # Add the new simplices to respective pos and neg lists
                # node.index_list.extend(new_simplices_indices)
                if node_sides[simplex][lonely_pos] == 1:
                    pos_list.append(new_simplices_indices[0])
                    neg_list.extend(new_simplices_indices[1:])
                else:
                    pos_list.extend(new_simplices_indices[1:])
                    neg_list.append(new_simplices_indices[0])

                # Need to append new equations and values for my new simplices
                new_simplices_equation = self._equations[np.array(node.index_list)[simplex]]
                self._equations.extend([new_simplices_equation, new_simplices_equation, new_simplices_equation])
                new_simplices_value = self._values[np.array(node.index_list)[simplex]]
                self._values.extend([new_simplices_value, new_simplices_value, new_simplices_value])

        # If all points in a simplex == zero, then append to node
        polygon_in_plane = np.all(node_sides == 0, axis=1)
        node.plane_indices.extend(np.array(node.index_list)[polygon_in_plane].tolist())

        # If all points in a simplex > zero, add to pos list
        polygon_positive = np.logical_and(is_all_positive, np.logical_not(polygon_in_plane))
        pos_list.extend(np.array(node.index_list)[polygon_positive].tolist())
        # If all points in a simplex < zero, then add to neg list
        polygon_negative = np.logical_and(is_all_negative, np.logical_not(polygon_in_plane))
        neg_list.extend(np.array(node.index_list)[polygon_negative].tolist())

        # print('Pos list post is', pos_list)
        # print('Neg list post is', neg_list)
        # print('Additional polygons at node', node.plane_indices)

        # Make children
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
