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
    # def __init__(self, index, index_list, parent=None):
    def __init__(self, index_list, parent=None):
        self.parent = parent  # Parent node
        self.pos = None  # Positive branch
        self.neg = None  # Negative branch
        # self.index = index  # Index of the simplex/plane used for splitting
        self.index = None
        self.plane_indices = np.array([])  # Indices of any other simplices that lie in the plane
        # self.index_list = index_list[index_list != index]
        self.index_list = index_list

    def set_children(self, pos_list, neg_list):
        """Set up child tree nodes"""
        if len(pos_list) != 0:
            # Set up a new node using a random plane from the subset
            # self.pos = BSP_Node(pos_list[np.random.randint(0, pos_list.shape[0])],
            #                     pos_list, parent=self)
            self.pos = BSP_Node(pos_list, parent=self)
        if len(neg_list) != 0:
            # self.neg = BSP_Node(neg_list[np.random.randint(0, neg_list.shape[0])],
            #                     neg_list, parent=self)
            self.neg = BSP_Node(neg_list, parent=self)
        self.index_list = np.array([])


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
        Number of polygons at each leaf node. Default is one.
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

        self._vertices = vertices
        self._equations = equations
        self._values = values
        self._simplices = simplices

        # Set up root node of tree
        # self._root = BSP_Node(np.random.randint(0, self._simplices.shape[0]),
        #                       np.arange(self._simplices.shape[0]))
        self._root = BSP_Node(np.arange(self._simplices.shape[0]))

        self.construct(leafsize-1)  # Only one plane at a leaf
        print('The final tree contains %i polygons' % simplices.shape[0])

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
        if node.index_list.shape[0] > leafsize:  # FIXME: Remove this
            # print('Index list is ', node.index_list)
            self._split(node)
            if node.pos is not None:
                # print(node.index, 'Constructing a new subtree on the positive branch.')
                self._construct(node.pos, leafsize)  # Wooooo recursion!!!1!
            if node.neg is not None:
                # print(node.index, 'Constructing a new subtree on the negative branch.')
                self._construct(node.neg, leafsize)  # Wooooo recursion!!!1!

    def _split(self, node):
        """Split the remaining polygons using current node selection"""
        # Generate up to 10 indices to try out for splitting
        # Can contain duplicates. Always generates 5 indices
        index_pile = node.index_list[np.random.randint(0, node.index_list.shape[0], size=min(10, node.index_list.shape[0]))]
        # Find the best index and use that one
        for i in range(len(index_pile)):
            trial_index = index_pile[i]

            # Check each point in each simplex (Minus the one at the trial index)
            trial_node_simplices = self._simplices[node.index_list[node.index_list != trial_index]]
            # Get every unique vertex in list
            trial_node_vertices = self._vertices[np.unique(trial_node_simplices)]

            trial_node_equation = self._equations[trial_index]
            trial_node_value = self._values[trial_index]
            trial_node_results = trial_node_equation[0]*trial_node_vertices[:, 0] \
                + trial_node_equation[1]*trial_node_vertices[:, 1] \
                + trial_node_equation[2]*trial_node_vertices[:, 2] \
                - trial_node_value

            trial_node_sides = np.sign(trial_node_results[np.searchsorted(np.unique(trial_node_simplices), trial_node_simplices)]).astype(np.int)
            trial_straddle = np.logical_and(np.any(trial_node_sides > 0, axis=1), np.any(trial_node_sides < 0, axis=1))
            # Quality of a split (smaller is better)
            trial_split_q = np.count_nonzero(trial_straddle)
            if i == 0:  # Could be moved outside the loop
                index = trial_index
                node_simplices = trial_node_simplices
                node_vertices = trial_node_vertices
                node_equation = trial_node_equation
                node_value = trial_node_value
                node_results = trial_node_results
                node_sides = trial_node_sides
                straddle = trial_straddle
                split_q = trial_split_q
            elif trial_split_q < split_q:
                index = trial_index
                node_simplices = trial_node_simplices
                node_vertices = trial_node_vertices
                node_equation = trial_node_equation
                node_value = trial_node_value
                node_results = trial_node_results
                node_sides = trial_node_sides
                straddle = trial_straddle
                split_q = trial_split_q

        # Remove from parent list permanently
        node.index_list = node.index_list[node.index_list != index]
        # Set node.index to index
        node.index = index

        if split_q != 0:
            # print(node_sides[straddle])
            # Grab all the details of the simplices to split
            # Two simplex variants
            # a -> [1, 1, -1]    b -> [1, 0, -1]
            type_b = np.any(node_sides[straddle] == 0, axis=1)
            type_a = np.logical_not(type_b)
            if np.any(type_a):
                simplices_a = node_simplices[straddle][type_a]

                # -ve sum along axis 1 returns the side with a single point
                lonely_sides = -np.sum(node_sides[straddle][type_a], axis=1)
                # Split simplices of type a into those with a single node on the positive side
                simplices_a_pos = simplices_a[lonely_sides == 1]
                # And those with a single node on the negative side
                simplices_a_neg = simplices_a[lonely_sides == -1]

                # Get the points in the simplices, isolating the points on their own
                pos_lp_positions = np.where(node_sides[straddle][type_a][lonely_sides == 1] == 1)
                neg_ln_positions = np.where(node_sides[straddle][type_a][lonely_sides == -1] == -1)
                neg_lp_positions = np.where(node_sides[straddle][type_a][lonely_sides == 1] == -1)
                pos_ln_positions = np.where(node_sides[straddle][type_a][lonely_sides == -1] == 1)

                lonely_vert = np.concatenate((simplices_a_pos[pos_lp_positions], simplices_a_neg[neg_ln_positions]))
                other_vert_1 = np.concatenate((simplices_a_pos[neg_lp_positions][::2], simplices_a_neg[pos_ln_positions][::2]))
                other_vert_2 = np.concatenate((simplices_a_pos[neg_lp_positions][1::2], simplices_a_neg[pos_ln_positions][1::2]))

                # Vectors connecting the lonely vertex with the other two
                vector_1 = self._vertices[other_vert_1] - self._vertices[lonely_vert]
                vector_2 = self._vertices[other_vert_2] - self._vertices[lonely_vert]
                # FIXME: Occasionally get very small values. Probably due to floating point errors
                # These points should be on the plane I think
                # Also causes div by zero errors
                line_param_1 = ((node_value - node_equation[0]*self._vertices[lonely_vert][:, 0]
                                 - node_equation[1]*self._vertices[lonely_vert][:, 1]
                                 - node_equation[2]*self._vertices[lonely_vert][:, 2])
                                / (node_equation[0]*vector_1[:, 0]
                                   + node_equation[1]*vector_1[:, 1]
                                   + node_equation[2]*vector_1[:, 2]))
                line_param_2 = ((node_value - node_equation[0]*self._vertices[lonely_vert][:, 0]
                                 - node_equation[1]*self._vertices[lonely_vert][:, 1]
                                 - node_equation[2]*self._vertices[lonely_vert][:, 2])
                                / (node_equation[0]*vector_2[:, 0]
                                   + node_equation[1]*vector_2[:, 1]
                                   + node_equation[2]*vector_2[:, 2]))
                intersect_1 = self._vertices[lonely_vert] + vector_1*np.tile(line_param_1, (3, 1)).T
                intersect_2 = self._vertices[lonely_vert] + vector_2*np.tile(line_param_2, (3, 1)).T
            if np.any(type_b):
                simplices_b = node_simplices[straddle][type_b]
                # Get the points in the simplices, isolating the point on the plane
                plane_positions = np.where(node_sides[straddle][type_b] == 0)
                # FIXME: Probably efficiency savings to be had here
                non_plane_positions = np.where(node_sides[straddle][type_b] != 0)
                plane_vert = simplices_b[plane_positions]
                non_plane_vert = simplices_b[non_plane_positions]
                non_plane_vert_1 = non_plane_vert[::2]
                non_plane_vert_2 = non_plane_vert[1::2]

                # Just one vector and intersection this time around
                vector_b = self._vertices[non_plane_vert_1] - self._vertices[non_plane_vert_2]

                line_param_b = ((node_value - node_equation[0]*self._vertices[non_plane_vert_2][:, 0]
                                 - node_equation[1]*self._vertices[non_plane_vert_2][:, 1]
                                 - node_equation[2]*self._vertices[non_plane_vert_2][:, 2])
                                / (node_equation[0]*vector_b[:, 0]
                                   + node_equation[1]*vector_b[:, 1]
                                   + node_equation[2]*vector_b[:, 2]))

                intersect_b = self._vertices[non_plane_vert_2] + vector_b*np.tile(line_param_b, (3, 1)).T

            # if np.any(type_a) and np.any(type_b):
                # Append the new simplex indices of both types to node_simplices

        # If all points in a simplex == zero, then append to node
        polygon_in_plane = np.all(node_sides == 0, axis=1)
        node.plane_indices = node.index_list[polygon_in_plane]

        # If any points in a simplex > zero, add to pos list
        polygon_positive = np.any(node_sides == 1, axis=1)
        pos_list = node.index_list[polygon_positive]
        # If any points in a simplex < zero, then add to neg list
        polygon_negative = np.any(node_sides == -1, axis=1)
        neg_list = node.index_list[polygon_negative]

        # Make child nodes
        node.set_children(pos_list, neg_list)


class PolyMesh:
    """
    A polygonal mesh of points. Used to express a non-concave (2.5D) surface
    in 3D space. This surface is indexed and can be used to rapidly find all
    grid nodes within a given axial distance of any facet. It can be queried to
    return all points which are within the area of effect of the surface and
    the respective distances to the surface in each axial direction. It is
    intended for use in impementing variants of the immersed boundary method.

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

    def query(self, q_points):
        """
        Query a set of points to find axial distances to the boundary surface.
        Distances are returned with units of dx (grid increment).

        Parameters
        ----------
        q_points : array_like
            Array of points to query grouped as [[x, y, z], [x, y, z], ...]

        Returns
        -------
        z_dist : ndarray
            Distance to the surface in the z direction for the respective points
            in q_points. Values of NaN indicate that the surface does not
            occlude the point in this direction.
        y_pos_dist : ndarray
            Distances to the surface in the positive y direction. Same behaviours
            as z_dist
        y_neg_dist : ndarray
            Distances to the surface in the negative y direction. Same behaviours
            as z_dist
        x_pos_dist : ndarray
            Distances to the surface in the positive x direction. Same behaviours
            as z_dist
        x_neg_dist : ndarray
            Distances to the surface in the negative x direction. Same behaviours
            as z_dist
        """
        self._query_points = np.array(q_points, dtype=np.float32)
        self._query_points /= self._grid.spacing  # Convert to grid-index space

        # Create an array of indices to actually send through the tree
        # This means that respective positions can be retained without searching
        full_indices = np.arange(self._query_points.shape[0])

        # Initialise arrays for axial distances to be stored in
        self._z_dist = np.empty((self._query_points.shape[0]))
        self._z_dist[:] = np.nan
        self._y_pos_dist = np.empty((self._query_points.shape[0]))
        self._y_pos_dist[:] = np.nan
        self._y_neg_dist = np.empty((self._query_points.shape[0]))
        self._y_neg_dist[:] = np.nan
        self._x_pos_dist = np.empty((self._query_points.shape[0]))
        self._x_pos_dist[:] = np.nan
        self._x_neg_dist = np.empty((self._query_points.shape[0]))
        self._x_neg_dist[:] = np.nan

        # Start the traversal
        print('Starting query')
        self._query(self._tree._root, full_indices)

        print('z distances', self._z_dist)
        print('y pos distances', self._y_pos_dist)
        print('y neg distances', self._y_neg_dist)
        print('x pos distances', self._x_pos_dist)
        print('x neg distances', self._x_neg_dist)

    def _query(self, node, query_indices):
        """The recursive traversal for querying the tree"""
        if node.plane_indices.size != 0:
            print('There are extra indices at this node', node.plane_indices)
        # Want to find the half spaces of all the query points
        qp = self._query_points[query_indices]  # Points to find half spaces of
        node_equation = self._equations[node.index]
        node_value = self._values[node.index]
        node_results = node_equation[0]*qp[:, 0] \
            + node_equation[1]*qp[:, 1] \
            + node_equation[2]*qp[:, 2] \
            - node_value

        point_spaces = np.sign(node_results)  # Reduces half spaces to -1, 0, 1
        # Need to figure out what to do with points that lie in a plane
        # Also need to figure out what to do about additonal planes at the node

        # Check near sides
        # Process the ones where the positive is the near side
        if node.pos is not None and query_indices[point_spaces == 1].shape[0] != 0:
            self._query(node.pos, query_indices[point_spaces == 1])

        # Process the ones where the negative is the near side
        if node.neg is not None and query_indices[point_spaces == -1].shape[0] != 0:
            self._query(node.neg, query_indices[point_spaces == -1])

        # Z axis
        # Check occlusion of points with no distances
        no_z_distance = np.isnan(self._z_dist[query_indices])
        if np.nonzero(no_z_distance) != 0:  # No point checking if all distances filled
            z_occluded = self._occludes(self._query_points[query_indices[no_z_distance]],
                                        node.index, 'z')
            # Measure distance to occluded points
            if np.nonzero(z_occluded) != 0:
                new_z_dists = self._distance(self._query_points[query_indices[no_z_distance][z_occluded]],
                                             node.index, 'z')
                self._z_dist[query_indices[no_z_distance][z_occluded]] = new_z_dists

        # Y axis
        # Check occlusion of points with no distances
        no_y_distance = np.logical_or(np.isnan(self._y_pos_dist[query_indices]),
                                      np.isnan(self._y_neg_dist[query_indices]))
        if np.nonzero(no_y_distance) != 0:  # No point checking if all distances filled
            y_occluded = self._occludes(self._query_points[query_indices[no_y_distance]],
                                        node.index, 'y')
            # Measure distance to occluded points
            if np.nonzero(y_occluded) != 0:
                new_y_dists = self._distance(self._query_points[query_indices[no_y_distance][y_occluded]],
                                             node.index, 'y')
                self._y_pos_dist[query_indices[no_y_distance][y_occluded][new_y_dists >= 0]] = new_y_dists[new_y_dists >= 0]
                self._y_neg_dist[query_indices[no_y_distance][y_occluded][new_y_dists <= 0]] = new_y_dists[new_y_dists <= 0]

        # X axis
        # Check occlusion of points with no distances
        no_x_distance = np.logical_or(np.isnan(self._x_pos_dist[query_indices]),
                                      np.isnan(self._x_neg_dist[query_indices]))
        if np.nonzero(no_x_distance) != 0:  # No point checking if all distances filled
            x_occluded = self._occludes(self._query_points[query_indices[no_x_distance]],
                                        node.index, 'x')
            # Measure distance to occluded points
            if np.nonzero(x_occluded) != 0:
                new_x_dists = self._distance(self._query_points[query_indices[no_x_distance][x_occluded]],
                                             node.index, 'x')
                self._x_pos_dist[query_indices[no_x_distance][x_occluded][new_x_dists >= 0]] = new_x_dists[new_x_dists >= 0]
                self._x_neg_dist[query_indices[no_x_distance][x_occluded][new_x_dists <= 0]] = new_x_dists[new_x_dists <= 0]

        # Check far sides
        # Process the ones where the positive is the near side
        if node.neg is not None and query_indices[point_spaces == 1].shape[0] != 0:
            self._query(node.neg, query_indices[point_spaces == 1])

        # Process the ones where the negative is the near side
        if node.pos is not None and query_indices[point_spaces == -1].shape[0] != 0:
            self._query(node.pos, query_indices[point_spaces == -1])

    def _occludes(self, pt, simplex, axis):
        """
        A function to check whether a set of points are occluded by a simplex
        on a specified axis.
        """
        # FIXME: Make this check for occlusion on an array of simplices
        # We are fine down to line 428 atm

        vertices = self._points[self._simplices[simplex]]
        if axis == 'x':
            # p0, p1, p2 are vertices, p is the array of test points
            p0, p1, p2 = vertices

            area = -p1[1]*p2[2] + p0[1]*(-p1[2] + p2[2]) + p0[2]*(p1[1] - p2[1]) + p1[2]*p2[1]
            if area == 0:  # This plane is axially aligned
                false_array = np.empty((pt.shape[0]), dtype=np.bool)
                false_array[:] = False
                return false_array
            s = (p0[1]*p2[2] - p0[2]*p2[1] + (p2[1] - p0[1])*pt[:, 2] + (p0[2] - p2[2])*pt[:, 1])/area
            t = (p0[2]*p1[1] - p0[1]*p1[2] + (p0[1] - p1[1])*pt[:, 2] + (p1[2] - p0[2])*pt[:, 1])/area

            return np.logical_and.reduce((s > 0, t > 0, 1-s-t > 0))

        if axis == 'y':
            # p0, p1, p2 are vertices, p is the array of test points
            p0, p1, p2 = vertices

            area = -p1[0]*p2[2] + p0[0]*(-p1[2] + p2[2]) + p0[2]*(p1[0] - p2[0]) + p1[2]*p2[0]
            if area == 0:  # This plane is axially aligned
                false_array = np.empty((pt.shape[0]), dtype=np.bool)
                false_array[:] = False
                return false_array
            s = (p0[0]*p2[2] - p0[2]*p2[0] + (p2[0] - p0[0])*pt[:, 2] + (p0[2] - p2[2])*pt[:, 0])/area
            t = (p0[2]*p1[0] - p0[0]*p1[2] + (p0[0] - p1[0])*pt[:, 2] + (p1[2] - p0[2])*pt[:, 0])/area

            return np.logical_and.reduce((s > 0, t > 0, 1-s-t > 0))

        if axis == 'z':
            # p0, p1, p2 are vertices, p is the array of test points
            p0, p1, p2 = vertices

            area = -p1[0]*p2[1] + p0[0]*(-p1[1] + p2[1]) + p0[1]*(p1[0] - p2[0]) + p1[1]*p2[0]
            if area == 0:  # This plane is axially aligned
                false_array = np.empty((pt.shape[0]), dtype=np.bool)
                false_array[:] = False
                return false_array
            s = (p0[0]*p2[1] - p0[1]*p2[0] + (p2[0] - p0[0])*pt[:, 1] + (p0[1] - p2[1])*pt[:, 0])/area
            t = (p0[1]*p1[0] - p0[0]*p1[1] + (p0[0] - p1[0])*pt[:, 1] + (p1[1] - p0[1])*pt[:, 0])/area

            return np.logical_and.reduce((s > 0, t > 0, 1-s-t > 0))

    def _distance(self, pt, simplex, axis):
        """
        Measures the axial distance between points and a simplex along a specified
        axis.
        """
        A, B, C = self._equations[simplex]
        D = self._values[simplex]
        if axis == 'z':
            dist = (D - A*pt[:, 0] - B*pt[:, 1])/C
            return pt[:, 2] - dist
        if axis == 'y':
            dist = (D - A*pt[:, 0] - C*pt[:, 2])/B
            return pt[:, 1] - dist
        if axis == 'x':
            dist = (D - B*pt[:, 1] - C*pt[:, 2])/A
            return pt[:, 0] - dist

    @property
    def lastquery(self):  # Maybe make this return the points too
        """The distances from the last query of the surface"""
        distances = (self._z_dist, self._y_pos_dist, self._y_neg_dist,
                     self._x_pos_dist, self._x_neg_dist)
        return distances
