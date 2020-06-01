"""
Data structures to be used in the implementation of immersed boundaries of any
sort.
"""

import numpy as np

from scipy.spatial import Delaunay

__all__ = ['PolySurface']


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
        self._root = BSP_Node(np.arange(self._simplices.shape[0]))

        # Temp quality measure
        self._unbalanced_split = 0
        self._balanced_split = 0

        self.construct(leafsize-1)  # Only one plane at a leaf
        print('The initial polygon count was %i' % simplices.shape[0])
        print('The final tree contains %i polygons' % self._simplices.shape[0])
        print('The number of balanced splits was', self._balanced_split)
        print('The number of unbalanced splits was', self._unbalanced_split)

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
            # Look for changes across this function

            self._split(node)
            if node.pos is not None and node.neg is not None:
                self._balanced_split += 1
            elif (node.pos is not None and node.neg is None) or (node.pos is None and node.neg is not None):
                self._unbalanced_split += 1
            if node.pos is not None:
                # print(node.index, 'Constructing a new subtree on the positive branch.')
                self._construct(node.pos, leafsize)  # Wooooo recursion!!!1!
            if node.neg is not None:
                # print(node.index, 'Constructing a new subtree on the negative branch.')
                self._construct(node.neg, leafsize)  # Wooooo recursion!!!1!

    def _split(self, node):
        """Split the remaining polygons using current node selection"""
        # Heuristic weights
        # split_weight = 1
        # balance_weight = 0.3

        # Generate up to 10 indices to try out for splitting
        # Can contain duplicates. Generates up to 10 indices
        index_pile = node.index_list[np.random.randint(0, node.index_list.shape[0], size=min(10, node.index_list.shape[0]))]
        # Find the best index and use that one
        for i in range(len(index_pile)):
            trial_index = index_pile[i]

            # Check each point in each simplex (Minus the one at the trial index)
            # These are temporarily set to copy
            trial_node_simplices = np.copy(self._simplices[node.index_list[node.index_list != trial_index]])
            # Get every unique vertex in list
            trial_node_vertices = np.copy(self._vertices[np.unique(trial_node_simplices)])

            # I think the wheels fall off because trial_node_equation is still an array
            # So this is just a pointer to the original array, rather than a copy
            # Thus the original array gets modified
            # But where is this getting modified?
            trial_node_equation = np.copy(self._equations[trial_index])
            trial_node_value = self._values[trial_index]
            trial_node_results = trial_node_equation[0]*trial_node_vertices[:, 0] \
                + trial_node_equation[1]*trial_node_vertices[:, 1] \
                + trial_node_equation[2]*trial_node_vertices[:, 2] \
                - trial_node_value

            # The .round() here exists to kill any div by zero errors
            # These come about when a vertex on the plane gets pushed to one halfspace by float errors
            # This causes the plane to be earmarked for splitting then produces a div by zero
            # Might want increasing in the future, but fine for now
            # 2 is safest. If recursion limit hit, this is probably why
            trial_node_sides = np.sign(trial_node_results.round(2)[np.searchsorted(np.unique(trial_node_simplices), trial_node_simplices)]).astype(np.int)
            trial_straddle = np.logical_and(np.any(trial_node_sides > 0, axis=1), np.any(trial_node_sides < 0, axis=1))
            # Quality of a split (smaller is better)
            trial_split_q = np.count_nonzero(trial_straddle)
            # Normalised mismatch between size of the two branches
            trial_balance_q = abs(np.count_nonzero(trial_node_sides == 1) - np.count_nonzero(trial_node_sides == -1))  # /(trial_node_simplices.shape[0] + 1)
            # trial_heuristic_q = trial_split_q*trial_balance_q
            trial_heuristic_q = max(trial_split_q, 1)*max(trial_balance_q, 1)
            # print('Weighted trial_split_q is', split_weight*trial_split_q)
            # print('Weighted trial_balance_q is', balance_weight*trial_balance_q)
            # print('The unweighted product is', trial_split_q*trial_balance_q)
            if i == 0:  # Could be moved outside the loop
                index = trial_index
                node_simplices = trial_node_simplices
                node_equation = trial_node_equation
                node_value = trial_node_value
                node_sides = trial_node_sides
                straddle = trial_straddle
                split_q = trial_split_q
                heuristic_q = trial_heuristic_q
            elif trial_heuristic_q < heuristic_q:
                index = trial_index
                node_simplices[:] = trial_node_simplices[:]
                node_equation[:] = trial_node_equation[:]
                node_value = trial_node_value
                node_sides[:] = trial_node_sides[:]
                straddle[:] = trial_straddle[:]
                split_q = trial_split_q
                heuristic_q = trial_heuristic_q

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
                # I think I forgot to swap to self._simplices

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

                # I think this produces a shuffling effect
                lonely_vert = np.concatenate((simplices_a_pos[pos_lp_positions], simplices_a_neg[neg_ln_positions]))
                other_vert_1 = np.concatenate((simplices_a_pos[neg_lp_positions][::2], simplices_a_neg[pos_ln_positions][::2]))
                other_vert_2 = np.concatenate((simplices_a_pos[neg_lp_positions][1::2], simplices_a_neg[pos_ln_positions][1::2]))

                # Vectors connecting the lonely vertex with the other two
                vector_1 = self._vertices[other_vert_1] - self._vertices[lonely_vert]
                vector_2 = self._vertices[other_vert_2] - self._vertices[lonely_vert]
                # FIXME: Occasionally get very small values. Probably due to floating point errors
                # Want to do something to check the value if everything is going to catch fire
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

            if np.any(type_a) and np.any(type_b):  # FIXME: Will occasionally go wonky
                # Need to extend self._equations and self._values
                # equations_a and values_a have the concatenate to assemble them in the same order as the simplices
                # This is due to the way the type A simplices are split and reconcatenated
                # equations_a = np.tile(self._equations[node.index_list[straddle][type_a]], (3, 1))
                equations_a = np.tile(np.concatenate((self._equations[node.index_list[straddle][type_a][lonely_sides == 1]],
                                                      self._equations[node.index_list[straddle][type_a][lonely_sides == -1]])), (3, 1))
                equations_b = np.tile(self._equations[node.index_list[straddle][type_b]], (2, 1))
                self._equations = np.concatenate((self._equations, equations_a, equations_b))

                # values_a = np.tile(self._values[node.index_list[straddle][type_a]], 3)
                values_a = np.tile(np.concatenate((self._values[node.index_list[straddle][type_a][lonely_sides == 1]],
                                                   self._values[node.index_list[straddle][type_a][lonely_sides == -1]])), 3)
                values_b = np.tile(self._values[node.index_list[straddle][type_b]], 2)
                self._values = np.concatenate((self._values, values_a, values_b))

                # Remove the indices of the simplices that were split from node_simplices
                node.index_list = node.index_list[np.logical_not(straddle)]
                # Append the new simplex indices of both types to node_simplices
                node.index_list = np.concatenate((node.index_list,
                                                  np.arange(self._simplices.shape[0],
                                                            self._simplices.shape[0]
                                                            + 3*intersect_1.shape[0]
                                                            + 2*intersect_b.shape[0])))
                # Append new simplices to self._simplices
                # v0, v2, p0 (other_vert_1, other_vert_2, intersect_1)
                new_simplices_a1 = np.array((other_vert_1, other_vert_2,
                                             np.arange(self._vertices.shape[0],
                                                       self._vertices.shape[0] + intersect_1.shape[0]))).T
                # v1, p0, p1 (lonely_vert, intersect_1, intersect_2)
                new_simplices_a2 = np.array((lonely_vert,
                                             np.arange(self._vertices.shape[0],
                                                       self._vertices.shape[0] + intersect_1.shape[0]),
                                             np.arange(self._vertices.shape[0] + intersect_1.shape[0],
                                                       self._vertices.shape[0] + 2*intersect_2.shape[0]))).T
                # v2, p0, p1 (other_vert_2, intersect_1, intersect_2)
                new_simplices_a3 = np.array((other_vert_2,
                                             np.arange(self._vertices.shape[0],
                                                       self._vertices.shape[0] + intersect_1.shape[0]),
                                             np.arange(self._vertices.shape[0] + intersect_1.shape[0],
                                                       self._vertices.shape[0] + 2*intersect_2.shape[0]))).T
                # v0, v1, p0 (plane_vert, non_plane_vert_1, intersect_b)
                new_simplices_b1 = np.array((plane_vert, non_plane_vert_1,
                                             np.arange(self._vertices.shape[0] + 2*intersect_2.shape[0],
                                                       self._vertices.shape[0] + 2*intersect_2.shape[0] + intersect_b.shape[0]))).T
                # new_simplices_b2  # v0, v2, p0 (plane_vert, non_plane_vert_2, intersect_b)
                new_simplices_b2 = np.array((plane_vert, non_plane_vert_2,
                                             np.arange(self._vertices.shape[0] + 2*intersect_2.shape[0],
                                                       self._vertices.shape[0] + 2*intersect_2.shape[0] + intersect_b.shape[0]))).T
                # Concatenate the above in order with self._simplices
                self._simplices = np.concatenate((self._simplices,
                                                 new_simplices_a1, new_simplices_a2,
                                                 new_simplices_a3, new_simplices_b1,
                                                 new_simplices_b2))
                # Append new vertices to self._vertices
                self._vertices = np.concatenate((self._vertices, intersect_1, intersect_2, intersect_b))

                # Extend node_sides with new polygons (can do with an any check, since they are never going to straddle)
                ns_a1_sides = np.sign(node_equation[0]*self._vertices[new_simplices_a1[:, 0]]
                                      + node_equation[1]*self._vertices[new_simplices_a1[:, 1]]
                                      + node_equation[2]*self._vertices[new_simplices_a1[:, 2]]
                                      - node_value)
                ns_a2_sides = np.sign(node_equation[0]*self._vertices[new_simplices_a2[:, 0]]
                                      + node_equation[1]*self._vertices[new_simplices_a2[:, 1]]
                                      + node_equation[2]*self._vertices[new_simplices_a2[:, 2]]
                                      - node_value)
                ns_a3_sides = np.sign(node_equation[0]*self._vertices[new_simplices_a3[:, 0]]
                                      + node_equation[1]*self._vertices[new_simplices_a3[:, 1]]
                                      + node_equation[2]*self._vertices[new_simplices_a3[:, 2]]
                                      - node_value)
                ns_b1_sides = np.sign(node_equation[0]*self._vertices[new_simplices_b1[:, 0]]
                                      + node_equation[1]*self._vertices[new_simplices_b1[:, 1]]
                                      + node_equation[2]*self._vertices[new_simplices_b1[:, 2]]
                                      - node_value)
                ns_b2_sides = np.sign(node_equation[0]*self._vertices[new_simplices_b2[:, 0]]
                                      + node_equation[1]*self._vertices[new_simplices_b2[:, 1]]
                                      + node_equation[2]*self._vertices[new_simplices_b2[:, 2]]
                                      - node_value)
                node_sides = node_sides[np.logical_not(straddle)]  # Remove split simplices sides
                node_sides = np.concatenate((node_sides, ns_a1_sides, ns_a2_sides,
                                             ns_a3_sides, ns_b1_sides, ns_b2_sides))

            elif np.any(type_a):
                # Need to extend self._equations and self._values
                equations_a = np.tile(np.concatenate((self._equations[node.index_list[straddle][type_a][lonely_sides == 1]],
                                                      self._equations[node.index_list[straddle][type_a][lonely_sides == -1]])), (3, 1))
                self._equations = np.concatenate((self._equations, equations_a))

                values_a = np.tile(np.concatenate((self._values[node.index_list[straddle][type_a][lonely_sides == 1]],
                                                   self._values[node.index_list[straddle][type_a][lonely_sides == -1]])), 3)
                self._values = np.concatenate((self._values, values_a))

                # Remove the indices of the simplices that were split from node_simplices
                node.index_list = node.index_list[np.logical_not(straddle)]
                # Append the new simplex indices of both types to node_simplices
                node.index_list = np.concatenate((node.index_list,
                                                  np.arange(self._simplices.shape[0],
                                                            self._simplices.shape[0]
                                                            + 3*intersect_1.shape[0])))
                # Append new simplices to self._simplices
                # v0, v2, p0 (other_vert_1, other_vert_2, intersect_1)
                new_simplices_a1 = np.array((other_vert_1, other_vert_2,
                                             np.arange(self._vertices.shape[0],
                                                       self._vertices.shape[0] + intersect_1.shape[0]))).T
                # v1, p0, p1 (lonely_vert, intersect_1, intersect_2)
                new_simplices_a2 = np.array((lonely_vert,
                                             np.arange(self._vertices.shape[0],
                                                       self._vertices.shape[0] + intersect_1.shape[0]),
                                             np.arange(self._vertices.shape[0] + intersect_1.shape[0],
                                                       self._vertices.shape[0] + 2*intersect_2.shape[0]))).T
                # v2, p0, p1 (other_vert_2, intersect_1, intersect_2)
                new_simplices_a3 = np.array((other_vert_2,
                                             np.arange(self._vertices.shape[0],
                                                       self._vertices.shape[0] + intersect_1.shape[0]),
                                             np.arange(self._vertices.shape[0] + intersect_1.shape[0],
                                                       self._vertices.shape[0] + 2*intersect_2.shape[0]))).T
                # Concatenate the above in order with self._simplices
                self._simplices = np.concatenate((self._simplices,
                                                 new_simplices_a1, new_simplices_a2,
                                                 new_simplices_a3))
                # Append new vertices to self._vertices
                self._vertices = np.concatenate((self._vertices, intersect_1, intersect_2))

                # Extend node_sides with new polygons (can do with an any check, since they are never going to straddle)
                ns_a1_sides = np.sign(node_equation[0]*self._vertices[new_simplices_a1[:, 0]]
                                      + node_equation[1]*self._vertices[new_simplices_a1[:, 1]]
                                      + node_equation[2]*self._vertices[new_simplices_a1[:, 2]]
                                      - node_value)
                ns_a2_sides = np.sign(node_equation[0]*self._vertices[new_simplices_a2[:, 0]]
                                      + node_equation[1]*self._vertices[new_simplices_a2[:, 1]]
                                      + node_equation[2]*self._vertices[new_simplices_a2[:, 2]]
                                      - node_value)
                ns_a3_sides = np.sign(node_equation[0]*self._vertices[new_simplices_a3[:, 0]]
                                      + node_equation[1]*self._vertices[new_simplices_a3[:, 1]]
                                      + node_equation[2]*self._vertices[new_simplices_a3[:, 2]]
                                      - node_value)
                node_sides = node_sides[np.logical_not(straddle)]  # Remove split simplices sides
                node_sides = np.concatenate((node_sides, ns_a1_sides, ns_a2_sides,
                                             ns_a3_sides))

            elif np.any(type_b):
                equations_b = np.tile(self._equations[node.index_list[straddle][type_b]], (2, 1))
                self._equations = np.concatenate((self._equations, equations_b))

                values_b = np.tile(self._values[node.index_list[straddle][type_b]], 2)
                self._values = np.concatenate((self._values, values_b))

                # Remove the indices of the simplices that were split from node_simplices
                node.index_list = node.index_list[np.logical_not(straddle)]
                # Append the new simplex indices of both types to node_simplices
                node.index_list = np.concatenate((node.index_list,
                                                  np.arange(self._simplices.shape[0],
                                                            self._simplices.shape[0]
                                                            + 2*intersect_b.shape[0])))

                # Append new simplices to self._simplices
                # v0, v1, p0 (plane_vert, non_plane_vert_1, intersect_b)
                new_simplices_b1 = np.array((plane_vert, non_plane_vert_1,
                                             np.arange(self._vertices.shape[0],
                                                       self._vertices.shape[0] + intersect_b.shape[0]))).T

                # v0, v2, p0 (plane_vert, non_plane_vert_2, intersect_b)
                new_simplices_b2 = np.array((plane_vert, non_plane_vert_2,
                                             np.arange(self._vertices.shape[0],
                                                       self._vertices.shape[0] + intersect_b.shape[0]))).T

                # Concatenate the above in order with self._simplices
                self._simplices = np.concatenate((self._simplices,
                                                 new_simplices_b1,
                                                 new_simplices_b2))
                # Append new vertices to self._vertices
                self._vertices = np.concatenate((self._vertices, intersect_b))

                # Extend node_sides with new polygons (can do with an any check, since they are never going to straddle)
                ns_b1_sides = np.sign(node_equation[0]*self._vertices[new_simplices_b1[:, 0]]
                                      + node_equation[1]*self._vertices[new_simplices_b1[:, 1]]
                                      + node_equation[2]*self._vertices[new_simplices_b1[:, 2]]
                                      - node_value)
                ns_b2_sides = np.sign(node_equation[0]*self._vertices[new_simplices_b2[:, 0]]
                                      + node_equation[1]*self._vertices[new_simplices_b2[:, 1]]
                                      + node_equation[2]*self._vertices[new_simplices_b2[:, 2]]
                                      - node_value)
                node_sides = node_sides[np.logical_not(straddle)]  # Remove split simplices sides
                node_sides = np.concatenate((node_sides, ns_b1_sides, ns_b2_sides))

        # If all points in a simplex == zero, then append to node
        # Could just do this with a not?
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


class PolySurface:
    """
    A polygonal surface of points. Used to express a non-concave (2.5D) surface
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
        assert len(grid.dimensions) == 3, "PolySurface is for 3D grids only"
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

    # FIXME: I don't need a load of these and should get rid

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

    # FIXME: Maybe want the tree as a cached property

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
        mesh = Delaunay(self._points[:, :2])  # , qhull_options="QJ")
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

    def fd_node_sides(self):
        """
        Check all nodes in the grid to check if they are outside or inside the
        boundary surface.

        Returns
        -------
        positive_mask : ndarray
            A boolean mask matching the size of the grid. True where the respective
            node lies on the positive side of the boundary surface.
        """
        grid_mesh = np.meshgrid(np.arange(self._grid.shape[0]),
                                np.arange(self._grid.shape[1]),
                                np.arange(self._grid.shape[2]))
        grid_x, grid_y, grid_z = grid_mesh
        self._grid_nodes = np.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T
        # Create an array of indices to send through the tree
        # This means that respective positions can be retained without searching
        full_indices = np.arange(self._grid_nodes.shape[0])
        self._positive_mask = np.zeros(self._grid.shape, dtype=np.bool)

        self._depth_measure = []
        print('Starting query')
        self._fd_node_sides(self._tree._root, full_indices, 0)
        print(self._depth_measure)
        print(len(self._depth_measure))
        print('The maximum tree depth is', max(self._depth_measure))
        print('The average tree depth is', np.mean(self._depth_measure))
        print('The ideal tree depth is approximately', np.log2(self._tree.simplices.shape[0]))
        return self._positive_mask

    def _fd_node_sides(self, node, query_indices, depth):
        """
        The recursive traversal for determining which side of the boundary nodes
        lie on.
        """
        qp = self._grid_nodes[query_indices]  # Points to find half spaces of
        if node.pos is not None or node.neg is not None:
            node_equation = self._tree._equations[node.index]
            node_value = self._tree._values[node.index]
            node_results = node_equation[0]*qp[:, 0] \
                + node_equation[1]*qp[:, 1] \
                + node_equation[2]*qp[:, 2] \
                - node_value

            point_spaces = np.sign(node_results.round(2))  # Reduces half spaces to -1, 0, 1

            if node.pos is not None and np.any(point_spaces == 1):
                self._fd_node_sides(node.pos, query_indices[point_spaces == 1], depth+1)

            if node.neg is not None and np.any(point_spaces == -1):
                self._fd_node_sides(node.neg, query_indices[point_spaces == -1], depth+1)

            # Points on the plane are on positive side
            plane_coords = self._grid_nodes[query_indices[point_spaces == 0]]
            self._positive_mask[plane_coords[:, 0],
                                plane_coords[:, 1],
                                plane_coords[:, 2]] = True
        else:
            # Find the vectors from the first vertex of the simplex to the query nodes
            position_vectors = qp - self._tree._vertices[self._tree._simplices[node.index]][0]
            # Dot this with the normal vector
            dot_normal = np.dot(position_vectors, self._tree._equations[node.index])
            dot_normal_sides = np.sign(dot_normal)
            # Set the entry in self._positive_mask
            positive_coords = self._grid_nodes[query_indices[dot_normal_sides >= 0]]
            self._positive_mask[positive_coords[:, 0],
                                positive_coords[:, 1],
                                positive_coords[:, 2]] = True
            self._depth_measure.append(depth)

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

        return self._z_dist, self._y_pos_dist, self._y_neg_dist, self._x_pos_dist, self._x_neg_dist

    def _query(self, node, query_indices):
        """The recursive traversal for querying the tree"""
        # if node.plane_indices.size != 0:
        #     print('There are extra indices at this node', node.plane_indices)
        # Want to find the half spaces of all the query points
        qp = self._query_points[query_indices]  # Points to find half spaces of
        node_equation = self._tree._equations[node.index]
        node_value = self._tree._values[node.index]
        node_results = node_equation[0]*qp[:, 0] \
            + node_equation[1]*qp[:, 1] \
            + node_equation[2]*qp[:, 2] \
            - node_value

        point_spaces = np.sign(node_results)  # Reduces half spaces to -1, 0, 1
        # Possibly want a round on this to deal with floating point errors

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
        # FIXME, want to catch no query points
        if np.nonzero(no_z_distance) != 0:  # No point checking if all distances filled
            z_occluded = self._occludes(self._query_points[query_indices[no_z_distance]],
                                        np.append(node.plane_indices, node.index), 'z')
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
                                        np.append(node.plane_indices, node.index), 'y')
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
                                        np.append(node.plane_indices, node.index), 'x')
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

    def _occludes(self, pt, simplices, axis):
        """
        A function to check whether a set of points are occluded by a simplex
        on a specified axis.
        """
        # FIXME: Make this check for occlusion on an array of simplices
        # We are fine down to line 428 atm

        vertices = self._tree._vertices[self._tree._simplices[simplices]]
        p0 = vertices[:, 0]
        p1 = vertices[:, 1]
        p2 = vertices[:, 2]

        if axis == 'x':
            # p0, p1, p2 are vertices, pt is the array of test points
            area = -p1[:, 1]*p2[:, 2] + p0[:, 1]*(-p1[:, 2] + p2[:, 2]) + p0[:, 2]*(p1[:, 1] - p2[:, 1]) + p1[:, 2]*p2[:, 1]
            if np.any(area == 0):  # This plane is axially aligned
                false_array = np.empty((pt.shape[0]), dtype=np.bool)
                false_array[:] = False
                return false_array
            s1 = np.broadcast_to((p0[:, 1]*p2[:, 2] - p0[:, 2]*p2[:, 1])[:, np.newaxis], (p0.shape[0], pt.shape[0]))
            s2 = np.outer((p2[:, 1] - p0[:, 1]), pt[:, 2])
            s3 = np.outer((p0[:, 2] - p2[:, 2]), pt[:, 1])
            s4 = np.broadcast_to(area[:, np.newaxis], (p0.shape[0], pt.shape[0]))
            s = (s1 + s2 + s3)/s4
            t1 = np.broadcast_to((p0[:, 2]*p1[:, 1] - p0[:, 1]*p1[:, 2])[:, np.newaxis], (p0.shape[0], pt.shape[0]))
            t2 = np.outer((p0[:, 1] - p1[:, 1]), pt[:, 2])
            t3 = np.outer((p1[:, 2] - p0[:, 2]), pt[:, 1])
            t4 = np.broadcast_to(area[:, np.newaxis], (p0.shape[0], pt.shape[0]))
            t = (t1 + t2 + t3)/t4

            return np.any(np.logical_and.reduce((s >= 0, t >= 0, 1-s-t >= 0), axis=0), axis=0)

        if axis == 'y':
            area = -p1[:, 0]*p2[:, 2] + p0[:, 0]*(-p1[:, 2] + p2[:, 2]) + p0[:, 2]*(p1[:, 0] - p2[:, 0]) + p1[:, 2]*p2[:, 0]
            if np.any(area == 0):  # This plane is axially aligned
                false_array = np.empty((pt.shape[0]), dtype=np.bool)
                false_array[:] = False
                return false_array
            s1 = np.broadcast_to((p0[:, 0]*p2[:, 2] - p0[:, 2]*p2[:, 0])[:, np.newaxis], (p0.shape[0], pt.shape[0]))
            s2 = np.outer((p2[:, 0] - p0[:, 0]), pt[:, 2])
            s3 = np.outer((p0[:, 2] - p2[:, 2]), pt[:, 0])
            s4 = np.broadcast_to(area[:, np.newaxis], (p0.shape[0], pt.shape[0]))
            s = (s1 + s2 + s3)/s4
            t1 = np.broadcast_to((p0[:, 2]*p1[:, 0] - p0[:, 0]*p1[:, 2])[:, np.newaxis], (p0.shape[0], pt.shape[0]))
            t2 = np.outer((p0[:, 0] - p1[:, 0]), pt[:, 2])
            t3 = np.outer((p1[:, 2] - p0[:, 2]), pt[:, 0])
            t4 = np.broadcast_to(area[:, np.newaxis], (p0.shape[0], pt.shape[0]))
            t = (t1 + t2 + t3)/t4

            return np.any(np.logical_and.reduce((s >= 0, t >= 0, 1-s-t >= 0), axis=0), axis=0)

        if axis == 'z':
            area = -p1[:, 0]*p2[:, 1] + p0[:, 0]*(-p1[:, 1] + p2[:, 1]) + p0[:, 1]*(p1[:, 0] - p2[:, 0]) + p1[:, 1]*p2[:, 0]

            if np.any(area == 0):  # This plane is axially aligned
                print('Everything has gone wrong, area should not be zero in the z plane')
                print(vertices)
                print(self._tree._equations[simplices])
                print(self._tree._values[simplices])
            # S calculation is split into parts as it is very messy for the array version
            s1 = np.broadcast_to((p0[:, 0]*p2[:, 1] - p0[:, 1]*p2[:, 0])[:, np.newaxis], (p0.shape[0], pt.shape[0]))
            s2 = np.outer((p2[:, 0] - p0[:, 0]), pt[:, 1])
            s3 = np.outer((p0[:, 1] - p2[:, 1]), pt[:, 0])
            s4 = np.broadcast_to(area[:, np.newaxis], (p0.shape[0], pt.shape[0]))
            s = (s1 + s2 + s3)/s4
            t1 = np.broadcast_to((p0[:, 1]*p1[:, 0] - p0[:, 0]*p1[:, 1])[:, np.newaxis], (p0.shape[0], pt.shape[0]))
            t2 = np.outer((p0[:, 0] - p1[:, 0]), pt[:, 1])
            t3 = np.outer((p1[:, 1] - p0[:, 1]), pt[:, 0])
            t4 = np.broadcast_to(area[:, np.newaxis], (p0.shape[0], pt.shape[0]))
            t = (t1 + t2 + t3)/t4

            # These are >= as a point under the edge of a polygon is still occluded
            return np.any(np.logical_and.reduce((s >= 0, t >= 0, 1-s-t >= 0), axis=0), axis=0)

    def _distance(self, pt, simplex, axis):
        """
        Measures the axial distance between points and a simplex along a specified
        axis.
        """
        A, B, C = self._tree._equations[simplex]
        D = self._tree._values[simplex]
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
