"""
Functions used in the evaluation of modified stencils for implementing
immersed boundaries.
"""
import os

import pandas as pd
import numpy as np

from devito import Coefficient, Dimension, Function
from devito.logger import warning
from devitoboundary import __file__
from devitoboundary.stencils.stencils import StencilSet
from devitoboundary.stencils.stencil_utils import standard_stencil, get_grid_offset

_feps = np.finfo(np.float32).eps  # Get the eps


def find_boundary_points(data):
    """
    Find points immediately adjacent to boundary and return their locations.

    Parameters
    ----------
    data : ndarray
        The distance function for a particular axis

    Returns
    -------
    x, y, z : ndarray
        The x, y, and z indices of boundary adjacent points (where "boundary
        adjacent" refers to points where the distance function is not at its
        filler values).
    """
    fill_val = np.amin(data)  # Filler value for the distance function
    x, y, z = np.where(data != fill_val)

    return x, y, z


def build_dataframe(data, spacing, grid_offset, eval_offset):
    """
    Return a dataframe with columns x, y, z, eta

    Parameters
    ----------
    data : ndarray
        The distance function for a particular axis
    spacing : float
        The spacing of the grid
    grid_offset : float
        The grid offset for this axis
    eval_offset : float
        The relative offset at which the derivative is evaluated

    Returns
    -------
    points : pandas DataFrame
        The dataframe of points and their respective eta values
    """
    x, y, z = find_boundary_points(data)

    eta = data[x, y, z]

    col_x = pd.Series(x.astype(int), name='x')
    col_y = pd.Series(y.astype(int), name='y')
    col_z = pd.Series(z.astype(int), name='z')

    if abs(eval_offset) < _feps:
        eta_r = pd.Series(np.where(eta >= 0, eta/spacing, np.NaN), name='eta_r')
        eta_l = pd.Series(np.where(eta <= 0, eta/spacing, np.NaN), name='eta_1')

        frame = {'x': col_x, 'y': col_y, 'z': col_z, 'eta_l': eta_l, 'eta_r': eta_r}
        points = pd.DataFrame(frame)
    elif abs(grid_offset) < _feps:
        if np.sign(eval_offset) == 1:
            eta_r = pd.Series(np.where(eta > 0, eta/spacing, np.NaN), name='eta_r')
            eta_l = pd.Series(np.where(eta <= 0, eta/spacing, np.NaN), name='eta_1')

            frame = {'x': col_x, 'y': col_y, 'z': col_z, 'eta_l': eta_l, 'eta_r': eta_r}
            points = pd.DataFrame(frame)

            # Explicitly remove <= -1 (prevents spurious double points)
            points = points[np.logical_or(points.eta_l > -1, np.isnan(points.eta_l))]
        elif np.sign(eval_offset) == -1:
            eta_r = pd.Series(np.where(eta >= 0, eta/spacing, np.NaN), name='eta_r')
            eta_l = pd.Series(np.where(eta < 0, eta/spacing, np.NaN), name='eta_1')

            frame = {'x': col_x, 'y': col_y, 'z': col_z, 'eta_l': eta_l, 'eta_r': eta_r}
            points = pd.DataFrame(frame)
            # Explicitly remove >= 1
            points = points[np.logical_or(points.eta_r < 1, np.isnan(points.eta_r))]
    elif grid_offset >= _feps:
        eta_r = pd.Series(np.where(eta > 0, eta/spacing, np.NaN), name='eta_r')
        eta_l = pd.Series(np.where(eta <= 0, eta/spacing, np.NaN), name='eta_1')

        frame = {'x': col_x, 'y': col_y, 'z': col_z, 'eta_l': eta_l, 'eta_r': eta_r}
        points = pd.DataFrame(frame)

        # Explicitly remove <= -1
        points = points[np.logical_or(points.eta_l > -1, np.isnan(points.eta_l))]
    elif grid_offset <= -_feps:
        eta_r = pd.Series(np.where(eta >= 0, eta/spacing, np.NaN), name='eta_r')
        eta_l = pd.Series(np.where(eta < 0, eta/spacing, np.NaN), name='eta_1')

        frame = {'x': col_x, 'y': col_y, 'z': col_z, 'eta_l': eta_l, 'eta_r': eta_r}
        points = pd.DataFrame(frame)
        # Explicitly remove >= 1
        points = points[np.logical_or(points.eta_r < 1, np.isnan(points.eta_r))]

    return points


def apply_grid_offset(df, axis, grid_offset, eval_offset):
    """
    Shift eta values according to grid offset.
    """
    df.eta_l -= grid_offset
    df.eta_r -= grid_offset

    # Want to check the grid offset and evaluation offset
    # If both are non-zero then skip this next bit
    if abs(grid_offset) < _feps or abs(eval_offset) < _feps:
        if np.sign(grid_offset) == 1:
            eta_r_mask = df.eta_r < 0
            eta_l_mask = df.eta_l <= -1

            df.loc[eta_r_mask, 'eta_l'] = df.eta_r[eta_r_mask]
            df.loc[eta_r_mask, 'eta_r'] = np.NaN

            df.loc[eta_l_mask, 'eta_l'] += 1

            df.loc[eta_l_mask, axis] -= 1

        elif np.sign(grid_offset) == -1:
            eta_r_mask = df.eta_r > 1
            eta_l_mask = df.eta_l >= 0

            df.loc[eta_l_mask, 'eta_r'] = df.eta_l[eta_l_mask]
            df.loc[eta_l_mask, 'eta_l'] = np.NaN

            df.loc[eta_r_mask, 'eta_r'] -= 1

            df.loc[eta_r_mask, axis] += 1

        # Aggregate and reset the index to undo the grouping
        df = df.groupby(['z', 'y', 'x']).agg({'eta_l': 'max', 'eta_r': 'min'}).reset_index()

    # Make sure zero distances appear on both sides
    # Only wants to be done for non-staggered systems of equations
    # No evaluation offset
    if abs(eval_offset) < _feps:
        l_zero_mask = df.eta_l == 0
        r_zero_mask = df.eta_r == 0
        df.loc[l_zero_mask, 'eta_r'] = 0
        df.loc[r_zero_mask, 'eta_l'] = 0

    return df


def calculate_reciprocals(df, axis, side):
    """
    Calculate reciprocal distances from known values.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe of points for which reciprocals should be calculated
    axis : str
        The axis along which reciprocals should be calculated. Should be 'x',
        'y', or 'z'
    side : str
        The side from which reciprocals should be calculated

    Returns
    -------
    reciprocals : pandas DataFrame
        The dataframe of reciprocal points and their eta values
    """
    increment = -1 if side == 'l' else 1
    other = 'eta_r' if side == 'l' else 'eta_l'  # Other side
    column = 'eta_' + side

    reciprocals = df[pd.notna(df[column])].copy()
    reciprocals[axis] += increment
    reciprocals[other] = reciprocals[column] - increment
    reciprocals[column] = np.NaN

    return reciprocals


def get_data_inc_reciprocals(data, spacing, axis, grid_offset, eval_offset):
    """
    Calculate and consolidate reciprocal values, returning resultant dataframe.

    Parameters
    ----------
    data : ndarray
        The axial distance function data for the specified axis
    spacing : float
        The grid spacing for the specified axis
    axis : str
        The specified axis
    grid_offset : float
        The grid offset for this axis
    eval_offset : float
        The relative offset at which the derivative is evaluated

    Returns
    -------
    aggregated_data : pandas DataFrame
        Dataframe of points including reciprocal distances, consolidated down
    """

    df = build_dataframe(data, spacing, grid_offset, eval_offset)

    df = apply_grid_offset(df, axis, grid_offset, eval_offset)

    reciprocals_l = calculate_reciprocals(df, axis, 'l')
    reciprocals_r = calculate_reciprocals(df, axis, 'r')

    full_df = df.append([reciprocals_l, reciprocals_r])

    # Group and aggregate to consolidate points doubled up by this process
    aggregated_data = full_df.groupby(['z', 'y', 'x']).agg({'eta_l': 'min', 'eta_r': 'min'})

    return aggregated_data


def add_distance_column(data):
    """Adds a distance column to the dataframe and initialises with zeroes"""
    data['dist'] = 0


def split_types(data, axis, axis_size):
    """
    Splits points into the five categories:

    first        o->|
    last         |<-o
    double       |<-o->|
    paired_left  |<-o--------->|
    paired_right |<---------o->|

    Where o marks point location and | is the boundary

    Parameters
    ----------
    data : pandas DataFrame
        The dataframe containing the points to be split
    axis : str
        The axis for which points should be split
    axis_size : int
        The length of the axis (in terms of grid points)

    Returns
    -------
    first, last, double, paired_left, paired_right : pandas DataFrame
        Dataframes for each of the categories
    """
    levels = ['z', 'y', 'x']
    levels.remove(axis)

    first = data.groupby(level=levels).head(1).copy()  # Get head
    last = data.groupby(level=levels).tail(1).copy()  # Get tail

    double_cond = np.logical_and(pd.notna(data.eta_l),
                                 pd.notna(data.eta_r))
    double = data[double_cond]  # Get double-sided

    f_index = first.index  # Get the indices of the easy categories
    l_index = last.index
    d_index = double.index

    paired = data.drop(f_index).drop(l_index).drop(d_index)  # Remove these bits
    paired_left = paired[pd.notna(paired.eta_l)].copy()
    paired_right = paired[pd.notna(paired.eta_r)].copy()

    # Paired points in paired_left and paired_right match up
    paired_dist = (paired_right.index.get_level_values(axis).to_numpy()
                   - paired_left.index.get_level_values(axis).to_numpy())
    paired_left.dist = paired_dist
    paired_right.dist = -paired_dist

    paired_left.eta_r = paired_right.eta_r.to_numpy()  # Ugly, since in most cases these wouldn't fit together
    paired_right.eta_l = paired_left.eta_l.to_numpy()

    first.dist = -f_index.get_level_values(axis).to_numpy()
    last.dist = axis_size - 1 - l_index.get_level_values(axis).to_numpy()

    # Drop any values outside computational domain (generated by reciprocity)
    # Needs to be done after the drop or invalid values end up in the paired categories
    first = first[first.index.get_level_values(axis) >= 0]
    last = last[last.index.get_level_values(axis) < axis_size]

    return first, last, double, paired_left, paired_right


def drop_outside_points(df, segment):
    """
    Drop points where the grid node is outside the domain.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe of points
    segment : ndarray
        The interior-exterior segmentation of the domain
    """
    x_vals = df.index.get_level_values('x').to_numpy()
    y_vals = df.index.get_level_values('y').to_numpy()
    z_vals = df.index.get_level_values('z').to_numpy()

    ext_int = segment[x_vals, y_vals, z_vals] == 1
    return df[ext_int]


def shift_grid_endpoint(df, axis, grid_offset, eval_offset, paired_point=None):
    """
    If the last point within the domain in the direction opposite to staggering
    is not a grid node, then an extra grid node needs to be included on this side.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe of points
    axis : str
        The axis along which to increment. Can be 'x', 'y', or 'z'
    grid_offset : float
        The grid offset for this axis
    eval_offset : float
        The relative offset at which the derivative is evaluated
    """
    df = df.copy()  # Stop inplace modification
    # I think this is not strictly the best way to do this, but is definitely
    # more simple than the alternative
    # FIXME: I think this might be able to produce points outside the domain
    x_ind = df.index.get_level_values('x').to_numpy()
    y_ind = df.index.get_level_values('y').to_numpy()
    z_ind = df.index.get_level_values('z').to_numpy()

    if abs(grid_offset) < _feps:
        if np.sign(eval_offset) == 1:
            # Make a mask for points where shift is necessary
            mask = np.logical_and(df.eta_l + 0.5 < _feps, df.dist >= 0)
            mask = mask.to_numpy()

            if axis == 'x':
                x_ind[mask] -= 1
            elif axis == 'y':
                y_ind[mask] -= 1
            elif axis == 'z':
                z_ind[mask] -= 1

            # Increment eta_l, distance
            df.loc[mask, 'eta_l'] += 1
            df.loc[mask, 'dist'] += 1

        elif np.sign(eval_offset) == -1:
            # Make a mask for points where shift is necessary
            mask = np.logical_and(df.eta_r - 0.5 > -_feps, df.dist <= 0)
            mask = mask.to_numpy()

            if axis == 'x':
                x_ind[mask] += 1
            elif axis == 'y':
                y_ind[mask] += 1
            elif axis == 'z':
                z_ind[mask] += 1

            # Increment eta_r, distance
            df.loc[mask, 'eta_r'] -= 1
            df.loc[mask, 'dist'] -= 1

    else:  # Non-zero grid offset
        if np.sign(grid_offset) == -1:
            # Make a mask for points where shift is necessary
            mask = np.logical_and(df.eta_r - 1 > _feps, df.dist <= 0)  # Less forgiving
            mask = mask.to_numpy()

            if axis == 'x':
                x_ind[mask] += 1
            elif axis == 'y':
                y_ind[mask] += 1
            elif axis == 'z':
                z_ind[mask] += 1

            # Increment eta_r, distance
            df.loc[mask, 'eta_r'] -= 1
            df.loc[mask, 'dist'] -= 1

        elif np.sign(grid_offset) == 1:
            # Make a mask for points where shift is necessary
            mask = np.logical_and(df.eta_l + 1 < -_feps, df.dist >= 0)  # Less forgiving
            mask = mask.to_numpy()

            if axis == 'x':
                x_ind[mask] -= 1
            elif axis == 'y':
                y_ind[mask] -= 1
            elif axis == 'z':
                z_ind[mask] -= 1

            # Increment eta_l, distance
            df.loc[mask, 'eta_l'] += 1
            df.loc[mask, 'dist'] += 1

    # Add the new incremented indices
    df['x'] = x_ind
    df['y'] = y_ind
    df['z'] = z_ind

    df = df.set_index(['z', 'y', 'x'])
    return df


def apply_dist(df, point_type):
    """
    Add distances to eta_l and eta_r for paired cases. (Also applied to double
    cases).
    """
    if point_type == 'paired_left':
        df.eta_r += df.dist
    elif point_type == 'paired_right':
        df.eta_l += df.dist
    elif point_type == 'double':
        # If dist is positive, add to eta_r
        eta_r_mask = df.dist > 0
        df.loc[eta_r_mask, 'eta_r'] += df.loc[eta_r_mask, 'dist']
        # Else if dist is negative, subtract from eta_l
        eta_l_mask = df.dist < 0
        df.loc[eta_l_mask, 'eta_l'] += df.loc[eta_l_mask, 'dist']

    return df


def get_n_pts(df, point_type, space_order, eval_offset):
    """
    Get the number of points associated with each boundary-adjacent point.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe of points
    point_type : str
        The category which the set of points fall into. Can be 'first', 'last',
        'double', 'paired_left', or 'paired_right'.
    space_order : int
        The order of the function which stencils are being generated for
    eval_offset : float
        The relative offset at which the derivative should be evaluated.
    """
    if point_type == 'first':
        if abs(eval_offset) >= _feps:
            # Need to increase max number of points to use if eta sign is wrong
            modifier_eta_r = np.where(df.eta_r.to_numpy() < _feps, 1, 0)
        else:
            modifier_eta_r = 0

        n_pts = np.minimum(int(space_order/2)+modifier_eta_r, 1-df.dist.to_numpy())

    elif point_type == 'last':
        if abs(eval_offset) >= _feps:
            # Need to increase max number of points to use if eta sign is wrong
            modifier_eta_l = np.where(df.eta_l.to_numpy() > -_feps, 1, 0)
        else:
            modifier_eta_l = 0

        n_pts = np.minimum(int(space_order/2)+modifier_eta_l, 1+df.dist.to_numpy())

    elif point_type == 'double':
        # Needs to consider dist, as in staggered cases, may have a second point
        n_pts = np.absolute(df.dist) + 1

    elif point_type == 'paired_left':
        if abs(eval_offset) >= _feps:
            # Increase number of points based on eta_l
            modifier_eta_l = np.where(df.eta_l.to_numpy() - eval_offset > -_feps, 1, 0)
        else:
            modifier_eta_l = 0

        n_pts = np.minimum(int(space_order/2)+modifier_eta_l, df.dist.to_numpy())

    elif point_type == 'paired_right':
        if abs(eval_offset) >= _feps:
            # Increase number of points based on eta_r, but cap affected by eta_l
            modifier_eta_l = np.where(df.eta_l.to_numpy() - eval_offset > -_feps, 1, 0)
            modifier_eta_r = np.where(df.eta_r.to_numpy() - eval_offset < _feps, 1, 0)
        else:
            modifier_eta_l = 0
            modifier_eta_r = 0

        n_pts = np.minimum(int(space_order/2)+modifier_eta_r,
                           1-df.dist.to_numpy()-np.minimum(int(space_order/2)+modifier_eta_l,
                                                           -df.dist.to_numpy()))

    df['n_pts'] = n_pts

    return df


def get_next_point(df, inc, axis):
    """
    Increment to get the next points along.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe of points
    inc : int
        The amount to increment
    axis : str
        The axis along which to increment. Can be 'x', 'y', or 'z'
    """
    df = df.copy()
    # Need to increment the axis
    x_ind = df.index.get_level_values('x').to_numpy()
    y_ind = df.index.get_level_values('y').to_numpy()
    z_ind = df.index.get_level_values('z').to_numpy()

    if axis == 'x':
        x_ind += inc
    elif axis == 'y':
        y_ind += inc
    elif axis == 'z':
        z_ind += inc

    # Add the new incremented indices
    df['x'] = x_ind
    df['y'] = y_ind
    df['z'] = z_ind

    df = df.set_index(['z', 'y', 'x'])
    # Need to increment eta_l
    df.eta_l -= inc
    # Need to increment eta_r
    df.eta_r -= inc

    return df


def get_master_df(msk_pts, point_type, pts):
    """
    Create a dataframe containing all the points for a given points
    count.

    Parameters
    ----------
    msk_pts : pandas DataFrame
         A dataframe of boundary-adjacent points which have the same number of
         modified stencils associated with them.
    point_type : str
        The category which the set of points fall into. Can be 'first', 'last',
        'double', 'paired_left', or 'paired_right'.
    pts : int
        The number of modified stencils associated with each boundary-adjacent
        point.
    """
    # Make a big master dataframe for this number of points
    if point_type == 'last' or point_type == 'paired_left':
        frames = [get_next_point(msk_pts, inc, 'x') for inc in range(pts)]
        master_df = pd.concat(frames)
    elif point_type == 'first' or point_type == 'paired_right':
        frames = [get_next_point(msk_pts, -inc, 'x') for inc in range(pts)]
        master_df = pd.concat(frames)
    elif point_type == 'double':
        # Wants to behave as a special kind of 'paired'
        inc_dir = np.sign(msk_pts.dist.to_numpy())
        frames = [get_next_point(msk_pts, inc_dir*inc, 'x') for inc in range(pts)]
        master_df = pd.concat(frames)

        # Drop points exactly on the boundary
        drop_mask = np.logical_and(np.abs(master_df.eta_l) > _feps,
                                   np.abs(master_df.eta_r) > _feps)
        master_df = master_df[drop_mask]

    # Drop out of bounds points
    return master_df


def get_key_mask(key, df, max_ext_points):
    """
    Get the mask for a given key.

    Parameters
    ----------
    key : tuple of float
        The key consisting of the inner bounds at which the stencil is valid
    df : pandas DataFrame
        The dataframe of points
    max_ext_points : int
        The maximum number of points required by the extrapolation
    """
    # Unpack key
    eta_l_in, eta_r_in = key
    eta_l_out = eta_l_in - 0.5
    eta_r_out = eta_r_in + 0.5

    # Make a mask for this key
    if np.isnan(eta_l_in):
        l_msk = np.logical_or(np.isnan(df.eta_l), df.eta_l < -max_ext_points - _feps)
    else:
        l_msk = np.logical_and(df.eta_l > eta_l_out - _feps,
                               df.eta_l < eta_l_in - _feps)

    if np.isnan(eta_r_in):
        r_msk = np.logical_or(np.isnan(df.eta_r), df.eta_r >= max_ext_points + _feps)
    else:
        r_msk = np.logical_and(df.eta_r > eta_r_in + _feps,
                               df.eta_r < eta_r_out + _feps)
    key_msk = np.logical_and(l_msk, r_msk)

    return key_msk


def eval_stencils(df, sten_lambda, max_ext_points):
    """
    Evaluate the stencils for this particular stencil variant
    and associated points.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe of points
    sten_lambda : dict
        The dictionary containing the function to evaluate the coefficient for
        each point.
    max_ext_points : int
        The maximum number of points required by the extrapolation
    """
    # Make a set of empty stencils to fill
    stencils = np.zeros((len(df), 1+2*max_ext_points))

    # Loop over stencil indices
    for index in sten_lambda:
        func = sten_lambda[index]
        stencils[:, index+max_ext_points] = func(df.eta_l, df.eta_r)

    return stencils


def fill_weights(df, stencils, weights, dim_limit):
    """
    Fill the weight function with stencil coefficients.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe of points
    stencils : ndarray
        The stencils corresponding with each of the points in the DataFrame
    weights : Function
        The Function containing the finite difference coefficients
    """
    # Get the point indices
    x_ind = df.index.get_level_values('x').to_numpy()
    y_ind = df.index.get_level_values('y').to_numpy()
    z_ind = df.index.get_level_values('z').to_numpy()

    valid_x = np.logical_and(x_ind >= 0, x_ind < dim_limit)
    valid_y = np.logical_and(x_ind >= 0, y_ind < dim_limit)
    valid_z = np.logical_and(x_ind >= 0, z_ind < dim_limit)

    valid = np.logical_and(valid_x, np.logical_and(valid_y, valid_z))

    x_ind = x_ind[valid]
    y_ind = y_ind[valid]
    z_ind = z_ind[valid]

    # Fill the weights
    weights.data[x_ind, y_ind, z_ind] = stencils[valid]


def fill_stencils(df, point_type, max_ext_points, lambdas, weights, dim_limit):
    """
    Fill the stencil weights using the identified points.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe of points
    point_type : str
        The category which the set of points fall into. Can be 'first', 'last',
        'double', 'paired_left', or 'paired_right'.
    max_ext_points : int
        The maximum number of points required by the extrapolation
    lambdas : dict
        The functions for stencils to be evaluated
    weights : Function
        The Function containing the finite difference coefficients
    """
    for pts in range(df.n_pts.min(), df.n_pts.max()+1):
        msk_pts = df[df.n_pts == pts]

        # Get all the points
        master_df = get_master_df(msk_pts, point_type, pts)

        # Now loop over keys
        for key in lambdas:
            # Make a mask for this key
            key_msk = get_key_mask(key, master_df, max_ext_points)

            # Now evaluate the stencils and drop them into place
            sten_lambda = lambdas[key]

            msk_pts = master_df[key_msk]

            mod_stencils = eval_stencils(msk_pts, sten_lambda, max_ext_points)

            fill_weights(msk_pts, mod_stencils, weights, dim_limit)


def get_component_weights(data, axis, function, deriv, lambdas, interior,
                          max_ext_points, eval_offset):
    """
    Take a component of the distance field and return the associated weight
    function.

    Parameters
    ----------
    data : ndarray
        The field of the axial distance function for the specified axis
    axis : int
        The axis along which the stencils are orientated. Can be 0, 1, or 2
    function : devito Function
        The function for which stencils should be calculated
    deriv : int
        The order of the derivative to which the stencils pertain
    lambdas : dict
        The functions for stencils to be evaluated
    interior : ndarray
        The interior-exterior segmentation of the domain
    max_ext_points : int
        The maximum number of points required by the extrapolation
    eval_offset : float
        The relative offset at which the derivative should be evaluated.
        Used for setting the default fill stencil.

    Returns
    -------
    w : devito Function
        Function containing the stencil coefficients
    """
    grid_offset = get_grid_offset(function, axis)

    f_grid = function.grid

    dim_limit = f_grid.shape[axis]
    axis_dim = 'x' if axis == 0 else 'y' if axis == 1 else 'z'

    full_data = get_data_inc_reciprocals(data, f_grid.spacing[axis], axis_dim,
                                         grid_offset, eval_offset)

    add_distance_column(full_data)

    first, last, double, paired_left, paired_right \
        = split_types(full_data, axis_dim, f_grid.shape[axis])

    # Need to drop exterior points and shift grid endpoint
    first = drop_outside_points(first, interior)
    last = drop_outside_points(last, interior)
    double = drop_outside_points(double, interior)
    paired_left = drop_outside_points(paired_left, interior)
    paired_right = drop_outside_points(paired_right, interior)

    first = shift_grid_endpoint(first, axis_dim, grid_offset, eval_offset)
    last = shift_grid_endpoint(last, axis_dim, grid_offset, eval_offset)
    double = shift_grid_endpoint(double, axis_dim, grid_offset, eval_offset)
    paired_left = shift_grid_endpoint(paired_left, axis_dim, grid_offset, eval_offset)
    paired_right = shift_grid_endpoint(paired_right, axis_dim, grid_offset, eval_offset)

    double = apply_dist(double, 'double')
    paired_left = apply_dist(paired_left, 'paired_left')
    paired_right = apply_dist(paired_right, 'paired_right')

    # Additional dimension for storing weights
    # This will be dependent on the number of extrapolation points required
    # and the space order.
    dim_size = max(function.space_order//2, max_ext_points)
    s_dim = Dimension(name='s'+str(2*dim_size))
    ncoeffs = 2*dim_size + 1

    w_shape = f_grid.shape + (ncoeffs,)
    w_dims = f_grid.dimensions + (s_dim,)

    w = Function(name='w_'+function.name+'_'+axis_dim, dimensions=w_dims, shape=w_shape)

    # Do the initial stencil fill, padding where needs be
    if max_ext_points > function.space_order//2:
        # Need to zero pad the standard stencil
        zero_pad = max_ext_points - function.space_order//2
        w.data[:, :, :, zero_pad:-zero_pad] = standard_stencil(deriv,
                                                               function.space_order,
                                                               offset=eval_offset)
        # Needs to return a warning if padding is used for the time being
        # Will need a dummy function to create the substitutions, with a higher
        # order function to substitute into
        warning("Generated stencils have been padded due to required number of"
                " extrapolation points. A dummy function will be needed to"
                " create the substitutions. The required order for substitution"
                " is {}".format(2*max_ext_points))
    else:
        w.data[:] = standard_stencil(deriv, function.space_order,
                                     offset=eval_offset)
    w.data[interior == -1] = 0

    # Fill the stencils
    if len(first.index) != 0:
        first = get_n_pts(first, 'first', function.space_order, eval_offset)
        fill_stencils(first, 'first', max_ext_points, lambdas, w, dim_limit)

    if len(last.index) != 0:
        last = get_n_pts(last, 'last', function.space_order, eval_offset)
        fill_stencils(last, 'last', max_ext_points, lambdas, w, dim_limit)

    if len(double.index) != 0:
        double = get_n_pts(double, 'double', function.space_order, eval_offset)
        fill_stencils(double, 'double', max_ext_points, lambdas, w, dim_limit)

    if len(paired_left.index) != 0:
        paired_left = get_n_pts(paired_left, 'paired_left', function.space_order, eval_offset)
        fill_stencils(paired_left, 'paired_left', max_ext_points, lambdas, w, dim_limit)

    if len(paired_right.index) != 0:
        paired_right = get_n_pts(paired_right, 'paired_right', function.space_order, eval_offset)
        fill_stencils(paired_right, 'paired_right', max_ext_points, lambdas, w, dim_limit)

    w.data[:] /= f_grid.spacing[axis]**deriv  # Divide everything through by spacing

    return w


def get_weights(data, function, deriv, bcs, interior, fill_function=None,
                eval_offsets=(0., 0., 0.)):
    """
    Get the modified stencil weights for a function and derivative given the
    axial distances.

    Parameters
    ----------
    data : devito VectorFunction
        The axial distance function
    function : devito Function
        The function for which stencils should be calculated
    deriv : int
        The order of the derivative to which the stencils pertain
    bcs : list of devito Eq
        The boundary conditions which should hold at the surface
    interior : ndarray
        The interior-exterior segmentation of the domain
    fill_function : devito Function
        A secondary function to use when creating the Coefficient objects. If none
        then the Function supplied with the function argument will be used instead.
    eval_offsets : tuple of float
        The relative offsets at which derivatives should be evaluated for each
        axis.

    Returns
    -------
    substitutions : tuple of Coefficient
        The substitutions to be included in the devito equation
    """
    cache = os.path.dirname(__file__) + '/extrapolation_cache.dat'

    # This wants to start as an empty list
    weights = []
    for axis in range(3):
        # Check any != filler value in data[axis].data
        # TODO: Could just calculate this rather than finding the minimum
        fill_val = np.amin(data[axis].data)
        if np.any(data[axis].data != fill_val):
            print(deriv, function, function.grid.dimensions[axis])
            # If True, then behave as normal
            # If False then pass
            stencils = StencilSet(deriv, eval_offsets[axis], bcs, cache=cache)
            lambdas = stencils.lambdaify
            max_ext_points = stencils.max_ext_points

            axis_weights = get_component_weights(data[axis].data, axis, function,
                                                 deriv, lambdas, interior,
                                                 max_ext_points, eval_offsets[axis])

            if fill_function is None:
                weights.append(Coefficient(deriv, function,
                                           function.grid.dimensions[axis],
                                           axis_weights))
            else:
                weights.append(Coefficient(deriv, fill_function,
                                           fill_function.grid.dimensions[axis],
                                           axis_weights))
        else:
            pass  # No boundary-adjacent points so don't return any subs
    # Raise error if list is empty
    if len(weights) == 0:
        raise ValueError("No boundary-adjacent points in provided fields")
    return tuple(weights)
