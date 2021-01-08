import pandas as pd
import numpy as np


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
    fill_val = min(data)  # Filler value for the distance function
    x, y, z = np.where(data != fill_val)

    return x, y, z


def build_dataframe(data, spacing):
    """
    Return a dataframe with columns x, y, z, eta

    Parameters
    ----------
    data : ndarray
        The distance function for a particular axis
    spacing : float
        The spacing of the grid

    Returns
    -------
    points : pandas DataFrame
        The dataframe of points and their respective eta values
    """
    x, y, z = find_boundary_points(data)

    eta = data[x, y, z]

    col_x = pd.Series(x.astype(np.int), name='x')
    col_y = pd.Series(y.astype(np.int), name='y')
    col_z = pd.Series(z.astype(np.int), name='z')

    eta_r = pd.Series(np.where(eta >= 0, eta/spacing, np.NaN), name='eta_r')
    eta_l = pd.Series(np.where(eta <= 0, eta/spacing, np.NaN), name='eta_1')

    frame = {'x': col_x, 'y': col_y, 'z': col_z, 'eta_l': eta_l, 'eta_r': eta_r}
    points = pd.DataFrame(frame)
    return points


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


def get_data_inc_reciprocals(data, spacing, axis):
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

    Returns
    -------
    aggregated_data : pandas DataFrame
        Dataframe of points including reciprocal distances, consolidated down
    """

    df = build_dataframe(data, spacing)
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

    # FIXME: Can copies be removed?
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
