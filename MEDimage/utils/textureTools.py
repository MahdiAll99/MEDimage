#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List, Union

import numpy as np


def get_neighbour_direction(d=1.8,
                            distance="euclidian",
                            centre=False,
                            complete=False,
                            dim3=True) -> np.ndarray:
    """Defines transitions to neighbour voxels.

    Note:
        This code was adapted from the in-house radiomics software created at
        OncoRay, Dresden, Germany.

    Args:
        d (float, optional): Max ``distance`` between voxels.
        distance (str, optional): Distance norm used to compute distances. must be
                                  "manhattan", "l1", "l_1", "euclidian", "l2", "l_2", "chebyshev", "linf" or "l_inf".
        centre (bool, optional): Flags whether the [0,0,0] direction should be included
        complete(bool, optional): Flags whether all directions should be computed (True)
                                  or just the primary ones (False). For example, including [0,0,1] and [0,0,-1]
                                  directions may lead to redundant texture matrices.
        dim3(bool, optional): flags whether full 3D (True) or only in-slice (2D; False)
                              directions should be considered.

    Returns:
        ndarray: set of k neighbour direction vectors.
    """

    # Base transition vector
    trans = np.arange(start=-np.ceil(d), stop=np.ceil(d)+1)
    n = np.size(trans)

    # Build transition array [x,y,z]
    nbrs = np.array([rep(x=trans, each=n * n, times=1),
                     rep(x=trans, each=n, times=n),
                     rep(x=trans, each=1, times=n * n)], dtype=np.int32)

    # Initiate maintenance index
    index = np.zeros(np.shape(nbrs)[1], dtype=bool)

    # Remove neighbours more than distance d from the center ----------------

    # Manhattan distance
    if distance.lower() in ["manhattan", "l1", "l_1"]:
        index = np.logical_or(index, np.sum(np.abs(nbrs), axis=0) <= d)
    # Eucldian distance
    if distance.lower() in ["euclidian", "l2", "l_2"]:
        index = np.logical_or(index, np.sqrt(
            np.sum(np.multiply(nbrs, nbrs), axis=0)) <= d)
    # Chebyshev distance
    if distance.lower() in ["chebyshev", "linf", "l_inf"]:
        index = np.logical_or(index, np.max(np.abs(nbrs), axis=0) <= d)

    # Check if centre voxel [0,0,0] should be maintained; False indicates removal
    if centre is False:
        index = np.logical_and(index, (np.sum(np.abs(nbrs), axis=0)) > 0)

    # Check if a complete neighbourhood should be returned
    # False indicates that only half of the vectors are returned
    if complete is False:
        index[np.arange(start=0, stop=len(index)//2 + 1)] = False

    # Check if neighbourhood should be 3D or 2D
    if dim3 is False:
        index[nbrs[2, :] != 0] = False

    return nbrs[:, index]


def rep(x: np.ndarray,
        each=1,
        times=1) -> np.ndarray:
    """Replicates the values in ``x``.
    Replicates the :func:`"rep"` function found in R for tiling and repeating vectors.

    Note:
        Code was adapted from the in-house radiomics software created at OncoRay,
        Dresden, Germany.

    Args:
        x (ndarray): Array to replicate.
        each (int): Integer (non-negative) giving the number of times to repeat
                    each element of the passed array.
        times (int): Integer (non-negative). Each element of ``x`` is repeated each times.

    Returns:
        ndarray: Array with same values but replicated.
    """

    each = int(each)
    times = int(times)

    if each > 1:
        x = np.repeat(x, repeats=each)

    if times > 1:
        x = np.tile(x, reps=times)

    return x

def get_value(x: np.ndarray,
              index: int,
              replace_invalid=True) -> np.ndarray:
    """Retrieves intensity values from an image intensity table used for computing
    texture features.

    Note:
        Code was adapted from the in-house radiomics software created at OncoRay,
        Dresden, Germany.

    Args:
        x (ndarray): set of intensity values.
        index (int): Index to the provided set of intensity values.
        replace_invalid (bool, optional): If True, invalid indices will be replaced
                                          by a placeholder "NaN" value.

    Returns:
        ndarray: Array of the intensity values found at the requested indices.

    """

    # Initialise placeholder
    read_x = np.zeros(np.shape(x))

    # Read variables for valid indices
    read_x[index >= 0] = x[index[index >= 0]]

    if replace_invalid:
        # Set variables for invalid indices to nan
        read_x[index < 0] = np.nan

        # Set variables for invalid initial indices to nan
        read_x[np.isnan(x)] = np.nan

    return read_x


def coord2index(x: np.ndarray,
                y: np.ndarray,
                z: np.ndarray,
                dims: Union[List, np.ndarray]) -> Union[np.ndarray,
                                                        List]:
    """Translate requested coordinates to row indices in image intensity tables.

    Note:
        Code was adapted from the in-house radiomics software created at OncoRay,
        Dresden, Germany.

    Args:
        x (ndarray): set of discrete x-coordinates.
        y (ndarray): set of discrete y-coordinates.
        z (ndarray): set of discrete z-coordinates.
        dims (ndarray or List): dimensions of the image.

    Returns:
        ndarray or List: Array or List of indexes corresponding the requested coordinates

    """

    # Translate coordinates to indices
    index = z + y * dims[2] + x * dims[2] * dims[1]

    # Mark invalid transitions
    index[np.logical_or(x < 0, x >= dims[0])] = -99999
    index[np.logical_or(y < 0, y >= dims[1])] = -99999
    index[np.logical_or(z < 0, z >= dims[2])] = -99999

    return index


def is_list_all_none(x: List) -> bool:
    """Determines if all list elements are None.

    Args:
        x (List): List of elements to check.

    Returns:
        bool: True if all elemets in `x` are None.

    """
    return all(y is None for y in x)
