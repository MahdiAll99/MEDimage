#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def get_neighbour_direction(d=1.8, distance="euclidian", centre=False,
                            complete=False, dim3=True):
    """
    Defines transitions to neighbour voxels.

    :param d: max distance between voxels.
    :param distance: distance norm used to compute distances.
    :param centre: flags whether the [0,0,0] direction should be included
    :param complete: flags whether all directions should be computed (True)
    or just the primary ones (False). For example, including [0,0,1] and
    [0,0,-1] directions may lead to
    redundant texture matrices.
    :param dim3: flags whether full 3D (True) or only in-slice (2D; False)
     directions should be considered.
    :return: set of k neighbour direction vectors

    This code was adapted from the in-house radiomics software created at
    OncoRay, Dresden, Germany.
    -------------------------------------------------------------------------
    AUTHOR(S): MEDomicsLab consortium
    -------------------------------------------------------------------------
    STATEMENT:
    This file is part of <https://github.com/MEDomics/MEDomicsLab/>,
    a package providing MATLAB programming tools for radiomics analysis.
     --> Copyright (C) MEDomicsLab consortium.

    This package is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This package is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this package.  If not, see <http://www.gnu.org/licenses/>.
    -------------------------------------------------------------------------
    """

    # Base transition vector
    trans = np.arange(start=-np.ceil(d), stop=np.ceil(d) + 1)
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


def rep(x, each=1, times=1):
    """"
    This function replicates the "rep" function found in R for tiling and
    repeating vectors.

    Code was adapted from the in-house radiomics software created at OncoRay,
    Dresden, Germany.
    -------------------------------------------------------------------------
    AUTHOR(S): MEDomicsLab consortium
    -------------------------------------------------------------------------
    STATEMENT:
    This file is part of <https://github.com/MEDomics/MEDomicsLab/>,
    a package providing MATLAB programming tools for radiomics analysis.
     --> Copyright (C) MEDomicsLab consortium.

    This package is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This package is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this package.  If not, see <http://www.gnu.org/licenses/>.
    -------------------------------------------------------------------------
    """

    each = int(each)
    times = int(times)

    if each > 1:
        x = np.repeat(x, repeats=each)

    if times > 1:
        x = np.tile(x, reps=times)

    return x


def get_value(x, index, replace_invalid=True):
    """
    Retrieve intensity values from an image intensity table used for computing
    texture features

    :param x: set of intensity values.
    :param index: index to the provided set of intensity values.
    :param replace_invalid: whether entries corresponding invalid indices
     should be replaced by a placeholder "NaN" value.
    :return: intensity values found at the requested indices.

    Code was adapted from the in-house radiomics software created at OncoRay,
    Dresden, Germany.
    -------------------------------------------------------------------------
    AUTHOR(S): MEDomicsLab consortium
    -------------------------------------------------------------------------
    STATEMENT:
    This file is part of <https://github.com/MEDomics/MEDomicsLab/>,
    a package providing MATLAB programming tools for radiomics analysis.
     --> Copyright (C) MEDomicsLab consortium.

    This package is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This package is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this package.  If not, see <http://www.gnu.org/licenses/>.
    -------------------------------------------------------------------------
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


def coord2index(x, y, z, dims):
    """
    Translate requested coordinates to row indices in image intensity tables.

    :param x: set of discrete x coordinates
    :param y: set of discrete y coordinates
    :param z: set of discrete z coordinates
    :param dims: dimensions of the image
    :return: index corresponding the requested coordinates

    Typical use involves finding the index to neighbouring voxels.
    A check for invalid transitions is therefore performed.

    Code was adapted from the in-house radiomics software created at OncoRay,
    Dresden, Germany.
    -------------------------------------------------------------------------
    AUTHOR(S): MEDomicsLab consortium
    -------------------------------------------------------------------------
    STATEMENT:
    This file is part of <https://github.com/MEDomics/MEDomicsLab/>,
    a package providing MATLAB programming tools for radiomics analysis.
     --> Copyright (C) MEDomicsLab consortium.

    This package is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This package is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this package.  If not, see <http://www.gnu.org/licenses/>.
    -------------------------------------------------------------------------
    """

    # Translate coordinates to indices
    index = z + y * dims[2] + x * dims[2] * dims[1]

    # Mark invalid transitions
    index[np.logical_or(x < 0, x >= dims[0])] = -99999
    index[np.logical_or(y < 0, y >= dims[1])] = -99999
    index[np.logical_or(z < 0, z >= dims[2])] = -99999

    return index


def is_list_all_none(x):
    """Determines if all list elements are None"""
    return all(y is None for y in x)
