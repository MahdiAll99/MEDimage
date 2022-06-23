#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

from ..biomarkers.get_ngldm_matrix import get_ngldm_matrix
from copy import deepcopy
from ..utils.textureTools import get_neighbour_direction, is_list_all_none, coord2index, get_value


def get_matrix(vol: np.ndarray)-> np.ndarray:
    """Compute neighbouring grey level dependence matrix.

    Args:
        vol(ndarray): 3D volume, isotropically resampled, quantized,
       (e.g. n_g = 32, levels = [1, ..., n_g]) with NaNs
       outside the region of interest

        levels (ndarray or List): Vector containing the quantized gray-levels 
            in the tumor region (or reconstruction levels of quantization).

    Returns:
        ndarray: Array of neighbouring grey level dependence matrix of 'roi_only'.

    """

    vol = vol.copy()

    # GET THE ngldm MATRIX
    # Correct definition, without any assumption
    levels = np.arange(1, np.max(vol[~np.isnan(vol[:])].astype("int"))+1)
    ngldm = get_ngldm_matrix(vol, levels)
    
    return ngldm

def lde(ngldm: np.ndarray)-> float:
    """
    Computes low dependence emphasis feature.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: low depence emphasis value

    """
    n_s = np.sum(ngldm)
    # Normalization of ngldm
    ngldm = ngldm/n_s
    sz = np.shape(ngldm)  # Size of ngldm
    c_vect = range(1, sz[1]+1)  # Row vectors
    pd = np.sum(ngldm, 0)  # Dependence Count Vector

    return (np.matmul(pd, np.transpose(np.power(1.0/np.array(c_vect), 2))))

def hde(ngldm: np.ndarray)-> float:
    """
    Computes high dependence emphasis feature.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: high depence emphasis value

    """
    n_s = np.sum(ngldm)
    # Normalization of ngldm
    ngldm = ngldm/n_s
    sz = np.shape(ngldm)  # Size of ngldm
    c_vect = range(1, sz[1]+1)  # Row vectors
    pd = np.sum(ngldm, 0)  # Dependence Count Vector

    return (np.matmul(pd, np.transpose(np.power(np.array(c_vect), 2))))

def lgce(ngldm: np.ndarray)-> float:
    """
    Computes low grey level count emphasis feature.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: low grey level count emphasis value

    """
    n_s = np.sum(ngldm)
    # Normalization of ngldm
    ngldm = ngldm/n_s
    sz = np.shape(ngldm)  # Size of ngldm
    r_vect = range(1, sz[0]+1)  # Column vectors
    pg = np.transpose(np.sum(ngldm, 1))  # Gray-Level Vector

    return np.matmul(pg, np.transpose(np.power(1.0/np.array(r_vect), 2)))

def hgce(ngldm: np.ndarray)-> float:
    """
    Computes high grey level count emphasis feature.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: high grey level count emphasis value

    """
    n_s = np.sum(ngldm)
    # Normalization of ngldm
    ngldm = ngldm/n_s
    sz = np.shape(ngldm)  # Size of ngldm
    r_vect = range(1, sz[0]+1)  # Column vectors
    pg = np.transpose(np.sum(ngldm, 1))  # Gray-Level Vector

    return np.matmul(pg, np.transpose(np.power(np.array(r_vect), 2)))

def ldlge(ngldm: np.ndarray)-> float:
    """
    Computes low dependence low grey level emphasis feature.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: low dependence low grey level emphasis value

    """
    n_s = np.sum(ngldm)
    # Normalization of ngldm
    ngldm = ngldm/n_s
    sz = np.shape(ngldm)  # Size of ngldm
    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    # Column and row indicators for each entry of the ngldm
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)

    return np.sum(np.sum(ngldm*(np.power(1.0/r_mat, 2))*(np.power(1.0/c_mat, 2))))

def ldhge(ngldm: np.ndarray)-> float:
    """
    Computes low dependence high grey level emphasis feature.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: low dependence high grey level emphasis value

    """
    n_s = np.sum(ngldm)
    # Normalization of ngldm
    ngldm = ngldm/n_s
    sz = np.shape(ngldm)  # Size of ngldm
    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    # Column and row indicators for each entry of the ngldm
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)

    return np.sum(np.sum(ngldm*(np.power(r_mat, 2))*(np.power(1.0/c_mat, 2))))

def hdlge(ngldm: np.ndarray)-> float:
    """
    Computes high dependence low grey level emphasis feature.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: high dependence low grey level emphasis value

    """
    n_s = np.sum(ngldm)
    # Normalization of ngldm
    ngldm = ngldm/n_s
    sz = np.shape(ngldm)  # Size of ngldm
    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    # Column and row indicators for each entry of the ngldm
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)

    return np.sum(np.sum(ngldm*(np.power(1.0/r_mat, 2))*(np.power(c_mat, 2))))

def hdhge(ngldm: np.ndarray)-> float:
    """
    Computes high dependence high grey level emphasis feature.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: high dependence high grey level emphasis value

    """
    n_s = np.sum(ngldm)
    # Normalization of ngldm
    ngldm = ngldm/n_s
    sz = np.shape(ngldm)  # Size of ngldm
    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    # Column and row indicators for each entry of the ngldm
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)

    return np.sum(np.sum(ngldm*(np.power(r_mat, 2))*(np.power(c_mat, 2))))

def glnu(ngldm: np.ndarray)-> float:
    """
    Computes grey level non-uniformity feature.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: grey level non-uniformity value

    """
    n_s = np.sum(ngldm)
    # Normalization of ngldm
    ngldm = ngldm/n_s
    pg = np.transpose(np.sum(ngldm, 1))  # Gray-Level Vector

    return np.sum(np.power(pg, 2)) * n_s

def glnu_norm(ngldm: np.ndarray)-> float:
    """
    Computes grey level non-uniformity normalised feature.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: grey level non-uniformity normalised value

    """
    n_s = np.sum(ngldm)
    # Normalization of ngldm
    ngldm = ngldm/n_s
    pg = np.transpose(np.sum(ngldm, 1))  # Gray-Level Vector

    return np.sum(np.power(pg, 2))

def dcnu(ngldm: np.ndarray)-> float:
    """
    Computes dependence count non-uniformity feature.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: dependence count non-uniformity value

    """
    n_s = np.sum(ngldm)
    # Normalization of ngldm
    ngldm = ngldm/n_s
    pd = np.sum(ngldm, 0)  # Dependence Count Vector

    return np.sum(np.power(pd, 2)) * n_s

def dcnu_norm(ngldm: np.ndarray)-> float:
    """
    Computes dependence count non-uniformity normalised feature.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: dependence count non-uniformity normalised value

    """
    n_s = np.sum(ngldm)
    # Normalization of ngldm
    ngldm = ngldm/n_s
    pd = np.sum(ngldm, 0)  # Dependence Count Vector

    return np.sum(np.power(pd, 2))

def gl_var(ngldm: np.ndarray)-> float:
    """
    Computes grey level variance feature.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: grey level variance value

    """
    n_s = np.sum(ngldm)
    # Normalization of ngldm
    ngldm = ngldm/n_s
    sz = np.shape(ngldm)  # Size of ngldm
    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    # Column and row indicators for each entry of the ngldm
    _, r_mat = np.meshgrid(c_vect, r_vect)
    
    temp = r_mat * ngldm
    u = np.sum(temp)
    temp = (np.power(r_mat - u, 2)) * ngldm
    gl_var = np.sum(temp) 

    return gl_var

def dc_var(ngldm: np.ndarray)-> float:
    """
    Computes dependence count variance feature.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: dependence count variance value

    """
    n_s = np.sum(ngldm)
    # Normalization of ngldm
    ngldm = ngldm/n_s
    sz = np.shape(ngldm)  # Size of ngldm
    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    # Column and row indicators for each entry of the ngldm
    c_mat, _ = np.meshgrid(c_vect, r_vect)
    
    temp = c_mat * ngldm
    u = np.sum(temp)
    temp = (np.power(c_mat - u, 2)) * ngldm
    dc_var = np.sum(temp) 

    return dc_var

def dc_entr(ngldm: np.ndarray)-> float:
    """
    Computes dependence count entropy feature.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: dependence count entropy value

    """
    n_s = np.sum(ngldm)
    # Normalization of ngldm
    ngldm = ngldm/n_s
  
    val_pos = ngldm[np.nonzero(ngldm)]
    temp = val_pos * np.log2(val_pos)
    dc_entr = -np.sum(temp)

    return dc_entr

def dc_energy(ngldm: np.ndarray)-> float:
    """
    Computes dependence count energy feature.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: dependence count energy value

    """
    n_s = np.sum(ngldm)
    # Normalization of ngldm
    ngldm = ngldm/n_s
  
    temp = np.power(ngldm, 2)
    dc_energy = np.sum(temp)

    return dc_energy


def extract_all(vol, method="new"):
    """Compute NGLDMfeatures.
    -------------------------------------------------------------------------
     - vol: 3D volume, isotropically resampled, quantized,
       (e.g. n_g = 32, levels = [1, ..., n_g]) with NaNs
       outside the region of interest
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

    if method == "old":
        ngldm = get_ngldm_features_deprecated(vol=vol)

    elif method == "new":
        ngldm = get_ngldm_features(vol=vol, intensity_range=[np.nan, np.nan])

    else:
        raise ValueError("ngldm should either be calculated using the faster \"new\" method, or the slow \"old\" method.")

    return ngldm


def get_ngldm_features(vol, intensity_range, ngldm_spatial_method="3d", ngldm_diff_lvl=0.0, ngldm_dist=1.0):
    """
    Extract neighbouring grey level dependence matrix-based features from the intensity roi mask.

    :param vol: volume with discretised intensities as 3D numpy array (x, y, z).
    :param intensity_range: range of potential discretised intensities,
     provided as a list: [minimal discretised intensity, maximal discretised intensity]. If one or both values
     are unknown, replace the respective values with np.nan.
    :param ngldm_spatial_method: spatial method which determines the way neighbouring grey level dependence
     matrices are calculated and how features are determined. One of "2d", "2.5d" or "3d".
    :param ngldm_diff_lvl: also called coarseness. Coarseness determines which intensity differences are
     allowed for intensities to be considered similar. Typically 0, and changing discretisation levels may
     have the same effect as increasing the coarseness parameter.
    :param ngldm_dist: the chebyshev distance that forms a local neighbourhood around a center voxel.
    :return: dictionary with feature values.

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

    if type(ngldm_spatial_method) is not list:
        ngldm_spatial_method = [ngldm_spatial_method]

    if type(ngldm_diff_lvl) is not list:
        ngldm_diff_lvl = [ngldm_diff_lvl]

    if type(ngldm_dist) is not list:
        ngldm_dist = [ngldm_dist]

    # Get the roi in tabular format
    img_dims = vol.shape
    index_id = np.arange(start=0, stop=vol.size)
    coords = np.unravel_index(indices=index_id, shape=img_dims)
    df_img = pd.DataFrame({"index_id": index_id,
                           "g": np.ravel(vol),
                           "x": coords[0],
                           "y": coords[1],
                           "z": coords[2],
                           "roi_int_mask": np.ravel(np.isfinite(vol))})

    # Generate an empty feature list
    feat_list = []

    # Iterate over spatial arrangements
    for ii_spatial in ngldm_spatial_method:

        # Iterate over difference levels
        for ii_diff_lvl in ngldm_diff_lvl:

            # Iterate over distances
            for ii_dist in ngldm_dist:

                # Initiate list of ngldm objects
                ngldm_list = []

                # Perform 2D analysis
                if ii_spatial.lower() in ["2d", "2.5d"]:

                    # Iterate over slices
                    for ii_slice in np.arange(0, img_dims[2]):

                        # Add ngldm matrices to list
                        ngldm_list += [GreyLevelDependenceMatrix(distance=np.int(ii_dist), diff_lvl=ii_diff_lvl,
                                                                 spatial_method=ii_spatial.lower(), img_slice=ii_slice)]

                # Perform 3D analysis
                elif ii_spatial.lower() == "3d":

                    # Add ngldm matrices to list
                    ngldm_list += [GreyLevelDependenceMatrix(distance=np.int(ii_dist), diff_lvl=ii_diff_lvl,
                                                             spatial_method=ii_spatial.lower(), img_slice=None)]
                else:
                    raise ValueError("Spatial methods for ngldm should be \"2d\", \"2.5d\" or \"3d\".")

                # Calculate ngldm matrices
                for ngldm in ngldm_list:
                    ngldm.calculate_ngldm_matrix(df_img=df_img, img_dims=img_dims)

                # Merge matrices according to the given method
                upd_list = combine_ngldm_matrices(ngldm_list=ngldm_list, spatial_method=ii_spatial.lower())

                # Calculate features
                feat_run_list = []
                for ngldm in upd_list:
                    feat_run_list += [ngldm.compute_ngldm_features(intensity_range=intensity_range)]

                # Average feature values
                feat_list += [pd.concat(feat_run_list, axis=0).mean(axis=0, skipna=True).to_frame().transpose()]

    # Merge feature tables into a single dictionary
    df_feat = pd.concat(feat_list, axis=1).to_dict(orient="records")[0]

    return df_feat


def combine_ngldm_matrices(ngldm_list, spatial_method):
    """
    Function to merge neighbouring grey level dependence matrices prior to feature calculation.

    :param ngldm_list: list of GreyLevelDependenceMatrix objects.
    :param spatial_method: spatial method which determines the way neighbouring grey level dependence matrices are calculated and how features are determined.
     One of "2d", "2.5d" or "3d".
    :return: list of one or more merged GreyLevelDependenceMatrix objects.

    This code was adapted from the in-house radiomics software created at OncoRay, Dresden, Germany.
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

    # Initiate empty list
    use_list = []

    if spatial_method == "2d":
        # Average features over slice: maintain original ngldms

        # Make copy of ngldm_list
        use_list = []
        for ngldm in ngldm_list:
            use_list += [ngldm.copy()]

    elif spatial_method in ["2.5d", "3d"]:
        # Merge all ngldms into a single representation

        # Select all matrices within the slice
        sel_matrix_list = []
        for ngldm_id in np.arange(len(ngldm_list)):
            sel_matrix_list += [ngldm_list[ngldm_id].matrix]

        # Check if any matrix has been created
        if is_list_all_none(sel_matrix_list):
            # No matrix was created
            use_list += [GreyLevelDependenceMatrix(distance=ngldm_list[0].distance, diff_lvl=ngldm_list[0].diff_lvl,
                                                   spatial_method=spatial_method, img_slice=None, matrix=None, n_v=0.0)]
        else:
            # Merge neighbouring grey level difference matrices
            merge_ngldm = pd.concat(sel_matrix_list, axis=0)
            merge_ngldm = merge_ngldm.groupby(by=["i", "j"]).sum().reset_index()

            # Update the number of voxels
            merge_n_v = 0.0
            for ngldm_id in np.arange(len(ngldm_list)):
                merge_n_v += ngldm_list[ngldm_id].n_v

            # Create new neighbouring grey level difference matrix
            use_list += [GreyLevelDependenceMatrix(distance=ngldm_list[0].distance, diff_lvl=ngldm_list[0].diff_lvl,
                                                   spatial_method=spatial_method, img_slice=None, matrix=merge_ngldm, n_v=merge_n_v)]
    else:
        use_list = None

    # Return to new ngldm list to calling function
    return use_list


class GreyLevelDependenceMatrix:

    def __init__(self, distance, diff_lvl, spatial_method, img_slice=None, matrix=None, n_v=None):
        """
        Initialising function for a new neighbouring grey level dependence matrix.
        :param distance: chebyshev distance used to determine the local neighbourhood.
        :param diff_lvl: coarseness parameter which determines which intensities are considered similar.
        :param spatial_method: spatial method used to calculate the ngldm: 2d, 2.5d or 3d
        :param img_slice: corresponding slice index (only if the ngldm corresponds to a 2d image slice)
        :param matrix: the actual ngldm in sparse format (row, column, count)
        :param n_v: the number of voxels in the volume
        """

        # Distance used
        self.distance = distance
        self.diff_lvl = diff_lvl

        # Slice for which the current matrix is extracted
        self.img_slice = img_slice

        # Spatial analysis method (2d, 2.5d, 3d)
        self.spatial_method = spatial_method

        # Place holders
        self.matrix = matrix
        self.n_v = n_v

    def copy(self):
        """Returns a copy of the GreyLevelDependenceMatrix object."""
        return deepcopy(self)

    def set_empty(self):
        """Creates an empty GreyLevelDependenceMatrix"""
        self.n_v = 0
        self.matrix = None

    def calculate_ngldm_matrix(self, df_img, img_dims):
        """
        Function that calculates an ngldm for the settings provided during initialisation and the input image.

        :param df_img: data table containing image intensities, x, y and z coordinates, and mask labels corresponding to voxels in the volume.
        :param img_dims: dimensions of the image volume
        """

        # Check if the input image and roi exist
        if df_img is None:
            self.set_empty()
            return

        # Check if the roi contains any masked voxels. If this is not the case, don't construct the ngldm.
        if not np.any(df_img.roi_int_mask):
            self.set_empty()
            return

        if self.spatial_method == "3d":
            # Set up neighbour vectors
            nbrs = get_neighbour_direction(d=self.distance, distance="chebyshev", centre=False, complete=True, dim3=True)

            # Set up work copy
            df_ngldm = deepcopy(df_img)
        elif self.spatial_method in ["2d", "2.5d"]:
            # Set up neighbour vectors
            nbrs = get_neighbour_direction(d=self.distance, distance="chebyshev", centre=False, complete=True, dim3=False)

            # Set up work copy
            df_ngldm = deepcopy(df_img[df_img.z == self.img_slice])
            df_ngldm["index_id"] = np.arange(0, len(df_ngldm))
            df_ngldm["z"] = 0
            df_ngldm = df_ngldm.reset_index(drop=True)
        else:
            raise ValueError("The spatial method for neighbouring grey level dependence matrices should be one of \"2d\", \"2.5d\" or \"3d\".")

        # Set grey level of voxels outside ROI to NaN
        df_ngldm.loc[df_ngldm.roi_int_mask == False, "g"] = np.nan

        # Update number of voxels for current iteration
        self.n_v = np.sum(df_ngldm.roi_int_mask.values)

        # Initialise sum of grey levels and number of neighbours
        df_ngldm["occur"] = 0.0
        df_ngldm["n_nbrs"] = 0.0

        for k in range(0, np.shape(nbrs)[1]):
            # Determine potential transitions from valid voxels
            df_ngldm["to_index"] = coord2index(x=df_ngldm.x.values + nbrs[2, k],
                                               y=df_ngldm.y.values + nbrs[1, k],
                                               z=df_ngldm.z.values + nbrs[0, k],
                                               dims=img_dims)

            # Get grey level value from transitions
            df_ngldm["to_g"] = get_value(x=df_ngldm.g.values, index=df_ngldm.to_index.values)

            # Determine which voxels have valid neighbours
            sel_index = np.isfinite(df_ngldm.to_g)

            # Determine co-occurrence within diff_lvl
            df_ngldm.loc[sel_index, "occur"] += ((np.abs(df_ngldm.to_g - df_ngldm.g)[sel_index]) <= self.diff_lvl) * 1

        # Work with voxels within the intensity roi
        df_ngldm = df_ngldm[df_ngldm.roi_int_mask]

        # Drop superfluous columns
        df_ngldm = df_ngldm.drop(labels=["index_id", "x", "y", "z", "to_index", "to_g", "roi_int_mask"], axis=1)

        # Sum s over voxels
        df_ngldm = df_ngldm.groupby(by=["g", "occur"]).size().reset_index(name="n")

        # Rename columns
        df_ngldm.columns = ["i", "j", "s"]

        # Add one to dependency count as features are not defined for k=0
        df_ngldm.j += 1.0

        # Add matrix to object
        self.matrix = df_ngldm

    def compute_ngldm_features(self, intensity_range):
        """
        Computes neighbouring grey level dependence matrix features for the current neighbouring grey level dependence matrix.

        :param intensity_range: range of potential discretised intensities, provided as a list: [minimal discretised intensity, maximal
         discretised intensity]. If one or both values are unknown, replace the respective values with np.nan.
        :return: pandas data frame with values for each feature.
        """

        # Create feature table
        feat_names = ["Fngl_lde", "Fngl_hde", "Fngl_lgce", "Fngl_hgce", "Fngl_ldlge", "Fngl_ldhge", "Fngl_hdlge", "Fngl_hdhge",
                      "Fngl_glnu", "Fngl_glnu_norm", "Fngl_dcnu", "Fngl_dcnu_norm",
                      "Fngl_gl_var", "Fngl_dc_var", "Fngl_dc_entr", "Fngl_dc_energy"]
        df_feat = pd.DataFrame(np.full(shape=(1, len(feat_names)), fill_value=np.nan))
        df_feat.columns = feat_names

        # Don't return data for empty slices or slices without a good matrix
        if self.matrix is None:
            # Update names
            # df_feat.columns += self.parse_feature_names()
            return df_feat
        elif len(self.matrix) == 0:
            # Update names
            # df_feat.columns += self.parse_feature_names()
            return df_feat

        # Dependence count dataframe
        df_sij = deepcopy(self.matrix)
        df_sij.columns = ("i", "j", "sij")

        # Sum over grey levels
        df_si = df_sij.groupby(by="i")["sij"].agg(np.sum).reset_index().rename(columns={"sij": "si"})

        # Sum over dependence counts
        df_sj = df_sij.groupby(by="j")["sij"].agg(np.sum).reset_index().rename(columns={"sij": "sj"})

        # Constant definitions
        n_s = np.sum(df_sij.sij) * 1.0  # Number of neighbourhoods considered
        n_v = self.n_v  # Number of voxels

        ###############################################
        # ngldm features
        ###############################################

        # Low dependence emphasis
        df_feat.loc[0, "Fngl_lde"] = np.sum(df_sj.sj / df_sj.j ** 2.0) / n_s

        # High dependence emphasis
        df_feat.loc[0, "Fngl_hde"] = np.sum(df_sj.sj * df_sj.j ** 2.0) / n_s

        # Grey level non-uniformity
        df_feat.loc[0, "Fngl_glnu"] = np.sum(df_si.si ** 2.0) / n_s

        # Grey level non-uniformity, normalised
        df_feat.loc[0, "Fngl_glnu_norm"] = np.sum(df_si.si ** 2.0) / n_s ** 2.0

        # Dependence count non-uniformity
        df_feat.loc[0, "Fngl_dcnu"] = np.sum(df_sj.sj ** 2.0) / n_s

        # Dependence count non-uniformity, normalised
        df_feat.loc[0, "Fngl_dcnu_norm"] = np.sum(df_sj.sj ** 2.0) / n_s ** 2.0

        # Dependence count percentage
        # df_feat.loc[0, "ngl_dc_perc"] = n_s / n_v

        # Low grey level count emphasis
        df_feat.loc[0, "Fngl_lgce"] = np.sum(df_si.si / df_si.i ** 2.0) / n_s

        # High grey level count emphasis
        df_feat.loc[0, "Fngl_hgce"] = np.sum(df_si.si * df_si.i ** 2.0) / n_s

        # Low dependence low grey level emphasis
        df_feat.loc[0, "Fngl_ldlge"] = np.sum(df_sij.sij / (df_sij.i * df_sij.j) ** 2.0) / n_s

        # Low dependence high grey level emphasis
        df_feat.loc[0, "Fngl_ldhge"] = np.sum(df_sij.sij * df_sij.i ** 2.0 / df_sij.j ** 2.0) / n_s

        # High dependence low grey level emphasis
        df_feat.loc[0, "Fngl_hdlge"] = np.sum(df_sij.sij * df_sij.j ** 2.0 / df_sij.i ** 2.0) / n_s

        # High dependence high grey level emphasis
        df_feat.loc[0, "Fngl_hdhge"] = np.sum(df_sij.sij * df_sij.i ** 2.0 * df_sij.j ** 2.0) / n_s

        # Grey level variance
        mu = np.sum(df_sij.sij * df_sij.i) / n_s
        df_feat.loc[0, "Fngl_gl_var"] = np.sum((df_sij.i - mu) ** 2.0 * df_sij.sij) / n_s
        del mu

        # Dependence count variance
        mu = np.sum(df_sij.sij * df_sij.j) / n_s
        df_feat.loc[0, "Fngl_dc_var"] = np.sum((df_sij.j - mu) ** 2.0 * df_sij.sij) / n_s
        del mu

        # Dependence count entropy
        df_feat.loc[0, "Fngl_dc_entr"] = - np.sum(df_sij.sij * np.log2(df_sij.sij / n_s)) / n_s

        # Dependence count energy
        df_feat.loc[0, "Fngl_dc_energy"] = np.sum(df_sij.sij ** 2.0) / (n_s ** 2.0)

        # Update names
        # df_feat.columns += self.parse_feature_names()

        return df_feat

    def parse_feature_names(self):
        """"
        Adds additional settings-related identifiers to each feature.
        Not used currently, as the use of different settings for the
        neighbouring grey level dependence matrix is not supported.
        """
        parse_str = ""

        # Add distance
        parse_str += "_d" + str(np.round(self.distance, 1))

        # Add difference level
        parse_str += "_a" + str(np.round(self.diff_lvl, 0))

        # Add spatial method
        if self.spatial_method is not None:
            parse_str += "_" + self.spatial_method

        return parse_str


def get_ngldm_features_deprecated(vol):
    """
    Deprecated code. Calculated neighbouring grey level dependence matrix-based features.
    :param vol: Input volume.
    :return: Dictionary of NGTDM features.
    """
    ngldm = {'Fngl_lde': [],
             'Fngl_hde': [],
             'Fngl_lgce': [],
             'Fngl_hgce': [],
             'Fngl_ldlge': [],
             'Fngl_ldhge': [],
             'Fngl_hdlge': [],
             'Fngl_hdhge': [],
             'Fngl_glnu': [],
             'Fngl_glnu_norm': [],
             'Fngl_dcnu': [],
             'Fngl_dcnu_norm': [],
             'Fngl_gl_var': [],
             'Fngl_dc_var': [],
             'Fngl_dc_entr': [],
             'Fngl_dc_energy': []}

    vol = vol.copy()

    # GET THE ngldm MATRIX
    # Correct definition, without any assumption
    levels = np.arange(1, np.max(vol[~np.isnan(vol[:])].astype("int"))+1)
    ngldm = get_ngldm_matrix(vol, levels)
    n_s = np.sum(ngldm)
    # Normalization of ngldm
    ngldm = ngldm/n_s
    sz = np.shape(ngldm)  # Size of ngldm
    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    # Column and row indicators for each entry of the ngldm
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)
    pg = np.transpose(np.sum(ngldm, 1))  # Gray-Level Vector
    pd = np.sum(ngldm, 0)  # Dependence Count Vector

    # COMPUTING TEXTURES

    # Low dependence emphasis
    ngldm['Fngl_lde'] = (np.matmul(pd, np.transpose(np.power(1.0/np.array(c_vect), 2))))

    # High dependence emphasis
    ngldm['Fngl_hde'] = (np.matmul(pd, np.transpose(np.power(np.array(c_vect), 2))))

    # Low grey level count emphasis
    ngldm['Fngl_lgce'] = np.matmul(pg, np.transpose(np.power(1.0/np.array(r_vect), 2)))
    
    # High grey level count emphasis
    ngldm['Fngl_hgce'] = np.matmul(pg, np.transpose(np.power(np.array(r_vect), 2)))
    
    # Low dependence low grey level emphasis
    ngldm['Fngl_ldlge'] = np.sum(np.sum(ngldm*(np.power(1.0/r_mat, 2))*(np.power(1.0/c_mat, 2))))

    # Low dependence high grey level emphasis
    ngldm['Fngl_ldhge'] = np.sum(np.sum(ngldm*(np.power(r_mat, 2))*(np.power(1.0/c_mat, 2))))

    # High dependence low grey levels emphasis
    ngldm['Fngl_hdlge'] = np.sum(np.sum(ngldm*(np.power(1.0/r_mat, 2))*(np.power(c_mat, 2))))

    # High dependence high grey level emphasis
    ngldm['Fngl_hdhge'] = np.sum(np.sum(ngldm*(np.power(r_mat, 2))*(np.power(c_mat, 2))))

    # Gray level non-uniformity
    ngldm['Fngl_glnu'] = np.sum(np.power(pg, 2)) * n_s

    # Gray level non-uniformity normalised
    ngldm['Fngl_glnu_norm'] = np.sum(np.power(pg, 2))

    # Dependence count non-uniformity
    ngldm['Fngl_dcnu'] = np.sum(np.power(pd, 2)) * n_s

    # Dependence count non-uniformity normalised
    ngldm['Fngl_dcnu_norm'] = np.sum(np.power(pd, 2))

    # Dependence count percentage
    # Omitted, always evaluates to 1.

    # Grey level variance
    temp = r_mat * ngldm
    u = np.sum(temp)
    temp = (np.power(r_mat - u, 2)) * ngldm
    ngldm['Fngl_gl_var'] = np.sum(temp)

    # Dependence count variance
    temp = c_mat * ngldm
    u = np.sum(temp)
    temp = (np.power(c_mat - u, 2)) * ngldm
    ngldm['Fngl_dc_var'] = np.sum(temp)

    # Dependence count entropy
    val_pos = ngldm[np.nonzero(ngldm)]
    temp = val_pos * np.log2(val_pos)
    ngldm['Fngl_dc_entr'] = -np.sum(temp)

    # Dependence count energy
    temp = np.power(ngldm, 2)
    ngldm['Fngl_dc_energy'] = np.sum(temp)

    return ngldm
