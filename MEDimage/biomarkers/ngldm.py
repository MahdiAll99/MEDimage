#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
from typing import Dict, List

import numpy as np
import pandas as pd

from ..utils.textureTools import (coord2index, get_neighbour_direction,
                                  get_value, is_list_all_none)


def get_matrix(roi_only: np.array,
                     levels: np.ndarray) -> float:
    """Computes Neighbouring grey level dependence matrix.
    This matrix refers to "Neighbouring grey level dependence based features" (ID = REK0)  
    in the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        roi_only_int (ndarray): Smallest box containing the ROI, with the imaging data ready
            for texture analysis computations. Voxels outside the ROI are
            set to NaNs.
        levels (ndarray or List): Vector containing the quantized gray-levels
            in the tumor region (or reconstruction ``levels`` of quantization).

    Returns:
        ndarray: Array of neighbouring grey level dependence matrix of ``roi_only``.

    """
    roi_only = roi_only.copy()

    # PRELIMINARY
    level_temp = np.max(levels)+1
    roi_only[np.isnan(roi_only)] = level_temp
    levels = np.append(levels, level_temp)
    dim = np.shape(roi_only)
    if np.size(dim) == 2:
        np.append(dim, 1)

    q_2 = np.reshape(roi_only, np.prod(dim), order='F').astype("int")

    # QUANTIZATION EFFECTS CORRECTION (M. Vallieres)
    # In case (for example) we initially wanted to have 64 levels, but due to
    # quantization, only 60 resulted.
    # q_s = round(levels*adjust)/adjust;
    # q_2 = round(q_2*adjust)/adjust;
    q_s = levels.copy()

    # EL NAQA CODE
    q_3 = q_2*0
    lqs = np.size(q_s)
    for k in range(1, lqs+1):
        q_3[q_2 == q_s[k-1]] = k

    q_3 = np.reshape(q_3, dim, order='F')

    # Min dependence = 0, Max dependence = 26; So 27 columns
    ngldm = np.zeros((lqs, 27))
    for i in range(1, dim[0]+1):
        i_min = max(1, i-1)
        i_max = min(i+1, dim[0])
        for j in range(1, dim[1]+1):
            j_min = max(1, j-1)
            j_max = min(j+1, dim[1])
            for k in range(1, dim[2]+1):
                k_min = max(1, k-1)
                k_max = min(k+1, dim[2])
                val_q3 = q_3[i-1, j-1, k-1]
                count = 0
                for I2 in range(i_min, i_max+1):
                    for J2 in range(j_min, j_max+1):
                        for K2 in range(k_min, k_max+1):
                            if (I2 == i) & (J2 == j) & (K2 == k):
                                continue
                            else:
                                # a = 0
                                if (val_q3 - q_3[I2-1, J2-1, K2-1] == 0):
                                    count += 1

                ngldm[val_q3-1, count] = ngldm[val_q3-1, count] + 1

    # Last column was for the NaN voxels, to be removed
    ngldm = np.delete(ngldm, -1, 0)
    stop = np.nonzero(np.sum(ngldm, 0))[0][-1]
    ngldm = np.delete(ngldm, range(stop+1, np.shape(ngldm)[1]+1), 1)

    return ngldm

def extract_all(vol: np.ndarray) -> Dict :
    """Compute NGLDM features

    Args:
        vol (np.ndarray): volume with discretised intensities as 3D numpy array (x, y, z)
        method (str, optional): Either 'old' (deprecated) or 'new' (faster) method.

    Raises:
        ValueError: Ngldm should either be calculated using the faster \"new\" method, or the slow \"old\" method.

    Returns:
        dict: Dictionary of NGTDM features.
    """
    
    ngldm = get_ngldm_features(vol=vol)

    return ngldm


def get_ngldm_features(vol: np.ndarray,
                       ngldm_spatial_method: str="3d",
                       ngldm_diff_lvl: float=0.0,
                       ngldm_dist: float=1.0) -> Dict:
    """Extract neighbouring grey level dependence matrix-based features from the intensity roi mask.
    These features refer to "Neighbouring grey level dependence based features" (ID = REK0) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:

        vol (ndarray): volume with discretised intensities as 3D numpy array (x, y, z).
        intensity_range (ndarray): range of potential discretised intensities,
                         provided as a list: [minimal discretised intensity, maximal discretised intensity].
                         If one or both values are unknown, replace the respective values with np.nan.
        ngldm_spatial_method(str): spatial method which determines the way neighbouring grey level dependence
                              matrices are calculated and how features are determined. One of "2d", "2.5d" or "3d".
        ngldm_diff_lvl (float): also called coarseness. Coarseness determines which intensity 
                        differences are allowed for intensities to be considered similar. Typically 0, and
                        changing discretisation levels may have the same effect as increasing
                        the coarseness parameter.
        ngldm_dist (float): the chebyshev distance that forms a local neighbourhood around a center voxel.

    Returns:
        dict: dictionary with feature values.
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
                        ngldm_list += [GreyLevelDependenceMatrix(distance=int(ii_dist), diff_lvl=ii_diff_lvl,
                                                                 spatial_method=ii_spatial.lower(), img_slice=ii_slice)]

                # Perform 3D analysis
                elif ii_spatial.lower() == "3d":

                    # Add ngldm matrices to list
                    ngldm_list += [GreyLevelDependenceMatrix(distance=int(ii_dist), diff_lvl=ii_diff_lvl,
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
                    feat_run_list += [ngldm.compute_ngldm_features()]

                # Average feature values
                feat_list += [pd.concat(feat_run_list, axis=0).mean(axis=0, skipna=True).to_frame().transpose()]

    # Merge feature tables into a single dictionary
    df_feat = pd.concat(feat_list, axis=1).to_dict(orient="records")[0]

    return df_feat


def combine_ngldm_matrices(ngldm_list: List,
                           spatial_method: str) -> List:
    """Function to merge neighbouring grey level dependence matrices prior to feature calculation.

    Args:
        ngldm_list (List): list of GreyLevelDependenceMatrix objects.
        spatial_method (str): spatial method which determines the way neighbouring grey level
                              dependence matrices are calculated and how features are determined.
                              One of "2d", "2.5d" or "3d".

    Returns:
        List: list of one or more merged GreyLevelDependenceMatrix objects.
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

    def __init__(self,
                 distance: float,
                 diff_lvl: float,
                 spatial_method: str,
                 img_slice: np.ndarray=None,
                 matrix: np.ndarray=None,
                 n_v: float=None) ->  None:
        """Initialising function for a new neighbouring grey level dependence ``matrix``.
    
        Args:
            distance (float): chebyshev ``distance`` used to determine the local neighbourhood.
            diff_lvl (float): coarseness parameter which determines which intensities are considered similar.
            spatial_method (str): spatial method used to calculate the ngldm: 2d, 2.5d or 3d
            img_slice (ndarray): corresponding slice index (only if the ngldm corresponds to a 2d image slice)
            matrix (ndarray): the actual ngldm in sparse format (row, column, count)
            n_v (float): the number of voxels in the volume
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

    def calculate_ngldm_matrix(self,
                               df_img: pd.DataFrame,
                               img_dims: int):
        """
        Function that calculates an ngldm for the settings provided during initialisation and the input image.
        
        Args:
            df_img (pd.DataFrame): data table containing image intensities, x, y and z coordinates,
                and mask labels corresponding to voxels in the volume.
            img_dims (int): dimensions of the image volume
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

    def compute_ngldm_features(self) -> pd.DataFrame:
        """Computes neighbouring grey level dependence matrix features for the current neighbouring grey level dependence matrix.
        
        Returns:
            pandas data frame: with values for each feature.
        """
        # Create feature table
        feat_names = ["Fngl_lde",
                      "Fngl_hde",
                      "Fngl_lgce",
                      "Fngl_hgce",
                      "Fngl_ldlge",
                      "Fngl_ldhge",
                      "Fngl_hdlge",
                      "Fngl_hdhge",
                      "Fngl_glnu",
                      "Fngl_glnu_norm",
                      "Fngl_dcnu",
                      "Fngl_dcnu_norm",
                      "Fngl_gl_var",
                      "Fngl_dc_var",
                      "Fngl_dc_entr",
                      "Fngl_dc_energy"]
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
        """
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

def get_dict(vol: np.ndarray) -> dict:
    """
    Extract neighbouring grey level dependence matrix-based features from the intensity roi mask.

    Args:
        vol (ndarray): volume with discretised intensities as 3D numpy array (x, y, z)
    
    Returns:
        dict: dictionary with feature values

    """
    ngldm_dict =  get_ngldm_features(vol, intensity_range=[np.nan, np.nan])
    return ngldm_dict

def lde(ngldm_dict: np.ndarray)-> float:
    """
    Computes low dependence emphasis feature.
    This feature refers to "Fngl_lde" (ID = SODN) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: low depence emphasis value

    """
    return ngldm_dict["Fngl_lde"]

def hde(ngldm_dict: np.ndarray)-> float:
    """
    Computes high dependence emphasis feature.
    This feature refers to "Fngl_hde" (ID = IMOQ) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: high depence emphasis value

    """
    return ngldm_dict["Fngl_hde"]

def lgce(ngldm_dict: np.ndarray)-> float:
    """
    Computes low grey level count emphasis feature.
    This feature refers to "Fngl_lgce" (ID = TL9H) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: low grey level count emphasis value

    """
    return ngldm_dict["Fngl_lgce"]

def hgce(ngldm_dict: np.ndarray)-> float:
    """
    Computes high grey level count emphasis feature.
    This feature refers to "Fngl_hgce" (ID = OAE7) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: high grey level count emphasis value

    """
    return ngldm_dict["Fngl_hgce"]

def ldlge(ngldm_dict: np.ndarray)-> float:
    """
    Computes low dependence low grey level emphasis feature.
    This feature refers to "Fngl_ldlge" (ID = EQ3F) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: low dependence low grey level emphasis value

    """
    return ngldm_dict["Fngl_ldlge"]

def ldhge(ngldm_dict: np.ndarray)-> float:
    """
    Computes low dependence high grey level emphasis feature.
    This feature refers to "Fngl_ldhge" (ID = JA6D) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: low dependence high grey level emphasis value

    """
    return ngldm_dict["Fngl_ldhge"]

def hdlge(ngldm_dict: np.ndarray)-> float:
    """
    Computes high dependence low grey level emphasis feature.
    This feature refers to "Fngl_hdlge" (ID = NBZI) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: high dependence low grey level emphasis value

    """
    return ngldm_dict["Fngl_hdlge"]

def hdhge(ngldm_dict: np.ndarray)-> float:
    """
    Computes high dependence high grey level emphasis feature.
    This feature refers to "Fngl_hdhge" (ID = 9QMG) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: high dependence high grey level emphasis value

    """
    return ngldm_dict["Fngl_hdhge"]

def glnu(ngldm_dict: np.ndarray)-> float:
    """
    Computes grey level non-uniformity feature.
    This feature refers to "Fngl_glnu" (ID = FP8K) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: grey level non-uniformity value

    """
    return ngldm_dict["Fngl_glnu"]

def glnu_norm(ngldm_dict: np.ndarray)-> float:
    """
    Computes grey level non-uniformity normalised feature.
    This feature refers to "Fngl_glnu_norm" (ID = 5SPA) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: grey level non-uniformity normalised value

    """
    return ngldm_dict["Fngl_glnu_norm"]

def dcnu(ngldm_dict: np.ndarray)-> float:
    """
    Computes dependence count non-uniformity feature.
    This feature refers to "Fngl_dcnu" (ID = Z87G) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: dependence count non-uniformity value

    """
    return ngldm_dict["Fngl_dcnu"]

def dcnu_norm(ngldm_dict: np.ndarray)-> float:
    """
    Computes dependence count non-uniformity normalised feature.
    This feature refers to "Fngl_dcnu_norm" (ID = OKJI) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: dependence count non-uniformity normalised value

    """
    return ngldm_dict["Fngl_dcnu_norm"]

def gl_var(ngldm_dict: np.ndarray)-> float:
    """
    Computes grey level variance feature.
    This feature refers to "Fngl_gl_var" (ID = 1PFV) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: grey level variance value

    """
    return ngldm_dict["Fngl_gl_var"]

def dc_var(ngldm_dict: np.ndarray)-> float:
    """
    Computes dependence count variance feature.
    This feature refers to "Fngl_dc_var" (ID = DNX2) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: dependence count variance value

    """
    return ngldm_dict["Fngl_dc_var"]

def dc_entr(ngldm_dict: np.ndarray)-> float:
    """
    Computes dependence count entropy feature.
    This feature refers to "Fngl_dc_entr" (ID = FCBV) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: dependence count entropy value

    """
    return ngldm_dict["Fngl_dc_entr"]

def dc_energy(ngldm_dict: np.ndarray)-> float:
    """
    Computes dependence count energy feature.
    This feature refers to "Fngl_dc_energy" (ID = CAS9) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        ngldm (ndarray): array of neighbouring grey level dependence matrix
    
    Returns:
        float: dependence count energy value

    """
    return ngldm_dict["Fngl_dc_energy"]
