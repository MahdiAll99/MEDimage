#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import Dict

import numpy as np
import pandas as pd
from deprecated import deprecated

from ..biomarkers.getGLRLMmatrix import getGLRLMmatrix
from ..utils.textureTools import (coord2index, get_neighbour_direction,
                                  is_list_all_none)


def getGLRLMfeatures(vol, distCorrection=None, method="new") -> Dict:
    """Computes GLRLM features.

    Note:
        the intensity range is currently not used.

    Args:
        vol (ndarray): 3D volume, isotropically resampled, quantized
            (e.g. Ng = 32, levels = [1, ..., Ng]), with NaNs outside the region
            of interest.
        distCorrection (Union[bool, str], optional): Set this variable to true in order to use
            discretization length difference corrections as used here:
            <https://doi.org/10.1088/0031-9155/60/14/5471>.
            Set this variable to false to replicate IBSI results.
            Or use string and specify the norm for distance weighting. Weighting is 
            only performed if this argument is "manhattan", "euclidean" or "chebyshev".
        method (str, optional): Either 'old' (deprecated) or 'new' (faster) method.

    Returns:
        Dict: Dict of the GLRLM features.Compute GLRLMfeatures.

    Raises:
        ValueError: If `method` is not 'old' or 'new'.

    Todo:
        *Enable calculation of RLM features using different spatial
            methods (2d, 2.5d, 3d)
        *Enable calculation of RLM features using different RLM
            distance settings
        *Enable calculation of RLM features for different merge methods
            (average, slice_merge, dir_merge, vol_merge)
        *Provide the range of discretised intensities from a calling
            function and pass to get_rlm_features.
        *Test if distCorrection works as expected.

    """
    if method == "old":
        glrlm = get_rlm_features_deprecated(vol=vol, distCorrection=distCorrection)

    elif method == "new":
        glrlm = get_rlm_features(vol=vol, intensity_range=[np.nan, np.nan], dist_weight_norm=distCorrection)

    else:
        raise ValueError(
            "GLRLM should either be calculated using the faster \"new\" method, or the slow \"old\" method.")

    return glrlm


def get_rlm_features(vol, 
                    intensity_range, 
                    glrlm_spatial_method="3d", 
                    glrlm_merge_method="vol_merge", 
                    dist_weight_norm=None) -> Dict:
    """Extract run length matrix-based features from the intensity roi mask.

    Note:
        This code was adapted from the in-house radiomics software created at
        OncoRay, Dresden, Germany.
    
    Args:
        vol (ndarray): volume with discretised intensities as 3D numpy array (x, y, z).
        intensity_range (ndarray): range of potential discretised intensities,
            provided as a list: [minimal discretised intensity, maximal discretised
            intensity]. If one or both values are unknown, replace the respective values 
            with np.nan.
        glrlm_spatial_method (str, optional): spatial method which determines the way
            co-occurrence matrices are calculated and how features are determined.
            MUST BE "2d", "2.5d" or "3d".
        glrlm_merge_method (str, optional): merging method which determines how features are
            calculated. One of "average", "slice_merge", "dir_merge" and "vol_merge".
            Note that not all combinations of spatial and merge method are valid.
        dist_weight_norm (Union[bool, str], optional): norm for distance weighting. Weighting is only
            performed if this argument is either "manhattan", "euclidean", "chebyshev" or bool.
    
    Returns: 
        Dict: Dict of the length matrix features.

    """
    if type(glrlm_spatial_method) is not list:
        glrlm_spatial_method = [glrlm_spatial_method]

    if type(glrlm_merge_method) is not list:
        glrlm_merge_method = [glrlm_merge_method]

    if type(dist_weight_norm) is bool:
        if dist_weight_norm:
            dist_weight_norm = "euclidean"

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
    for ii_spatial in glrlm_spatial_method:
        # Initiate list of rlm objects
        rlm_list = []

        # Perform 2D analysis
        if ii_spatial.lower() in ["2d", "2.5d"]:
            # Iterate over slices
            for ii_slice in np.arange(0, img_dims[2]):
                # Get neighbour direction and iterate over neighbours
                nbrs = get_neighbour_direction(d=1, 
                                            distance="chebyshev", 
                                            centre=False, 
                                            complete=False, 
                                            dim3=False)
                
                for ii_direction in np.arange(0, np.shape(nbrs)[1]):
                    # Add rlm matrices to list
                    rlm_list += [RunLengthMatrix(direction=nbrs[:, ii_direction], 
                                                direction_id=ii_direction, 
                                                spatial_method=ii_spatial.lower(), 
                                                img_slice=ii_slice)]

        # Perform 3D analysis
        if ii_spatial.lower() == "3d":
            # Get neighbour direction and iterate over neighbours
            nbrs = get_neighbour_direction(d=1, 
                                        distance="chebyshev", 
                                        centre=False, 
                                        complete=False, 
                                        dim3=True)

            for ii_direction in np.arange(0, np.shape(nbrs)[1]):
                # Add rlm matrices to list
                rlm_list += [RunLengthMatrix(direction=nbrs[:, ii_direction], 
                                            direction_id=ii_direction, 
                                            spatial_method=ii_spatial.lower())]

        # Calculate run length matrices
        for rlm in rlm_list:
            rlm.calculate_rlm_matrix(df_img=df_img, 
                                    img_dims=img_dims, 
                                    dist_weight_norm=dist_weight_norm)

        # Merge matrices according to the given method
        for merge_method in glrlm_merge_method:
            upd_list = combine_rlm_matrices(rlm_list=rlm_list, 
                                            merge_method=merge_method, 
                                            spatial_method=ii_spatial.lower())

            # Skip if no matrices are available (due to illegal combinations of merge and spatial methods
            if upd_list is None:
                continue

            # Calculate features
            feat_run_list = []
            for rlm in upd_list:
                feat_run_list += [rlm.calculate_rlm_features(intensity_range=intensity_range)]

            # Average feature values
            feat_list += [pd.concat(feat_run_list, axis=0).mean(axis=0, skipna=True).to_frame().transpose()]

    # Merge feature tables into a single dictionary
    df_feat = pd.concat(feat_list, axis=1).to_dict(orient="records")[0]

    return df_feat


def combine_rlm_matrices(rlm_list, merge_method, spatial_method):
    """Merges run length matrices prior to feature calculation.

    Note:
        This code was adapted from the in-house radiomics software created at
        OncoRay, Dresden, Germany.
    
    Args:
        rlm_list (List): List of RunLengthMatrix objects.
        merge_method (str): Merging method which determines how features are calculated. 
            One of "average", "slice_merge", "dir_merge" and "vol_merge". Note that not all
            combinations of spatial and merge method are valid.
        spatial_method (str): spatial method which determines the way co-occurrence 
            matrices are calculated and how features are determined. One of "2d", "2.5d"
            or "3d".
    
    Returns: 
        List[CooccurrenceMatrix]: list of one or more merged RunLengthMatrix objects.

    """
    # Initiate empty list
    use_list = []

    # For average features over direction, maintain original run length matrices
    if merge_method == "average" and spatial_method in ["2d", "3d"]:
        # Make copy of rlm_list
        for rlm in rlm_list:
            use_list += [rlm._copy()]

        # Set merge method to average
        for rlm in use_list:
            rlm.merge_method = "average"

    # Merge rlms within each slice
    elif merge_method == "slice_merge" and spatial_method == "2d":
        # Find slice_ids
        slice_id = []
        for rlm in rlm_list:
            slice_id += [rlm.slice]

        # Iterate over unique slice_ids
        for ii_slice in np.unique(slice_id):
            slice_rlm_id = np.squeeze(np.where(slice_id == ii_slice))

            # Select all matrices within the slice
            sel_matrix_list = []
            for rlm_id in slice_rlm_id:
                sel_matrix_list += [rlm_list[rlm_id].matrix]

            # Check if any matrix has been created for the currently selected slice
            if is_list_all_none(sel_matrix_list):
                # No matrix was created
                use_list += [RunLengthMatrix(direction=None, 
                                            direction_id=None, 
                                            spatial_method=spatial_method, 
                                            img_slice=ii_slice,
                                            merge_method=merge_method, 
                                            matrix=None, 
                                            n_v=0.0)]
            else:
                # Merge matrices within the slice
                merge_rlm = pd.concat(sel_matrix_list, axis=0)
                merge_rlm = merge_rlm.groupby(by=["i", "r"]).sum().reset_index()

                # Update the number of voxels within the merged slice
                merge_n_v = 0.0
                for rlm_id in slice_rlm_id:
                    merge_n_v += rlm_list[rlm_id].n_v

                # Create new run length matrix
                use_list += [RunLengthMatrix(direction=None, 
                                            direction_id=None, 
                                            spatial_method=spatial_method, 
                                            img_slice=ii_slice,
                                            merge_method=merge_method, 
                                            matrix=merge_rlm, 
                                            n_v=merge_n_v)]

    # Merge rlms within each slice
    elif merge_method == "dir_merge" and spatial_method == "2.5d":
        # Find direction ids
        dir_id = []
        for rlm in rlm_list:
            dir_id += [rlm.direction_id]

        # Iterate over unique dir_ids
        for ii_dir in np.unique(dir_id):
            dir_rlm_id = np.squeeze(np.where(dir_id == ii_dir))

            # Select all matrices with the same direction
            sel_matrix_list = []
            for rlm_id in dir_rlm_id:
                sel_matrix_list += [rlm_list[rlm_id].matrix]

            # Check if any matrix has been created for the currently selected direction
            if is_list_all_none(sel_matrix_list):
                # No matrix was created
                use_list += [RunLengthMatrix(direction=rlm_list[dir_rlm_id[0]].direction, 
                                            direction_id=ii_dir, 
                                            spatial_method=spatial_method, 
                                            img_slice=None,
                                            merge_method=merge_method, 
                                            matrix=None, 
                                            n_v=0.0)]
            else:
                # Merge matrices with the same direction
                merge_rlm = pd.concat(sel_matrix_list, axis=0)
                merge_rlm = merge_rlm.groupby(by=["i", "r"]).sum().reset_index()

                # Update the number of voxels within the merged slice
                merge_n_v = 0.0
                for rlm_id in dir_rlm_id:
                    merge_n_v += rlm_list[rlm_id].n_v

                # Create new run length matrix
                use_list += [RunLengthMatrix(direction=rlm_list[dir_rlm_id[0]].direction, 
                                            direction_id=ii_dir, 
                                            spatial_method=spatial_method, 
                                            img_slice=None,
                                            merge_method=merge_method, 
                                            matrix=merge_rlm, 
                                            n_v=merge_n_v)]

    # Merge all rlms into a single representation
    elif merge_method == "vol_merge" and spatial_method in ["2.5d", "3d"]:
        # Select all matrices within the slice
        sel_matrix_list = []
        for rlm_id in np.arange(len(rlm_list)):
            sel_matrix_list += [rlm_list[rlm_id].matrix]

        # Check if any matrix has been created
        if is_list_all_none(sel_matrix_list):
            # No matrix was created
            use_list += [RunLengthMatrix(direction=None, 
                                        direction_id=None, 
                                        spatial_method=spatial_method, 
                                        img_slice=None,
                                        merge_method=merge_method, 
                                        matrix=None, 
                                        n_v=0.0)]
        else:
            # Merge run length matrices
            merge_rlm = pd.concat(sel_matrix_list, axis=0)
            merge_rlm = merge_rlm.groupby(by=["i", "r"]).sum().reset_index()

            # Update the number of voxels
            merge_n_v = 0.0
            for rlm_id in np.arange(len(rlm_list)):
                merge_n_v += rlm_list[rlm_id].n_v

            # Create new run length matrix
            use_list += [RunLengthMatrix(direction=None, 
                                        direction_id=None, 
                                        spatial_method=spatial_method, 
                                        img_slice=None,
                                        merge_method=merge_method, 
                                        matrix=merge_rlm, 
                                        n_v=merge_n_v)]

    else:
        use_list = None

    # Return to new rlm list to calling function
    return use_list


class RunLengthMatrix:
    """Class that contains a single run length matrix.

    Note :
        Code was adapted from the in-house radiomics software created at 
        OncoRay, Dresden, Germany.

    Args:
        direction (ndarray): Direction along which neighbouring voxels are found.
        direction_id (int): Direction index to identify unique direction vectors.
        spatial_method (str): Spatial method used to calculate the co-occurrence 
            matrix: "2d", "2.5d" or "3d".
        img_slice (ndarray, optional): Corresponding slice index (only if the 
            co-occurrence matrix corresponds to a 2d image slice).
        merge_method (str, optional): Method for merging the co-occurrence matrix 
            with other co-occurrence matrices.
        matrix (pandas.DataFrame, optional): The actual co-occurrence matrix in 
            sparse format (row, column, count).
        n_v (int, optional): The number of voxels in the volume.

    Attributes:
        direction (ndarray): Direction along which neighbouring voxels are found.
        direction_id (int): Direction index to identify unique direction vectors.
        spatial_method (str): Spatial method used to calculate the co-occurrence 
            matrix: "2d", "2.5d" or "3d".
        img_slice (ndarray): Corresponding slice index (only if the co-occurrence 
            matrix corresponds to a 2d image slice).
        merge_method (str): Method for merging the co-occurrence matrix with other 
            co-occurrence matrices.
        matrix (pandas.DataFrame): The actual co-occurrence matrix in sparse format 
            (row, column, count).
        n_v (int): The number of voxels in the volume.

        """

    def __init__(self, 
                direction, 
                direction_id, 
                spatial_method, 
                img_slice=None, 
                merge_method=None, 
                matrix=None, 
                n_v=None) -> None:
        """
        Initialising function for a new run length matrix
        """

        # Direction and slice for which the current matrix is extracted
        self.direction = direction
        self.direction_id = direction_id
        self.img_slice = img_slice

        # Spatial analysis method (2d, 2.5d, 3d) and merge method (average, slice_merge, dir_merge, vol_merge)
        self.spatial_method = spatial_method

        # Place holders
        self.merge_method = merge_method
        self.matrix = matrix
        self.n_v = n_v

    def _copy(self):
        """Returns a copy of the RunLengthMatrix object."""

        return deepcopy(self)

    def _set_empty(self):
        """Creates an empty RunLengthMatrix"""
        self.n_v = 0
        self.matrix = None

    def calculate_rlm_matrix(self, df_img, img_dims, dist_weight_norm) -> None:
        """Function that calculates a run length matrix for the settings provided 
        during initialisation and the input image.

        Args:
            df_img (pandas.DataFrame): Data table containing image intensities, x, y and z coordinates, 
                and mask labels corresponding to voxels in the volume.
            img_dims (ndarray, List[float]): Dimensions of the image volume.
            dist_weight_norm (str): Norm for distance weighting. Weighting is only 
                performed if this parameter is either "manhattan", "euclidean" or "chebyshev".

        Returns:
            None. Assigns the created image table (rlm matrix) to the `matrix` attribute.
        
        Raises:
            ValueError: 
                If `self.spatial_method` is not "2d", "2.5d" or "3d".
                If `dist_weight_norm` is not "manhattan", "euclidean" or "chebyshev".

        """

        # Check if the df_img actually exists
        if df_img is None:
            self._set_empty()
            return

        # Check if the roi contains any masked voxels. If this is not the case, don't construct the GLRLM.
        if not np.any(df_img.roi_int_mask):
            self._set_empty()
            return

        # Create local copies of the image table
        if self.spatial_method == "3d":
            df_rlm = deepcopy(df_img)
        elif self.spatial_method in ["2d", "2.5d"]:
            df_rlm = deepcopy(df_img[df_img.z == self.img_slice])
            df_rlm["index_id"] = np.arange(0, len(df_rlm))
            df_rlm["z"] = 0
            df_rlm = df_rlm.reset_index(drop=True)
        else:
            raise ValueError("The spatial method for grey level run length matrices should be one of \"2d\", \"2.5d\" or \"3d\".")

        # Set grey level of voxels outside ROI to NaN
        df_rlm.loc[df_rlm.roi_int_mask == False, "g"] = np.nan

        # Set the number of voxels
        self.n_v = np.sum(df_rlm.roi_int_mask.values)

        # Determine update index number for direction
        if (self.direction[2] + self.direction[1] * img_dims[2] + self.direction[0] * img_dims[2] * img_dims[1]) >= 0:
            curr_dir = self.direction
        else:
            curr_dir = - self.direction

        # Step size
        ind_update = curr_dir[2] + curr_dir[1] * img_dims[2] + curr_dir[0] * img_dims[2] * img_dims[1]

        # Generate information concerning segments
        n_seg = ind_update  # Number of segments

        # Check if the number of segments is greater than one
        if n_seg == 0:
            self._set_empty()
            return

        seg_len = (len(df_rlm) - 1) // ind_update + 1  # Nominal segment length
        trans_seg_len = np.tile([seg_len - 1], reps=n_seg)  # Initial segment length for transitions (nominal length - 1)
        full_len_trans = n_seg - n_seg * seg_len + len(df_rlm)  # Number of full segments
        trans_seg_len[0:full_len_trans] += 1  # Update full segments

        # Create transition vector
        trans_vec = np.tile(np.arange(start=0, stop=len(df_rlm), step=ind_update), reps=ind_update)
        trans_vec += np.repeat(np.arange(start=0, stop=n_seg), repeats=seg_len)
        trans_vec = trans_vec[trans_vec < len(df_rlm)]

        # Determine valid transitions
        to_index = coord2index(x=df_rlm.x.values + curr_dir[0],
                               y=df_rlm.y.values + curr_dir[1],
                               z=df_rlm.z.values + curr_dir[2],
                               dims=img_dims)

        # Determine which transitions are valid
        end_ind = np.nonzero(to_index[trans_vec] < 0)[0]  # Find transitions that form an endpoints

        # Get an interspersed array of intensities. Runs are broken up by np.nan
        intensities = np.insert(df_rlm.g.values[trans_vec], end_ind + 1, np.nan)

        # Determine run length start and end indices
        rle_end = np.array(np.append(np.where(intensities[1:] != intensities[:-1]), len(intensities) - 1))
        rle_start = np.cumsum(np.append(0, np.diff(np.append(-1, rle_end))))[:-1]

        # Generate dataframe
        df_rltable = pd.DataFrame({"i": intensities[rle_start],
                                   "r": rle_end - rle_start + 1})
        df_rltable = df_rltable.loc[~np.isnan(df_rltable.i), :]
        df_rltable = df_rltable.groupby(by=["i", "r"]).size().reset_index(name="n")

        if dist_weight_norm in ["manhattan", "euclidean", "chebyshev"]:
            if dist_weight_norm == "manhattan":
                weight = sum(abs(self.direction))
            elif dist_weight_norm == "euclidean":
                weight = np.sqrt(sum(np.power(self.direction, 2.0)))
            elif dist_weight_norm == "chebyshev":
                weight = np.max(abs(self.direction))
            df_rltable.n /= weight

        # Add matrix to object
        self.matrix = df_rltable

    def calculate_rlm_features(self, intensity_range) -> pd.DataFrame:
        """Computes run length matrix features for the current run length matrix.

        Note:
            the intensity range is currently not used.

        Args:
            intensity_range (ndarray): Range of potential discretised intensities,
                provided as a list: [minimal discretised intensity, maximal discretised intensity]. 
                If one or both values are unknown, replace the respective values with np.nan.

        Returns:
            pandas.DataFrame: Data frame with values for each feature.

        """
        # Create feature table
        feat_names = ["Frlm_sre", "Frlm_lre", "Frlm_lgre", "Frlm_hgre", "Frlm_srlge", "Frlm_srhge", "Frlm_lrlge",
                      "Frlm_lrhge", "Frlm_glnu", "Frlm_glnu_norm", "Frlm_rlnu", "Frlm_rlnu_norm", "Frlm_r_perc",
                      "Frlm_gl_var", "Frlm_rl_var", "Frlm_rl_entr"]
        df_feat = pd.DataFrame(np.full(shape=(1, len(feat_names)), fill_value=np.nan))
        df_feat.columns = feat_names

        # Don't return data for empty slices or slices without a good matrix
        if self.matrix is None:
            # Update names
            # df_feat.columns += self._parse_feature_names()
            return df_feat
        elif len(self.matrix) == 0:
            # Update names
            # df_feat.columns += self._parse_feature_names()
            return df_feat

        # Create local copy of the run length matrix and set column names
        df_rij = deepcopy(self.matrix)
        df_rij.columns = ["i", "j", "rij"]

        # Sum over grey levels
        df_ri = df_rij.groupby(by="i")["rij"].agg(np.sum).reset_index().rename(columns={"rij": "ri"})

        # Sum over run lengths
        df_rj = df_rij.groupby(by="j")["rij"].agg(np.sum).reset_index().rename(columns={"rij": "rj"})

        # Constant definitions
        n_s = np.sum(df_rij.rij) * 1.0  # Number of runs
        n_v = self.n_v * 1.0  # Number of voxels

        ##############################################
        ######          GLRLM features          ######
        ##############################################
        # Short runs emphasis
        df_feat.loc[0, "Frlm_sre"] = np.sum(df_rj.rj / df_rj.j ** 2.0) / n_s

        # Long runs emphasis
        df_feat.loc[0, "Frlm_lre"] = np.sum(df_rj.rj * df_rj.j ** 2.0) / n_s

        # Grey level non-uniformity
        df_feat.loc[0, "Frlm_glnu"] = np.sum(df_ri.ri ** 2.0) / n_s

        # Grey level non-uniformity, normalised
        df_feat.loc[0, "Frlm_glnu_norm"] = np.sum(df_ri.ri ** 2.0) / n_s ** 2.0

        # Run length non-uniformity
        df_feat.loc[0, "Frlm_rlnu"] = np.sum(df_rj.rj ** 2.0) / n_s

        # Run length non-uniformity
        df_feat.loc[0, "Frlm_rlnu_norm"] = np.sum(df_rj.rj ** 2.0) / n_s ** 2.0

        # Run percentage
        df_feat.loc[0, "Frlm_r_perc"] = n_s / n_v

        # Low grey level run emphasis
        df_feat.loc[0, "Frlm_lgre"] = np.sum(df_ri.ri / df_ri.i ** 2.0) / n_s

        # High grey level run emphasis
        df_feat.loc[0, "Frlm_hgre"] = np.sum(df_ri.ri * df_ri.i ** 2.0) / n_s

        # Short run low grey level emphasis
        df_feat.loc[0, "Frlm_srlge"] = np.sum(df_rij.rij / (df_rij.i * df_rij.j) ** 2.0) / n_s

        # Short run high grey level emphasis
        df_feat.loc[0, "Frlm_srhge"] = np.sum(df_rij.rij * df_rij.i ** 2.0 / df_rij.j ** 2.0) / n_s

        # Long run low grey level emphasis
        df_feat.loc[0, "Frlm_lrlge"] = np.sum(df_rij.rij * df_rij.j ** 2.0 / df_rij.i ** 2.0) / n_s

        # Long run high grey level emphasis
        df_feat.loc[0, "Frlm_lrhge"] = np.sum(df_rij.rij * df_rij.i ** 2.0 * df_rij.j ** 2.0) / n_s

        # Grey level variance
        mu = np.sum(df_rij.rij * df_rij.i) / n_s
        df_feat.loc[0, "Frlm_gl_var"] = np.sum((df_rij.i - mu) ** 2.0 * df_rij.rij) / n_s

        # Run length variance
        mu = np.sum(df_rij.rij * df_rij.j) / n_s
        df_feat.loc[0, "Frlm_rl_var"] = np.sum((df_rij.j - mu) ** 2.0 * df_rij.rij) / n_s

        # Zone size entropy
        df_feat.loc[0, "Frlm_rl_entr"] = - np.sum(df_rij.rij * np.log2(df_rij.rij / n_s)) / n_s

        # Update names
        # df_feat.columns += self._parse_feature_names()

        return df_feat

    def _parse_feature_names(self) -> str:
        """"
        Adds additional settings-related identifiers to each feature.
        Not used currently, as the use of different settings for the
        run length matrix is not supported.
        """
        parse_str = ""

        # Add spatial method
        if self.spatial_method is not None:
            parse_str += "_" + self.spatial_method

        # Add merge method
        if self.merge_method is not None:
            if self.merge_method == "average":
                parse_str += "_avg"
            if self.merge_method == "slice_merge":
                parse_str += "_s_mrg"
            if self.merge_method == "dir_merge":
                parse_str += "_d_mrg"
            if self.merge_method == "vol_merge":
                parse_str += "_v_mrg"

        return parse_str


@deprecated(reason="Use the new and the faster method get_rlm_features()")
def get_rlm_features_deprecated(vol, distCorrection) -> Dict:
    """Calculates grey level run length matrix features.

     Note:
        Deprecated code. Calculates grey level run length features, but slowly.
        A newer and faster method is available : `get_rlm_features()`

    Args:
        vol (ndarray): 3D input volume.
        distCorrection (Union[bool, str], optional): Set this variable to true in order to use
            discretization length difference corrections as used here:
            <https://doi.org/10.1088/0031-9155/60/14/5471>.
            Set this variable to false to replicate IBSI results.
            Or use string and specify the norm for distance weighting. Weighting is 
            only performed if this argument is "manhattan", "euclidean" or "chebyshev".

    Returns:
        Dict: Dict of GLCM features.

    """

    glrlm = {'Frlm_sre': [],
             'Frlm_lre': [],
             'Frlm_lgre': [],
             'Frlm_hgre': [],
             'Frlm_srlge': [],
             'Frlm_srhge': [],
             'Frlm_lrlge': [],
             'Frlm_lrhge': [],
             'Frlm_glnu': [],
             'Frlm_glnu_norm': [],
             'Frlm_rlnu': [],
             'Frlm_rlnu_norm': [],
             'Frlm_r_perc': [],
             'Frlm_gl_var': [],
             'Frlm_rl_var': [],
             'Frlm_rl_entr': []}

    # GET THE GLRLM MATRIX
    vol = vol.copy()
    # Correct definition, without any assumption
    levels = np.arange(1, np.max(vol[~np.isnan(vol[:])])+1)

    if distCorrection is None:
        GLRLM = getGLRLMmatrix(vol, levels)
    else:
        GLRLM = (getGLRLMmatrix(vol, levels, distCorrection))

    Ns = np.sum(GLRLM)
    GLRLM = GLRLM/Ns  # Normalization of GLRLM
    sz = np.shape(GLRLM)  # Size of GLRLM
    cVect = range(1, sz[1]+1)  # Row vectors
    rVect = range(1, sz[0]+1)  # Column vectors
    # Column and row indicators for each entry of the GLRLM
    cMat, rMat = np.meshgrid(cVect, rVect)
    pg = np.transpose(np.sum(GLRLM, 1))  # Gray-Level Run-Number Vector
    pr = np.sum(GLRLM, 0)  # Run-Length Run-Number Vector

    ##############################################
    ######          GLRLM features          ######
    ##############################################
    # Short runs emphasis
    glrlm['Frlm_sre'] = (np.matmul(pr, np.transpose(np.power(1.0/np.array(cVect), 2))))

    # Long runs emphasis
    glrlm['Frlm_lre'] = np.matmul(pr, np.transpose(np.power(np.array(cVect), 2)))

    # Low grey level run emphasis
    glrlm['Frlm_lgre'] = np.matmul(pg, np.transpose(np.power(1.0/np.array(rVect), 2)))

    # High grey level run emphasis
    glrlm['Frlm_hgre'] = np.matmul(pg, np.transpose(np.power(np.array(rVect), 2)))

    # Short run low grey level emphasis
    glrlm['Frlm_srlge'] = np.sum(np.sum(GLRLM*(np.power(1.0/rMat, 2))*(np.power(1.0/cMat, 2))))

    # Short run high grey level emphasis
    glrlm['Frlm_srhge'] = np.sum(np.sum(GLRLM*(np.power(rMat, 2))*(np.power(1.0/cMat, 2))))

    # Long run low grey levels emphasis
    glrlm['Frlm_lrlge'] = np.sum(np.sum(GLRLM*(np.power(1.0/rMat, 2))*(np.power(cMat, 2))))

    # Long run high grey level emphasis
    glrlm['Frlm_lrhge'] = np.sum(np.sum(GLRLM*(np.power(rMat, 2))*(np.power(cMat, 2))))

    # Gray level non-uniformity
    temp = np.sum(np.power(pg, 2))
    glrlm['Frlm_glnu'] = temp * Ns

    # Gray level non-uniformity normalised
    glrlm['Frlm_glnu_norm'] = temp

    # Run length non-uniformity
    temp = np.sum(np.power(pr, 2))
    glrlm['Frlm_rlnu'] = temp * Ns

    # Run length non-uniformity normalised
    glrlm['Frlm_rlnu_norm'] = temp

    # Run percentage
    glrlm['Frlm_r_perc'] = np.sum(pg)/(np.matmul(pr, np.transpose(cVect)))

    # Grey level variance
    temp = rMat * GLRLM
    u = np.sum(temp)
    temp = (np.power(rMat - u, 2)) * GLRLM
    glrlm['Frlm_gl_var'] = np.sum(temp)

    # Run length variance
    temp = cMat * GLRLM
    u = np.sum(temp)
    temp = (np.power(cMat - u, 2)) * GLRLM
    glrlm['Frlm_rl_var'] = np.sum(temp)

    # Run entropy
    valPos = GLRLM[np.nonzero(GLRLM)]
    temp = valPos * np.log2(valPos)
    glrlm['Frlm_rl_entr'] = -np.sum(temp)

    return glrlm
