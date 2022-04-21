#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
from typing import Dict, List

import numpy as np
import pandas as pd
from deprecated import deprecated
from utils.textureTools import (coord2index, get_neighbour_direction,
                                get_value, is_list_all_none)

from biomarkers.gclm_CrossDiagProb import gclm_CrossDiagProb
from biomarkers.gclm_DiagProb import gclm_DiagProb
from biomarkers.getGLCMmatrix import getGLCMmatrix


def getGLCMfeatures(vol, distCorrection=None, method="new") -> Dict:
    """Computes GLCM features.

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
        Dict: Dict of the GLCM features.
    
    Raises:
        ValueError: If `method` is not 'old' or 'new'.

    Todo:
        *Enable calculation of CM features using different spatial
            methods (2d, 2.5d, 3d)
        *Enable calculation of CM features using different CM
            distance settings
        *Enable calculation of CM features for different merge methods
            (average, slice_merge, dir_merge, vol_merge)
        *Provide the range of discretised intensities from a calling
            function and pass to get_cm_features.
        *Test if distCorrection works as expected.

    """
    if method == "old":
        glcm = get_cm_features_deprecated(
            vol=vol, distCorrection=distCorrection)

    elif method == "new":
        glcm = get_cm_features(vol=vol, intensity_range=[
                               np.nan, np.nan], dist_weight_norm=distCorrection)

    else:
        raise ValueError(
            "GLCM should either be calculated using the faster \"new\" method, or the slow \"old\" method.")

    return glcm


def get_cm_features(vol, 
                    intensity_range, 
                    glcm_spatial_method="3d", 
                    glcm_dist=1.0, 
                    glcm_merge_method="vol_merge", 
                    dist_weight_norm=None) -> Dict:
    """Extracts co-occurrence matrix-based features from the intensity roi mask.

    Note:
        This code was adapted from the in-house radiomics software created at
        OncoRay, Dresden, Germany.
    
    Args:
        vol (ndarray): volume with discretised intensities as 3D numpy array (x, y, z).
        intensity_range (ndarray): range of potential discretised intensities,
            provided as a list: [minimal discretised intensity, maximal discretised
            intensity]. If one or both values are unknown, replace the respective values 
            with np.nan.
        glcm_spatial_method (str, optional): spatial method which determines the way
            co-occurrence matrices are calculated and how features are determined.
            MUST BE "2d", "2.5d" or "3d".
        glcm_dist (float, optional): chebyshev distance for comparison between neighbouring
            voxels.
        glcm_merge_method (, optional): merging method which determines how features are
            calculated. One of "average", "slice_merge", "dir_merge" and "vol_merge".
            Note that not all combinations of spatial and merge method are valid.
        dist_weight_norm (Union[bool, str], optional): norm for distance weighting. Weighting is only
            performed if this argument is either "manhattan", "euclidean", "chebyshev" or bool.
    
    Returns: 
        Dict: Dict of the GLCM features.
    
    Raises:
        ValueError: If `glcm_spatial_method` is not "2d", "2.5d" or "3d".
    
    """
    if type(glcm_spatial_method) is not list:
        glcm_spatial_method = [glcm_spatial_method]

    if type(glcm_dist) is not list:
        glcm_dist = [glcm_dist]

    if type(glcm_merge_method) is not list:
        glcm_merge_method = [glcm_merge_method]

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
    for ii_spatial in glcm_spatial_method:
        # Iterate over distances
        for ii_dist in glcm_dist:
            # Initiate list of glcm objects
            glcm_list = []
            # Perform 2D analysis
            if ii_spatial.lower() in ["2d", "2.5d"]:
                # Iterate over slices
                for ii_slice in np.arange(0, img_dims[2]):
                    # Get neighbour direction and iterate over neighbours
                    nbrs = get_neighbour_direction(
                        d=1, 
                        distance="chebyshev", 
                        centre=False, 
                        complete=False, 
                        dim3=False) * np.int(ii_dist)
                    for ii_direction in np.arange(0, np.shape(nbrs)[1]):
                        # Add glcm matrices to list
                        glcm_list += [CooccurrenceMatrix(distance=np.int(ii_dist), 
                                                        direction=nbrs[:, ii_direction], 
                                                        direction_id=ii_direction,
                                                        spatial_method=ii_spatial.lower(), 
                                                        img_slice=ii_slice)]

            # Perform 3D analysis
            elif ii_spatial.lower() == "3d":
                # Get neighbour direction and iterate over neighbours
                nbrs = get_neighbour_direction(d=1, 
                                            distance="chebyshev", 
                                            centre=False, 
                                            complete=False, 
                                            dim3=True) * np.int(ii_dist)

                for ii_direction in np.arange(0, np.shape(nbrs)[1]):
                    # Add glcm matrices to list
                    glcm_list += [CooccurrenceMatrix(distance=np.int(ii_dist), 
                                                    direction=nbrs[:, ii_direction], 
                                                    direction_id=ii_direction,
                                                    spatial_method=ii_spatial.lower())]

            else:
                raise ValueError(
                    "GCLM matrices can be determined in \"2d\", \"2.5d\" and \"3d\". \
                        The requested method (%s) is not implemented.", ii_spatial)

            # Calculate glcm matrices
            for glcm in glcm_list:
                glcm.calculate_cm_matrix(
                    df_img=df_img, img_dims=img_dims, dist_weight_norm=dist_weight_norm)

            # Merge matrices according to the given method
            for merge_method in glcm_merge_method:
                upd_list = combine_matrices(
                    glcm_list=glcm_list, merge_method=merge_method, spatial_method=ii_spatial.lower())

                # Skip if no matrices are available (due to illegal combinations of merge and spatial methods
                if upd_list is None:
                    continue

                # Calculate features
                feat_run_list = []
                for glcm in upd_list:
                    feat_run_list += [glcm.calculate_cm_features(
                        intensity_range=intensity_range)]

                # Average feature values
                feat_list += [pd.concat(feat_run_list, axis=0).mean(axis=0, skipna=True).to_frame().transpose()]

    # Merge feature tables into a single table and return as a dictionary
    df_feat = pd.concat(feat_list, axis=1).to_dict(orient="records")[0]

    return df_feat


def combine_matrices(glcm_list, merge_method, spatial_method) -> List:
    """Merges co-occurrence matrices prior to feature calculation.

    Note:
        This code was adapted from the in-house radiomics software created at
        OncoRay, Dresden, Germany.
    
    Args:
        glcm_list (List): List of CooccurrenceMatrix objects.
        merge_method (str): Merging method which determines how features are calculated. 
            One of "average", "slice_merge", "dir_merge" and "vol_merge". Note that not all
            combinations of spatial and merge method are valid.
        spatial_method (str): spatial method which determines the way co-occurrence 
            matrices are calculated and how features are determined. One of "2d", "2.5d"
            or "3d".
    
    Returns: 
        List[CooccurrenceMatrix]: list of one or more merged CooccurrenceMatrix objects.

    """
    # Initiate empty list
    use_list = []

    # For average features over direction, maintain original glcms
    if merge_method == "average" and spatial_method in ["2d", "3d"]:
        # Make copy of glcm_list
        for glcm in glcm_list:
            use_list += [glcm._copy()]

        # Set merge method to average
        for glcm in use_list:
            glcm.merge_method = "average"

    # Merge glcms by slice
    elif merge_method == "slice_merge" and spatial_method == "2d":
        # Find slice_ids
        slice_id = []
        for glcm in glcm_list:
            slice_id += [glcm.slice]

        # Iterate over unique slice_ids
        for ii_slice in np.unique(slice_id):
            slice_glcm_id = np.squeeze(np.where(slice_id == ii_slice))

            # Select all matrices within the slice
            sel_matrix_list = []
            for glcm_id in slice_glcm_id:
                sel_matrix_list += [glcm_list[glcm_id].matrix]

            # Check if any matrix has been created for the currently selected slice
            if is_list_all_none(sel_matrix_list):
                # No matrix was created
                use_list += [CooccurrenceMatrix(distance=glcm_list[slice_glcm_id[0]].distance, 
                                                direction=None, 
                                                direction_id=None,
                                                spatial_method=spatial_method, 
                                                img_slice=ii_slice, 
                                                merge_method=merge_method,
                                                matrix=None, 
                                                n_v=0.0)]
            else:
                # Merge matrices within the slice
                merge_cm = pd.concat(sel_matrix_list, axis=0)
                merge_cm = merge_cm.groupby(by=["i", "j"]).sum().reset_index()

                # Update the number of voxels within the merged slice
                merge_n_v = 0.0
                for glcm_id in slice_glcm_id:
                    merge_n_v += glcm_list[glcm_id].n_v

                # Create new cooccurrence matrix
                use_list += [CooccurrenceMatrix(distance=glcm_list[slice_glcm_id[0]].distance, 
                                                direction=None, 
                                                direction_id=None,
                                                spatial_method=spatial_method, 
                                                img_slice=ii_slice, 
                                                merge_method=merge_method,
                                                matrix=merge_cm, 
                                                n_v=merge_n_v)]

    # Merge glcms by direction
    elif merge_method == "dir_merge" and spatial_method == "2.5d":
        # Find slice_ids
        dir_id = []
        for glcm in glcm_list:
            dir_id += [glcm.direction_id]

        # Iterate over unique directions
        for ii_dir in np.unique(dir_id):
            dir_glcm_id = np.squeeze(np.where(dir_id == ii_dir))

            # Select all matrices with the same direction
            sel_matrix_list = []
            for glcm_id in dir_glcm_id:
                sel_matrix_list += [glcm_list[glcm_id].matrix]

            # Check if any matrix has been created for the currently selected direction
            if is_list_all_none(sel_matrix_list):
                # No matrix was created
                use_list += [CooccurrenceMatrix(distance=glcm_list[dir_glcm_id[0]].distance, 
                                                direction=glcm_list[dir_glcm_id[0]].direction, 
                                                direction_id=ii_dir,
                                                spatial_method=spatial_method, 
                                                img_slice=None, 
                                                merge_method=merge_method,
                                                matrix=None, n_v=0.0)]
            else:
                # Merge matrices with the same direction
                merge_cm = pd.concat(sel_matrix_list, axis=0)
                merge_cm = merge_cm.groupby(by=["i", "j"]).sum().reset_index()

                # Update the number of voxels for the merged matrices with the same direction
                merge_n_v = 0.0
                for glcm_id in dir_glcm_id:
                    merge_n_v += glcm_list[glcm_id].n_v

                # Create new cooccurrence matrix
                use_list += [CooccurrenceMatrix(distance=glcm_list[dir_glcm_id[0]].distance, 
                                                direction=glcm_list[dir_glcm_id[0]].direction, 
                                                direction_id=ii_dir,
                                                spatial_method=spatial_method, 
                                                img_slice=None, 
                                                merge_method=merge_method,
                                                matrix=merge_cm, 
                                                n_v=merge_n_v)]

    # Merge all glcms into a single representation
    elif merge_method == "vol_merge" and spatial_method in ["2.5d", "3d"]:
        # Select all matrices within the slice
        sel_matrix_list = []
        for glcm_id in np.arange(len(glcm_list)):
            sel_matrix_list += [glcm_list[glcm_id].matrix]

        # Check if any matrix was created
        if is_list_all_none(sel_matrix_list):
            # In case no matrix was created
            use_list += [CooccurrenceMatrix(distance=glcm_list[0].distance, 
                                            direction=None, 
                                            direction_id=None,
                                            spatial_method=spatial_method, 
                                            img_slice=None, 
                                            merge_method=merge_method,
                                            matrix=None, 
                                            n_v=0.0)]
        else:
            # Merge cooccurrence matrices
            merge_cm = pd.concat(sel_matrix_list, axis=0)
            merge_cm = merge_cm.groupby(by=["i", "j"]).sum().reset_index()

            # Update the number of voxels
            merge_n_v = 0.0
            for glcm_id in np.arange(len(glcm_list)):
                merge_n_v += glcm_list[glcm_id].n_v

            # Create new cooccurrence matrix
            use_list += [CooccurrenceMatrix(distance=glcm_list[0].distance, 
                                            direction=None, 
                                            direction_id=None,
                                            spatial_method=spatial_method, 
                                            img_slice=None, 
                                            merge_method=merge_method,
                                            matrix=merge_cm, 
                                            n_v=merge_n_v)]
    else:
        use_list = None

    return use_list


class CooccurrenceMatrix:
    """ Class that contains a single co-occurrence matrix.

    Note :
        Code was adapted from the in-house radiomics software created at 
        OncoRay, Dresden, Germany.

    Args:
        distance (int): Chebyshev distance.
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
        distance (int): Chebyshev distance.
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
                distance, 
                direction, 
                direction_id, 
                spatial_method, 
                img_slice=None, 
                merge_method=None, 
                matrix=None, 
                n_v=None) -> None:

        # Distance used
        self.distance = distance

        # Direction and slice for which the current matrix is extracted
        self.direction = direction
        self.direction_id = direction_id
        self.img_slice = img_slice

        # Spatial analysis method (2d, 2.5d, 3d) and merge method (average, slice_merge, dir_merge, vol_merge)
        self.spatial_method = spatial_method
        self.merge_method = merge_method

        # Place holders
        self.matrix = matrix
        self.n_v = n_v

    def _copy(self):
        """
        Returns a copy of the co-occurrence matrix object.
        """
        return deepcopy(self)

    def calculate_cm_matrix(self, df_img, img_dims, dist_weight_norm) -> None:
        """Function that calculates a co-occurrence matrix for the settings provided during 
        initialisation and the input image.

        Args:
            df_img (pandas.DataFrame): Data table containing image intensities, x, y and z coordinates, 
                and mask labels corresponding to voxels in the volume.
            img_dims (ndarray | List[float]): Dimensions of the image volume.
            dist_weight_norm (str): Norm for distance weighting. Weighting is only 
                performed if this parameter is either "manhattan", "euclidean" or "chebyshev".

        Returns:
            None. Assigns the created image table (cm matrix) to the `matrix` attribute.
        
        Raises:
            ValueError: 
                If `self.spatial_method` is not "2d", "2.5d" or "3d".
                If `dist_weight_norm` is not "manhattan", "euclidean" or "chebyshev".

        """
        # Check if the roi contains any masked voxels. If this is not the case, don't construct the GLCM.
        if not np.any(df_img.roi_int_mask):
            self.n_v = 0
            self.matrix = None

            return None

        # Create local copies of the image table
        if self.spatial_method == "3d":
            df_cm = deepcopy(df_img)
        elif self.spatial_method in ["2d", "2.5d"]:
            df_cm = deepcopy(df_img[df_img.z == self.img_slice])
            df_cm["index_id"] = np.arange(0, len(df_cm))
            df_cm["z"] = 0
            df_cm = df_cm.reset_index(drop=True)
        else:
            raise ValueError(
                "The spatial method for grey level co-occurrence matrices should be one of \"2d\", \"2.5d\" or \"3d\".")

        # Set grey level of voxels outside ROI to NaN
        df_cm.loc[df_cm.roi_int_mask == False, "g"] = np.nan

        # Determine potential transitions
        df_cm["to_index"] = coord2index(x=df_cm.x.values + self.direction[0],
                                        y=df_cm.y.values + self.direction[1],
                                        z=df_cm.z.values + self.direction[2],
                                        dims=img_dims)

        # Get grey levels from transitions
        df_cm["to_g"] = get_value(x=df_cm.g.values, index=df_cm.to_index.values)

        # Check if any transitions exist.
        if np.all(np.isnan(df_cm[["to_g"]])):
            self.n_v = 0
            self.matrix = None

            return None

        # Count occurrences of grey level transitions
        df_cm = df_cm.groupby(by=["g", "to_g"]).size().reset_index(name="n")

        # Append grey level transitions in opposite direction
        df_cm_inv = pd.DataFrame({"g": df_cm.to_g, "to_g": df_cm.g, "n": df_cm.n})
        df_cm = df_cm.append(df_cm_inv, ignore_index=True)

        # Sum occurrences of grey level transitions
        df_cm = df_cm.groupby(by=["g", "to_g"]).sum().reset_index()

        # Rename columns
        df_cm.columns = ["i", "j", "n"]

        if dist_weight_norm in ["manhattan", "euclidean", "chebyshev"]:
            if dist_weight_norm == "manhattan":
                weight = sum(abs(self.direction))
            elif dist_weight_norm == "euclidean":
                weight = np.sqrt(sum(np.power(self.direction, 2.0)))
            elif dist_weight_norm == "chebyshev":
                weight = np.max(abs(self.direction))
            df_cm.n /= weight
              
        elif dist_weight_norm is not None:
            raise ValueError("Invalid distance norm was provided. Must be one of \
                \"manhattan\", \"euclidean\" or \"chebyshev\".")
            

        # Set the number of voxels
        self.n_v = np.sum(df_cm.n)

        # Add matrix and number of voxels to object
        self.matrix = df_cm

    def calculate_cm_features(self, intensity_range):
        """Wrapper to json.dump function.

        Args:
            intensity_range (ndarray): Range of potential discretised intensities,
                provided as a list: [minimal discretised intensity, maximal discretised intensity]. 
                If one or both values are unknown, replace the respective values with np.nan.

        Returns:
            pandas.DataFrame: Data frame with values for each feature.

        """
        # Create feature table
        feat_names = ["Fcm_joint_max", "Fcm_joint_avg", "Fcm_joint_var", "Fcm_joint_entr",
                      "Fcm_diff_avg", "Fcm_diff_var", "Fcm_diff_entr",
                      "Fcm_sum_avg", "Fcm_sum_var", "Fcm_sum_entr",
                      "Fcm_energy", "Fcm_contrast", "Fcm_dissimilarity",
                      "Fcm_inv_diff", "Fcm_inv_diff_norm", "Fcm_inv_diff_mom", 
                      "Fcm_inv_diff_mom_norm", "Fcm_inv_var", "Fcm_corr", 
                      "Fcm_auto_corr", "Fcm_clust_tend", "Fcm_clust_shade", 
                      "Fcm_clust_prom", "Fcm_info_corr1", "Fcm_info_corr2"]

        df_feat = pd.DataFrame(np.full(shape=(1, len(feat_names)), fill_value=np.nan))
        df_feat.columns = feat_names

        # Don't return data for empty slices or slices without a good matrix
        if self.matrix is None:
            # Update names
            df_feat.columns += self._parse_names()
            return df_feat
        elif len(self.matrix) == 0:
            # Update names
            df_feat.columns += self._parse_names()
            return df_feat

        # Occurrence data frames
        df_pij = deepcopy(self.matrix)
        df_pij["pij"] = df_pij.n / sum(df_pij.n)
        df_pi = df_pij.groupby(by="i")["pij"].agg(np.sum).reset_index().rename(columns={"pij": "pi"})
        df_pj = df_pij.groupby(by="j")["pij"].agg(np.sum).reset_index().rename(columns={"pij": "pj"})

        # Diagonal probilities p(i-j)
        df_pimj = deepcopy(df_pij)
        df_pimj["k"] = np.abs(df_pimj.i - df_pimj.j)
        df_pimj = df_pimj.groupby(by="k")["pij"].agg(np.sum).reset_index().rename(columns={"pij": "pimj"})

        # Cross-diagonal probabilities p(i+j)
        df_pipj = deepcopy(df_pij)
        df_pipj["k"] = df_pipj.i + df_pipj.j
        df_pipj = df_pipj.groupby(by="k")["pij"].agg(np.sum).reset_index().rename(columns={"pij": "pipj"})

        # Merger of df.p_ij, df.p_i and df.p_j
        df_pij = pd.merge(df_pij, df_pi, on="i")
        df_pij = pd.merge(df_pij, df_pj, on="j")

        # Constant definitions
        intensity_range_loc = deepcopy(intensity_range)
        if np.isnan(intensity_range[0]):
            intensity_range_loc[0] = np.min(df_pi.i) * 1.0
        if np.isnan(intensity_range[1]):
            intensity_range_loc[1] = np.max(df_pi.i) * 1.0
        # Number of grey levels
        n_g = intensity_range_loc[1] - intensity_range_loc[0] + 1.0

        ###############################################
        ######           GLCM features           ######
        ###############################################
        # Joint maximum
        df_feat.loc[0, "Fcm_joint_max"] = np.max(df_pij.pij)

        # Joint average
        df_feat.loc[0, "Fcm_joint_avg"] = np.sum(df_pij.i * df_pij.pij)

        # Joint variance
        mu = np.sum(df_pij.i * df_pij.pij)
        df_feat.loc[0, "Fcm_joint_var"] = np.sum((df_pij.i - mu) ** 2.0 * df_pij.pij)

        # Joint entropy
        df_feat.loc[0, "Fcm_joint_entr"] = -np.sum(df_pij.pij * np.log2(df_pij.pij))

        # Difference average
        df_feat.loc[0, "Fcm_diff_avg"] = np.sum(df_pimj.k * df_pimj.pimj)

        # Difference variance
        mu = np.sum(df_pimj.k * df_pimj.pimj)
        df_feat.loc[0, "Fcm_diff_var"] = np.sum((df_pimj.k - mu) ** 2.0 * df_pimj.pimj)

        # Difference entropy
        df_feat.loc[0, "Fcm_diff_entr"] = -np.sum(df_pimj.pimj * np.log2(df_pimj.pimj))

        # Sum average
        df_feat.loc[0, "Fcm_sum_avg"] = np.sum(df_pipj.k * df_pipj.pipj)

        # Sum variance
        mu = np.sum(df_pipj.k * df_pipj.pipj)
        df_feat.loc[0, "Fcm_sum_var"] = np.sum((df_pipj.k - mu) ** 2.0 * df_pipj.pipj)

        # Sum entropy
        df_feat.loc[0, "Fcm_sum_entr"] = -np.sum(df_pipj.pipj * np.log2(df_pipj.pipj))

        # Angular second moment
        df_feat.loc[0, "Fcm_energy"] = np.sum(df_pij.pij ** 2.0)

        # Contrast
        df_feat.loc[0, "Fcm_contrast"] = np.sum((df_pij.i - df_pij.j) ** 2.0 * df_pij.pij)

        # Dissimilarity
        df_feat.loc[0, "Fcm_dissimilarity"] = np.sum(np.abs(df_pij.i - df_pij.j) * df_pij.pij)

        # Inverse difference
        df_feat.loc[0, "Fcm_inv_diff"] = np.sum(df_pij.pij / (1.0 + np.abs(df_pij.i - df_pij.j)))

        # Inverse difference normalised
        df_feat.loc[0, "Fcm_inv_diff_norm"] = np.sum(df_pij.pij / (1.0 + np.abs(df_pij.i - df_pij.j) / n_g))

        # Inverse difference moment
        df_feat.loc[0, "Fcm_inv_diff_mom"] = np.sum(df_pij.pij / (1.0 + (df_pij.i - df_pij.j) ** 2.0))

        # Inverse difference moment normalised
        df_feat.loc[0, "Fcm_inv_diff_mom_norm"] = np.sum(df_pij.pij / (1.0 + (df_pij.i - df_pij.j) ** 2.0 / n_g ** 2.0))

        # Inverse variance
        df_sel = df_pij[df_pij.i != df_pij.j]
        df_feat.loc[0, "Fcm_inv_var"] = np.sum(df_sel.pij / (df_sel.i - df_sel.j) ** 2.0)
        del df_sel

        # Correlation
        mu_marg = np.sum(df_pi.i * df_pi.pi)
        var_marg = np.sum((df_pi.i - mu_marg) ** 2.0 * df_pi.pi)

        if var_marg == 0.0:
            df_feat.loc[0, "Fcm_corr"] = 1.0
        else:
            df_feat.loc[0, "Fcm_corr"] = 1.0 / var_marg * (np.sum(df_pij.i * df_pij.j * df_pij.pij) - mu_marg ** 2.0)

        del mu_marg, var_marg

        # Autocorrelation
        df_feat.loc[0, "Fcm_auto_corr"] = np.sum(df_pij.i * df_pij.j * df_pij.pij)

        # Information correlation 1
        hxy = -np.sum(df_pij.pij * np.log2(df_pij.pij))
        hxy_1 = -np.sum(df_pij.pij * np.log2(df_pij.pi * df_pij.pj))
        hx = -np.sum(df_pi.pi * np.log2(df_pi.pi))
        if len(df_pij) == 1 or hx == 0.0:
            df_feat.loc[0, "Fcm_info_corr1"] = 1.0
        else:
            df_feat.loc[0, "Fcm_info_corr1"] = (hxy - hxy_1) / hx
        del hxy, hxy_1, hx

        # Information correlation 2 - Note: iteration over combinations of i and j
        hxy = - np.sum(df_pij.pij * np.log2(df_pij.pij))
        hxy_2 = - np.sum(
                        np.tile(df_pi.pi, len(df_pj)) * np.repeat(df_pj.pj, len(df_pi)) * \
                        np.log2(np.tile(df_pi.pi, len(df_pj)) * np.repeat(df_pj.pj, len(df_pi)))
                        )

        if hxy_2 < hxy:
            df_feat.loc[0, "Fcm_info_corr2"] = 0
        else:
            df_feat.loc[0, "Fcm_info_corr2"] = np.sqrt(1 - np.exp(-2.0 * (hxy_2 - hxy)))
        del hxy, hxy_2

        # Cluster tendency
        mu = np.sum(df_pi.i * df_pi.pi)
        df_feat.loc[0, "Fcm_clust_tend"] = np.sum((df_pij.i + df_pij.j - 2 * mu) ** 2.0 * df_pij.pij)
        del mu

        # Cluster shade
        mu = np.sum(df_pi.i * df_pi.pi)
        df_feat.loc[0, "Fcm_clust_shade"] = np.sum((df_pij.i + df_pij.j - 2 * mu) ** 3.0 * df_pij.pij)
        del mu

        # Cluster prominence
        mu = np.sum(df_pi.i * df_pi.pi)
        df_feat.loc[0, "Fcm_clust_prom"] = np.sum((df_pij.i + df_pij.j - 2 * mu) ** 4.0 * df_pij.pij)

        del df_pi, df_pj, df_pij, df_pimj, df_pipj, n_g

        # Update names
        # df_feat.columns += self._parse_names()

        return df_feat

    def _parse_names(self):
        """"
        Adds additional settings-related identifiers to each feature.
        Not used currently, as the use of different settings for the
        co-occurrence matrix is not supported.
        """
        parse_str = ""

        # Add distance
        parse_str += "_d" + str(np.round(self.distance, 1))

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

@deprecated(reason="Use the new and the faster method get_cm_features()")
def get_cm_features_deprecated(vol, distCorrection):
    """Calculates co-occurrence matrix features

    Note:
        Deprecated code. Calculates co-occurrence matrix features, but slowl.
        A newer and faster method is available : `get_cm_features()`
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
    glcm = {'Fcm_joint_max': [],
            'Fcm_joint_avg': [],
            'Fcm_joint_var': [],
            'Fcm_joint_entr': [],
            'Fcm_diff_avg': [],
            'Fcm_diff_var': [],
            'Fcm_diff_entr': [],
            'Fcm_sum_avg': [],
            'Fcm_sum_var': [],
            'Fcm_sum_entr': [],
            'Fcm_energy': [],
            'Fcm_contrast': [],
            'Fcm_dissimilarity': [],
            'Fcm_inv_diff': [],
            'Fcm_inv_diff_norm': [],
            'Fcm_inv_diff_mom': [],
            'Fcm_inv_diff_mom_norm': [],
            'Fcm_inv_var': [],
            'Fcm_corr': [],
            'Fcm_auto_corr': [],
            'Fcm_clust_tend': [],
            'Fcm_clust_shade': [],
            'Fcm_clust_prom': [],
            'Fcm_info_corr_1': [],
            'Fcm_info_corr_2': []}

    # GET THE GLCM MATRIX
    #  Correct definition, without any assumption
    vol = vol.copy()
    levels = np.arange(1, np.max(vol[~np.isnan(vol[:])]) + 100 * np.finfo(float).eps)

    if distCorrection is None:
        GLCM = getGLCMmatrix(vol, levels)
    else:
        GLCM = getGLCMmatrix(
            vol, levels, distCorrection)

    p_ij = GLCM / np.sum(GLCM[:])  # Normalization of GLCM
    p_i = np.sum(p_ij, axis=1, keepdims=True)
    p_j = np.sum(p_ij, axis=0, keepdims=True)
    p_iminusj = gclm_DiagProb(p_ij)
    p_iplusj = gclm_CrossDiagProb(p_ij)
    Ng = np.max(np.shape(GLCM))
    vectNg = np.arange(1, Ng + 100 * np.finfo(float).eps)
    colGrid, rowGrid = np.meshgrid(vectNg, vectNg)

    ###############################################
    ######           GLCM features           ######
    ###############################################
    # Joint maximum
    glcm['Fcm_joint_max'] = np.max(p_ij[:])

    # Joint average
    temp = rowGrid * p_ij
    u = np.sum(temp)
    glcm['Fcm_joint_avg'] = u

    # Joint variance
    temp = np.power(rowGrid - u, 2) * p_ij
    var = np.sum(temp)
    glcm['Fcm_joint_var'] = var

    # Joint entropy
    pPos = p_ij[p_ij > 0]  # Exclusing those with 0 probability
    temp = pPos * np.log2(pPos)
    glcm['Fcm_joint_entr'] = -np.sum(temp)

    # Difference average
    k = np.arange(0, Ng)
    u = np.matmul(k, p_iminusj)  # k * p_iminusj
    glcm['Fcm_diff_avg'] = u

    # Difference variance
    var = np.matmul(np.power(k - u, 2), p_iminusj)
    glcm['Fcm_diff_var'] = var

    # Difference entropy
    kPos = p_iminusj[p_iminusj > 0]
    glcm['Fcm_diff_entr'] = - np.matmul(kPos.transpose(), np.log2(kPos))

    # Sum average
    k = np.arange(2, Ng * 2 + 100 * np.finfo(float).eps)
    u = np.matmul(k, p_iplusj)
    glcm['Fcm_sum_avg'] = u

    # Sum variance
    var = np.matmul(np.power(k - u, 2), p_iplusj)
    glcm['Fcm_sum_var'] = var

    # Sum entropy
    kPos = p_iplusj[p_iplusj > 0]
    glcm['Fcm_sum_entr'] = - np.matmul(kPos.transpose(), np.log2(kPos))

    # Angular second moment
    temp = np.power(p_ij, 2)
    glcm['Fcm_energy'] = np.sum(temp)

    # Contrast
    temp = np.power(rowGrid - colGrid, 2) * p_ij
    glcm['Fcm_contrast'] = np.sum(temp)

    # Dissimilarity
    temp = np.abs(rowGrid - colGrid) * p_ij
    glcm['Fcm_dissimilarity'] = np.sum(temp)

    # Inverse difference
    temp = p_ij / (1 + np.abs(rowGrid - colGrid))
    glcm['Fcm_inv_diff'] = np.sum(temp)

    # Inverse difference normalised
    temp = p_ij / (1 + np.abs(rowGrid - colGrid) / Ng)
    glcm['Fcm_inv_diff_norm'] = np.sum(temp)

    # Inverse difference moment
    temp = p_ij / (1 + np.power(rowGrid - colGrid, 2))
    glcm['Fcm_inv_diff_mom'] = np.sum(temp)

    # Inverse difference moment normalised
    temp = p_ij / (1 + (np.power(rowGrid - colGrid, 2) / np.power(Ng, 2)))
    glcm['Fcm_inv_diff_mom_norm'] = np.sum(temp)

    # Inverse variance
    p = 0
    for i in range(0, Ng):
        for j in range(i + 1, Ng):
            p = p + p_ij[i, j] / ((i - j) ** 2)
    glcm['Fcm_inv_var'] = 2 * p

    # Correlation
    u_i = np.matmul(vectNg, p_i)
    u_j = np.matmul(vectNg, p_j.transpose())
    std_i = np.sqrt(np.matmul(np.power(vectNg - u_i, 2), p_i))
    std_j = np.sqrt(np.matmul(np.power(vectNg - u_j, 2), p_j.transpose()))
    temp = rowGrid * colGrid * p_ij
    glcm['Fcm_corr'] = ((1 / (std_i * std_j)) * (-u_i * u_j + np.sum(temp)))[0]

    # Autocorrelation
    glcm['Fcm_auto_corr'] = np.sum(temp)

    # Cluster tendency
    temp = np.power((rowGrid + colGrid - u_i - u_j), 2) * p_ij
    glcm['Fcm_clust_tend'] = np.sum(temp)

    # Cluster shade
    temp = np.power((rowGrid + colGrid - u_i - u_j), 3) * p_ij
    glcm['Fcm_clust_shade'] = np.sum(temp)

    # Cluster prominence
    temp = np.power((rowGrid + colGrid - u_i - u_j), 4) * p_ij
    glcm['Fcm_clust_prom'] = np.sum(temp)

    # First measure of information correlation
    pPos = p_ij[p_ij > 0]
    temp = pPos * np.log2(pPos)
    HXY = -np.sum(temp)
    pPos = p_i[p_i > 0]
    temp = pPos * np.log2(pPos)
    HX = -np.sum(temp)
    p_i_temp = np.matlib.repmat(p_i, 1, Ng)
    p_j_temp = np.matlib.repmat(p_j, Ng, 1)
    p_temp = p_i_temp * p_j_temp
    pPos = p_ij[p_temp > 0]
    pPos_temp = p_temp[p_temp > 0]
    temp = pPos * np.log2(pPos_temp)
    HXY1 = -np.sum(temp)
    glcm['Fcm_info_corr_1'] = (HXY - HXY1) / HX

    # Second measure of information correlation
    temp = pPos_temp * np.log2(pPos_temp)
    HXY2 = -np.sum(temp)
    if HXY > HXY2:
        glcm['Fcm_info_corr_2'] = 0
    else:
        glcm['Fcm_info_corr_2'] = np.sqrt(1 - np.exp(-2 * (HXY2 - HXY)))

    return glcm
