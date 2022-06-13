#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
- Creation: June 2018
"""

import numpy as np
import scipy.ndimage as sc
import skimage.measure as skim


def getGLDZMmatrix(roi_only_int, mask, levels) -> np.ndarray:
    """Computes GLDZM matrix.

    Args:
        roi_only_int (ndarray): 3D volume, isotropically resampled, 
            quantized (e.g. n_g = 32, levels = [1, ..., n_g]), 
            with NaNs outside the region of interest.
        mask (ndarray): Morphological ROI mask.
        levels (ndarray or List): Vector containing the quantized gray-levels 
            in the tumor region (or reconstruction levels of quantization).

    Returns:
        ndarray: Grey level distance zone Matrix.

    Todo:
        *levels: should be removed at some point, no longer needed if we always
            quantize our volume such that `levels = 1,2,3,4,...,max(quantized Volume)`. 
            So simply calculate `levels = 1:max(roi_only(~isnan(roi_only(:))))`
            directly in this function
    
    """
    
    roi_only_int = roi_only_int.copy()
    levels = levels.copy().astype("int")
    morph_voxel_grid = mask.copy().astype(np.uint8)
    
    # COMPUTATION OF DISTANCE MAP
    morph_voxel_grid = np.pad(morph_voxel_grid, 
                            [1,1],
                            'constant', 
                            constant_values=0)
    
    # Computing the smallest ROI edge possible. 
    # Distances are determined in 3D
    binary_struct = sc.generate_binary_structure(rank=3, connectivity=1)
    perimeter = morph_voxel_grid - sc.binary_erosion(morph_voxel_grid, structure=binary_struct)
    perimeter = perimeter[1:-1,1:-1,1:-1] # Removing the padding.
    morph_voxel_grid = morph_voxel_grid[1:-1,1:-1,1:-1] # Removing the padding
    
    # +1 according to the definition of the IBSI    
    dist_map = sc.distance_transform_cdt(np.logical_not(perimeter), metric='cityblock') + 1  
    
    # INITIALIZATION
    # Since levels is always defined as 1,2,3,4,...,max(quantized Volume)
    n_g = np.size(levels)
    level_temp = np.max(levels) + 1
    roi_only_int[np.isnan(roi_only_int)] = level_temp
    # Since the ROI morph always encompasses ROI int,
    # using the mask as defined from ROI morph does not matter since 
    # we want to find the maximal possible distance.
    dist_init = np.max(dist_map[morph_voxel_grid == 1]) 
    GLDZM = np.zeros((n_g,dist_init))
        
    # COMPUTATION OF GLDZM
    temp = roi_only_int.copy().astype('int')
    for i in range(1,n_g+1):
        temp[roi_only_int!=levels[i-1]] = 0
        temp[roi_only_int==levels[i-1]] = 1
        conn_objects, n_zone = skim.label(temp,return_num = True)
        for j in range(1,n_zone+1): 
            col = np.min(dist_map[conn_objects==j]).astype("int")
            GLDZM[i-1,col-1] = GLDZM[i-1,col-1] + 1
              
    # REMOVE UNECESSARY COLUMNS
    stop = np.nonzero(np.sum(GLDZM,0))[0][-1]
    GLDZM = np.delete(GLDZM, range(stop+1, np.shape(GLDZM)[1]), 1)
    
    return  GLDZM
