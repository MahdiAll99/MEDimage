# -*- coding: utf-8 -*-
"""
- Creation: June 2018

"""

import numpy as np
import skimage.measure as skim
import scipy.ndimage as sc
  
def getGLDZMmatrix(ROIOnlyInt,mask,levels):
    """Compute GLDZMmatrix.
    -------------------------------------------------------------------------
    getGLDZMmatrix(ROIOnlyInt,mask,levels)
    -------------------------------------------------------------------------
    AUTHOR(S): 
     - Martin Vallieres <mart.vallieres@gmail.com>
     - Jorge Barrios Ginart <numeroj@gmail.com>
     - Olivier Morin <Olivier.Morin@ucsf.edu>
    -------------------------------------------------------------------------
    HISTORY:
    - Creation: June 2018
    -------------------------------------------------------------------------
    DISCLAIMER:
    "I'm not a programmer, I'm just a scientist doing stuff!"
    -------------------------------------------------------------------------
    STATEMENT:
    This file is part of <https://github.com/mvallieres/radiomics-develop/>, 
    a private repository dedicated to the development of programming code for
    new radiomics applications.
     --> Copyright (C) 2017  Martin Vallieres
        All rights reserved.
    This file is written on the basis of a scientific collaboration for the 
    "radiomics-develop" team.
    
    By using this file, all members of the team acknowledge that it is to be 
    kept private until public release. Other scientists willing to join the 
    "radiomics-develop" team is however highly encouraged. Please contact 
    Martin Vallieres for this matter.
    -------------------------------------------------------------------------
    - vol: 3D volume, isotropically resampled, quantized, with NaNs outside 
      the region of interest
    - levels: should be removed at some point, no longer needed if we always
      quantize our volume such that levels = 1,2,3,4,...,max(quantized Volume). 
      So simply calculate levels = 1:max(ROIOnly(~isnan(ROIOnly(:))))
      directly in this function
    -------------------------------------------------------------------------
    """
    
    ROIOnlyInt = ROIOnlyInt.copy()
    levels = levels.copy().astype("int")
    morph_voxel_grid = mask.copy().astype(np.uint8)
    
    # COMPUTATION OF DISTANCE MAP
    morph_voxel_grid = np.pad(morph_voxel_grid, [1,1],
                              'constant', constant_values=0)
    
    # Computing the smallest ROI edge possible. 
    # Distances are determined in 3D
    binary_struct = sc.generate_binary_structure(rank=3, connectivity=1)
    perimeter = morph_voxel_grid - sc.binary_erosion(
            morph_voxel_grid,structure=binary_struct)
    perimeter = perimeter[1:-1,1:-1,1:-1] # Removing the padding.
    morph_voxel_grid = morph_voxel_grid[1:-1,1:-1,1:-1] # Removing the padding
    # +1 according to the definition of the IBSI    
    dist_map = sc.distance_transform_cdt(np.logical_not(perimeter),
                                         metric='cityblock') + 1  
    
    # INITIALIZATION
    # Since levels is always defined as 1,2,3,4,...,max(quantized Volume)
    Ng = np.size(levels)
    levelTemp = np.max(levels) + 1
    ROIOnlyInt[np.isnan(ROIOnlyInt)] = levelTemp
    # Since the ROI morph always encompasses ROI int,
    # using the mask as defined from ROI morph does not matter since 
    # we want to find the maximal possible distance.
    distInit = np.max(dist_map[morph_voxel_grid == 1]) 
    GLDZM = np.zeros((Ng,distInit))
        
    # COMPUTATION OF GLDZM
    temp = ROIOnlyInt.copy().astype('int')
    for i in range(1,Ng+1):
        temp[ROIOnlyInt!=levels[i-1]] = 0
        temp[ROIOnlyInt==levels[i-1]] = 1
        connObjects, nZone = skim.label(temp,return_num = True)
        for j in range(1,nZone+1): 
            col = np.min(dist_map[connObjects==j]).astype("int")
            GLDZM[i-1,col-1] = GLDZM[i-1,col-1] + 1
              
    # REMOVE UNECESSARY COLUMNS
    stop = np.nonzero(np.sum(GLDZM,0))[0][-1]
    GLDZM = np.delete(GLDZM, range(stop+1, np.shape(GLDZM)[1]), 1)
    
    return  GLDZM