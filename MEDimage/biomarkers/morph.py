#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict

import numpy as np
import scipy.spatial as sc

from ..biomarkers.get_geary_c import get_geary_c
from ..biomarkers.get_mesh_area import get_mesh_area
from ..biomarkers.get_mesh_volume import get_mesh_volume
from ..biomarkers.get_moran_i import get_moran_i
from ..biomarkers.getOrientedBoundBox import minOrientedBoundBox
from ..biomarkers.min_vol_ellipse import min_vol_ellipse as minv
from ..biomarkers.utils import (get_area_dens_approx, get_axis_lengths, getCOM,
                                getMesh)


def vol(vol, mask_int, mask_morph, res):
    """Computes morphological volume feature

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the morphological volume feature

    """
    vol = vol.copy()
    # PADDING THE VOLUME WITH A LAYER OF NaNs
    # (reduce mesh computation errors of associated mask)
    vol = np.pad(vol, pad_width=1, mode="constant", constant_values=np.NaN)
    # PADDING THE MASKS WITH A LAYER OF 0's
    # (reduce mesh computation errors of associated mask)
    mask_int = mask_int.copy()
    mask_int = np.pad(mask_int, pad_width=1, mode="constant", constant_values=0.0)
    mask_morph = mask_morph.copy()
    mask_morph = np.pad(mask_morph, pad_width=1, mode="constant", constant_values=0.0)

    # GETTING IMPORTANT VARIABLES
    # XYZ refers to [Xc,Yc,Zc] in ref. [1].
    _, faces, vertices = getMesh(mask_morph, res)

    return get_mesh_volume(faces, vertices)
    

def extract_all(vol, 
                    mask_int, 
                    mask_morph, 
                    res, 
                    intensity=None, 
                    compute_moran_i=False, 
                    compute_geary_c=False) -> Dict:
    """Compute Morphological Features.
    
    Note:
        Moran's Index and Geary's C measure takes so much computation time. Please
        use `compute_moran_i` `compute_geary_c` carefully.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).
        intensity (str, optional): If 'arbitrary', some feature will not be computed.
            If 'definite' or None, all feature will be computed. If 'filter', most features
            will not be computed, except some. The 'filter' option encompasses
            'arbitrary', and is even more stringent. Please see below.
        compute_moran_i (bool, optional): True to compute Moran's Index.
        compute_geary_c (bool, optional): True to compute Geary's C measure.

    Raises:
        ValueError: `intensity` mus be either "arbitrary", "definite", "filter" or None.

    REFERENCES:
        [1] <https://arxiv.org/abs/1612.07003>

    """
    morph = {'Fmorph_vol': [],
             'Fmorph_approx_vol': [],
             'Fmorph_area': [],
             'Fmorph_av': [],
             'Fmorph_comp_1': [],
             'Fmorph_comp_2': [],
             'Fmorph_sph_dispr': [],
             'Fmorph_sphericity': [],
             'Fmorph_asphericity': [],
             'Fmorph_com': [],
             'Fmorph_diam': [],
             'Fmorph_pca_major': [],
             'Fmorph_pca_minor': [],
             'Fmorph_pca_least': [],
             'Fmorph_pca_elongation': [],
             'Fmorph_pca_flatness': [],  # until here
             'Fmorph_v_dens_aabb': [],
             'Fmorph_a_dens_aabb': [],
             'Fmorph_v_dens_ombb': [],
             'Fmorph_a_dens_ombb': [],
             'Fmorph_v_dens_aee': [],
             'Fmorph_a_dens_aee': [],
             'Fmorph_v_dens_mvee': [],
             'Fmorph_a_dens_mvee': [],
             'Fmorph_v_dens_conv_hull': [],
             'Fmorph_a_dens_conv_hull': [],
             'Fmorph_integ_int': [],
             'Fmorph_moran_i': [],
             'Fmorph_geary_c': []}

    # INTIALIZATION
    if intensity == 'definite' or intensity is None:
        definite = True
        im_filter = False
    elif intensity == 'arbitrary':
        definite = False
        im_filter = False
    elif intensity == 'filter':
        definite = False
        im_filter = True
    else:
        raise ValueError('Fifth argument must either be "arbitrary" or \
                         "definite" or "filter"')

    # PADDING THE VOLUME WITH A LAYER OF NaNs
    # (reduce mesh computation errors of associated mask)
    vol = vol.copy()
    vol = np.pad(vol, pad_width=1, mode="constant", constant_values=np.NaN)
    # PADDING THE MASKS WITH A LAYER OF 0's
    # (reduce mesh computation errors of associated mask)
    mask_int = mask_int.copy()
    mask_int = np.pad(mask_int, pad_width=1, mode="constant", constant_values=0.0)
    mask_morph = mask_morph.copy()
    mask_morph = np.pad(mask_morph, pad_width=1, mode="constant", constant_values=0.0)

    # GETTING IMPORTANT VARIABLES
    xgl_int = np.reshape(vol, np.size(vol), order='F')[np.where(
        np.reshape(mask_int, np.size(mask_int), order='F') == 1)[0]].copy()
    Xgl_morph = np.reshape(vol, np.size(vol), order='F')[np.where(
        np.reshape(mask_morph, np.size(mask_morph), order='F') == 1)[0]].copy()
    # XYZ refers to [Xc,Yc,Zc] in ref. [1].
    xyz_int, _, _ = getMesh(mask_int, res)
    # XYZ refers to [Xc,Yc,Zc] in ref. [1].
    xyz_morph, faces, vertices = getMesh(mask_morph, res)
    # [X,Y,Z] points of the convex hull.
    # convHull Matlab is convHull.simplices
    convHull = sc.ConvexHull(vertices)

    # STARTING COMPUTATION

    if im_filter is not True:
        # In mm^3
        volume = get_mesh_volume(faces, vertices)
        morph['Fmorph_vol'] = volume  # Volume

        # Approximate Volume
        morph['Fmorph_approx_vol'] = np.sum(mask_morph[:]) * np.prod(res)

        # Surface area
        # In mm^2
        area = get_mesh_area(faces, vertices)
        morph['Fmorph_area'] = area

        # Surface to volume ratio
        morph['Fmorph_av'] = area/volume

        # Compactness 1
        morph['Fmorph_comp_1'] = volume/((np.pi**(1/2))*(area**(3/2)))

        # Compactness 2
        morph['Fmorph_comp_2'] = 36*np.pi*(volume**2)/(area**3)

        # Spherical disproportion
        morph['Fmorph_sph_dispr'] = area/(36*np.pi*volume**2)**(1/3)

        # Sphericity
        morph['Fmorph_sphericity'] = ((36*np.pi*volume**2)**(1/3))/area

        # Asphericity
        morph['Fmorph_asphericity'] = ((area**3) / (36*np.pi*volume**2))**(1/3) - 1

        # Centre of mass shift
        morph['Fmorph_com'] = getCOM(xgl_int, Xgl_morph, xyz_int, xyz_morph)

        # Maximum 3D diameter
        morph['Fmorph_diam'] = np.max(sc.distance.pdist(convHull.points[convHull.vertices]))

        # Major axis length
        [major, minor, least] = get_axis_lengths(xyz_morph)
        morph['Fmorph_pca_major'] = 4*np.sqrt(major)

        # Minor axis length
        morph['Fmorph_pca_minor'] = 4*np.sqrt(minor)

        # Least axis length
        morph['Fmorph_pca_least'] = 4*np.sqrt(least)

        # Elongation
        morph['Fmorph_pca_elongation'] = np.sqrt(minor / major)

        # Flatness
        morph['Fmorph_pca_flatness'] = np.sqrt(least / major)

        # Volume density - axis-aligned bounding box
        xc_aabb = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
        yc_aabb = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
        zc_aabb = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
        v_aabb = xc_aabb * yc_aabb * zc_aabb
        morph['Fmorph_v_dens_aabb'] = volume / v_aabb

        # Area density - axis-aligned bounding box
        a_aabb = 2*xc_aabb*yc_aabb + 2*xc_aabb*zc_aabb + 2*yc_aabb*zc_aabb
        morph['Fmorph_a_dens_aabb'] = area / a_aabb

        # Volume density - oriented minimum bounding box
        # Implementation of Chan and Tan's algorithm (C.K. Chan, S.T. Tan.
        # Determination of the minimum bounding box of an
        # arbitrary solid: an iterative approach.
        # Comp Struc 79 (2001) 1433-1449
        bound_box_dims = minOrientedBoundBox(vertices)
        vol_bb = np.prod(bound_box_dims)
        morph['Fmorph_v_dens_ombb'] = volume / vol_bb

        # Area density - oriented minimum bounding box
        a_ombb = 2 * (bound_box_dims[0]*bound_box_dims[1] +
                     bound_box_dims[0]*bound_box_dims[2] +
                     bound_box_dims[1]*bound_box_dims[2])
        morph['Fmorph_a_dens_ombb'] = area / a_ombb

        # Volume density - approximate enclosing ellipsoid
        a = 2*np.sqrt(major)
        b = 2*np.sqrt(minor)
        c = 2*np.sqrt(least)
        v_aee = (4*np.pi*a*b*c)/3
        morph['Fmorph_v_dens_aee'] = volume / v_aee

        # Area density - approximate enclosing ellipsoid
        a_aee = get_area_dens_approx(a, b, c, 20)
        morph['Fmorph_a_dens_aee'] = area / a_aee

        # Volume density - minimum volume enclosing ellipsoid
        # (Rotate the volume first??)
        # Copyright (c) 2009, Nima Moshtagh
        # http://www.mathworks.com/matlabcentral/fileexchange/
        # 9542-minimum-volume-enclosing-ellipsoid
        # Subsequent singular value decomposition of matrix A and and
        # taking the inverse of the square root of the diagonal of the
        # sigma matrix will produce respective semi-axis lengths.
        # Subsequent singular value decomposition of matrix A and
        # taking the inverse of the square root of the diagonal of the
        # sigma matrix will produce respective semi-axis lengths.
        P = np.stack((convHull.points[convHull.simplices[:, 0], 0],
                      convHull.points[convHull.simplices[:, 1], 1],
                      convHull.points[convHull.simplices[:, 2], 2]), axis=1)
        A, _ = minv.min_vol_ellipse(np.transpose(P), 0.01)
        # New semi-axis lengths
        _, Q, _ = np.linalg.svd(A)
        a = 1/np.sqrt(Q[2])
        b = 1/np.sqrt(Q[1])
        c = 1/np.sqrt(Q[0])
        v_mvee = (4*np.pi*a*b*c)/3
        morph['Fmorph_v_dens_mvee'] = volume/v_mvee

        # Area density - minimum volume enclosing ellipsoid
        # Using a new set of (a,b,c), see Volume density - minimum
        # volume enclosing ellipsoid
        a_mvee = get_area_dens_approx(a, b, c, 20)
        morph['Fmorph_a_dens_mvee'] = area / a_mvee

        # Volume density - convex hull
        v_convex = convHull.volume
        morph['Fmorph_v_dens_conv_hull'] = volume / v_convex

        # Area density - convex hull
        a_convex = convHull.area
        morph['Fmorph_a_dens_conv_hull'] = area / a_convex

    # Integrated intensity
    if definite:
        morph['Fmorph_integ_int'] = np.mean(xgl_int) * volume

    # Moran's I index
    if compute_moran_i:
        vol_Mor = vol.copy()
        vol_Mor[mask_int == 0] = np.NaN
        morph['Fmorph_moran_i'] = get_moran_i(vol_Mor,res)

    # Geary's C measure
    if compute_geary_c:
        morph['Fmorph_geary_c'] = get_geary_c(vol_Mor,res)

    return morph
