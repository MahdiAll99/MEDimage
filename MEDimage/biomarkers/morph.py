#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict

import numpy as np
import scipy.spatial as sc

from ..biomarkers.get_geary_c import get_geary_c
from ..biomarkers.get_mesh_area import get_mesh_area
from ..biomarkers.get_mesh_volume import get_mesh_volume
from ..biomarkers.get_moran_i import get_moran_i
from ..biomarkers.get_oriented_bound_box import min_oriented_bound_box
from ..biomarkers.min_vol_ellipse import min_vol_ellipse as minv
from ..biomarkers.utils import (get_area_dens_approx, get_axis_lengths,
                                get_com, get_mesh)


def padding(vol: np.ndarray, 
            mask_int: np.ndarray, 
            mask_morph: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Padding the volume and masks.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.

    Returns:
        tuple of 3 ndarray: Volume and masks after padding.
    """

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

    return vol, mask_int, mask_morph

def get_variables(vol: np.ndarray, 
                  mask_int: np.ndarray, 
                  mask_morph: np.ndarray,
                  res: np.ndarray) -> tuple[np.ndarray, 
                                            np.ndarray, 
                                            np.ndarray, 
                                            np.ndarray, 
                                            np.ndarray, 
                                            np.ndarray, 
                                            np.ndarray]:
    """Compute variables usefull to calculate morphological features.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        tuple of 7 ndarray: Variables usefull to calculate morphological features.
    """
    # GETTING IMPORTANT VARIABLES
    xgl_int = np.reshape(vol, np.size(vol), order='F')[np.where(
        np.reshape(mask_int, np.size(mask_int), order='F') == 1)[0]].copy()
    Xgl_morph = np.reshape(vol, np.size(vol), order='F')[np.where(
        np.reshape(mask_morph, np.size(mask_morph), order='F') == 1)[0]].copy()
    # XYZ refers to [Xc,Yc,Zc] in ref. [1].
    xyz_int, _, _ = get_mesh(mask_int, res)
    # XYZ refers to [Xc,Yc,Zc] in ref. [1].
    xyz_morph, faces, vertices = get_mesh(mask_morph, res)
    # [X,Y,Z] points of the convex hull.
    # conv_hull Matlab is conv_hull.simplices
    conv_hull = sc.ConvexHull(vertices)

    return xgl_int, Xgl_morph, xyz_int, xyz_morph, faces, vertices, conv_hull

def extract_all(vol: np.ndarray, 
                mask_int: np.ndarray, 
                mask_morph: np.ndarray, 
                res: np.ndarray, 
                compute_moran_i: bool=False, 
                compute_geary_c: bool=False) -> Dict:
    """Compute Morphological Features.
    This features refer to Morphological family in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)
    
    Note:
        Moran's Index and Geary's C measure takes so much computation time. Please
        use `compute_moran_i` `compute_geary_c` carefully.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).
        compute_moran_i (bool, optional): True to compute Moran's Index.
        compute_geary_c (bool, optional): True to compute Geary's C measure.

    Raises:
        ValueError: `intensity` mus be either "arbitrary", "definite", "filter" or None.

    REFERENCES:
        [1] <https://arxiv.org/abs/1612.07003>
    """
    # Initialization of final structure (Dictionary) containing all features.
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
             'Fmorph_geary_c': []
            }

    #Initialization
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    xgl_int, Xgl_morph, xyz_int, xyz_morph, faces, vertices, conv_hull = get_variables(vol, mask_int, mask_morph, res)

    # STARTING COMPUTATION
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
    morph['Fmorph_av'] = area / volume

    # Compactness 1
    morph['Fmorph_comp_1'] = volume / ((np.pi**(1/2))*(area**(3/2)))

    # Compactness 2
    morph['Fmorph_comp_2'] = 36*np.pi*(volume**2) / (area**3)

    # Spherical disproportion
    morph['Fmorph_sph_dispr'] = area / (36*np.pi*volume**2)**(1/3)

    # Sphericity
    morph['Fmorph_sphericity'] = ((36*np.pi*volume**2)**(1/3)) / area

    # Asphericity
    morph['Fmorph_asphericity'] = ((area**3) / (36*np.pi*volume**2))**(1/3) - 1

    # Centre of mass shift
    morph['Fmorph_com'] = get_com(xgl_int, Xgl_morph, xyz_int, xyz_morph)

    # Maximum 3D diameter
    morph['Fmorph_diam'] = np.max(sc.distance.pdist(conv_hull.points[conv_hull.vertices]))

    # Major axis length
    [major, minor, least] = get_axis_lengths(xyz_morph)
    morph['Fmorph_pca_major'] = 4 * np.sqrt(major)

    # Minor axis length
    morph['Fmorph_pca_minor'] = 4 * np.sqrt(minor)

    # Least axis length
    morph['Fmorph_pca_least'] = 4 * np.sqrt(least)

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
    bound_box_dims = min_oriented_bound_box(vertices)
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
    v_aee = (4*np.pi*a*b*c) / 3
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
    p = np.stack((conv_hull.points[conv_hull.simplices[:, 0], 0],
                    conv_hull.points[conv_hull.simplices[:, 1], 1],
                    conv_hull.points[conv_hull.simplices[:, 2], 2]), axis=1)
    A, _ = minv.min_vol_ellipse(np.transpose(p), 0.01)
    # New semi-axis lengths
    _, Q, _ = np.linalg.svd(A)
    a = 1/np.sqrt(Q[2])
    b = 1/np.sqrt(Q[1])
    c = 1/np.sqrt(Q[0])
    v_mvee = (4*np.pi*a*b*c)/3
    morph['Fmorph_v_dens_mvee'] = volume / v_mvee

    # Area density - minimum volume enclosing ellipsoid
    # Using a new set of (a,b,c), see Volume density - minimum
    # volume enclosing ellipsoid
    a_mvee = get_area_dens_approx(a, b, c, 20)
    morph['Fmorph_a_dens_mvee'] = area / a_mvee

    # Volume density - convex hull
    v_convex = conv_hull.volume
    morph['Fmorph_v_dens_conv_hull'] = volume / v_convex

    # Area density - convex hull
    a_convex = conv_hull.area
    morph['Fmorph_a_dens_conv_hull'] = area / a_convex

    # Integrated intensity
    morph['Fmorph_integ_int'] = np.mean(xgl_int) * volume

    # Moran's I index
    if compute_moran_i:
        vol_Mor = vol.copy()
        vol_Mor[mask_int == 0] = np.NaN
        morph['Fmorph_moran_i'] = get_moran_i(vol_Mor, res)

    # Geary's C measure
    if compute_geary_c:
        morph['Fmorph_geary_c'] = get_geary_c(vol_Mor, res)

    return morph

def vol(vol: np.ndarray, 
        mask_int: np.ndarray, 
        mask_morph: np.ndarray, 
        res: np.ndarray) -> float:
    """Computes morphological volume feature.
    This feature refers to "Fmorph_vol" (id = RNUO) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the morphological volume feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, faces, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    volume = get_mesh_volume(faces, vertices)

    return volume  # Morphological volume feature

def approx_vol(vol: np.ndarray, 
               mask_int: np.ndarray, 
               mask_morph: np.ndarray, 
               res: np.ndarray) -> float:
    """Computes morphological approximate volume feature.
    This feature refers to "Fmorph_approx_vol" (id = YEKZ) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the morphological approximate volume feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    volume_appro = np.sum(mask_morph[:]) * np.prod(res)

    return volume_appro  # Morphological approximate volume feature

def area(vol: np.ndarray, 
         mask_int: np.ndarray, 
         mask_morph: np.ndarray, 
         res: np.ndarray) -> float:
    """Computes Surface area feature.
    This feature refers to "Fmorph_area" (id = COJJK) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the surface area feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, faces, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    area = get_mesh_area(faces, vertices)

    return area  # Surface area

def av(vol: np.ndarray, 
       mask_int: np.ndarray, 
       mask_morph: np.ndarray, 
       res: np.ndarray) -> float:
    """Computes Surface to volume ratio feature.
    This feature refers to "Fmorph_av" (id = 2PR5) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Surface to volume ratio feature.
    """
    volume = volume(vol, mask_int, mask_morph, res)
    area = area(vol, mask_int, mask_morph, res)
    ratio = area / volume

    return ratio  # Surface to volume ratio
    
def comp_1(vol: np.ndarray, 
           mask_int: np.ndarray, 
           mask_morph: np.ndarray, 
           res: np.ndarray) -> float:
    """Computes Compactness 1 feature.
    This feature refers to "Fmorph_comp_1" (id = SKGS) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Compactness 1 feature.
    """
    volume = volume(vol, mask_int, mask_morph, res)
    area = area(vol, mask_int, mask_morph, res)
    comp_1 = volume / ((np.pi**(1/2))*(area**(3/2)))

    return comp_1  # Compactness 1
    
def comp_2(vol: np.ndarray, 
           mask_int: np.ndarray, 
           mask_morph: np.ndarray, 
           res: np.ndarray) -> float:
    """Computes Compactness 2 feature.
    This feature refers to "Fmorph_comp_2" (id = BQWJ) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Compactness 2 feature.
    """
    volume = volume(vol, mask_int, mask_morph, res)
    area = area(vol, mask_int, mask_morph, res)
    comp_2 = 36*np.pi*(volume**2) / (area**3)

    return comp_2  # Compactness 2

def sph_dispr(vol: np.ndarray, 
              mask_int: np.ndarray, 
              mask_morph: np.ndarray, 
              res: np.ndarray) -> float:
    """Computes Spherical disproportion feature.
    This feature refers to "Fmorph_sph_dispr" (id = KRCK) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Spherical disproportion feature.
    """
    volume = volume(vol, mask_int, mask_morph, res)
    area = area(vol, mask_int, mask_morph, res)
    sph_dispr = area / (36*np.pi*volume**2)**(1/3)

    return sph_dispr  # Spherical disproportion

def sphericity(vol: np.ndarray, 
               mask_int: np.ndarray, 
               mask_morph: np.ndarray, 
               res: np.ndarray) -> float:
    """Computes Sphericity feature.
    This feature refers to "Fmorph_sphericity" (id = QCFX) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Sphericity feature.
    """
    volume = volume(vol, mask_int, mask_morph, res)
    area = area(vol, mask_int, mask_morph, res)
    sphericity = ((36*np.pi*volume**2)**(1/3)) / area

    return sphericity  # Sphericity

def asphericity(vol: np.ndarray, 
                mask_int: np.ndarray, 
                mask_morph: np.ndarray, 
                res: np.ndarray) -> float:
    """Computes Asphericity feature.
    This feature refers to "Fmorph_asphericity" (id =  25C) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Asphericity feature.
    """
    volume = volume(vol, mask_int, mask_morph, res)
    area = area(vol, mask_int, mask_morph, res)
    asphericity = ((area**3) / (36*np.pi*volume**2))**(1/3) - 1

    return asphericity  # Asphericity

def com(vol: np.ndarray, 
        mask_int: np.ndarray, 
        mask_morph: np.ndarray, 
        res: np.ndarray) -> float:
    """Computes Centre of mass shift feature.
    This feature refers to "Fmorph_com" (id =  KLM) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Centre of mass shift feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    xgl_int, Xgl_morph, xyz_int, xyz_morph, _, _, _ = get_variables(vol, mask_int, mask_morph, res)
    com = get_com(xgl_int, Xgl_morph, xyz_int, xyz_morph)

    return com  # Centre of mass shift

def diam(vol: np.ndarray, 
         mask_int: np.ndarray, 
         mask_morph: np.ndarray, 
         res: np.ndarray) -> float:
    """Computes Maximum 3D diameter feature.
    This feature refers to "Fmorph_diam" (id = L0JK) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Maximum 3D diameter feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, _, _, conv_hull = get_variables(vol, mask_int, mask_morph, res)
    diam = np.max(sc.distance.pdist(conv_hull.points[conv_hull.vertices]))

    return diam  # Maximum 3D diameter

def pca_major(vol: np.ndarray, 
              mask_int: np.ndarray, 
              mask_morph: np.ndarray, 
              res: np.ndarray) -> float:
    """Computes Major axis length feature.
    This feature refers to "Fmorph_pca_major" (id = TDIC) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Major axis length feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, xyz_morph, _, _, _ = get_variables(vol, mask_int, mask_morph, res)
    [major, _, _] = get_axis_lengths(xyz_morph)
    pca_major = 4 * np.sqrt(major)

    return pca_major  # Major axis length

def pca_minor(vol: np.ndarray, 
              mask_int: np.ndarray, 
              mask_morph: np.ndarray, 
              res: np.ndarray) -> float:
    """Computes Minor axis length feature.
    This feature refers to "Fmorph_pca_minor" (id = P9VJ) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Minor axis length feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, xyz_morph, _, _, _ = get_variables(vol, mask_int, mask_morph, res)
    [_, minor, _] = get_axis_lengths(xyz_morph)
    pca_minor = 4 * np.sqrt(minor)

    return pca_minor  # Minor axis length

def pca_least(vol: np.ndarray, 
              mask_int: np.ndarray, 
              mask_morph: np.ndarray, 
              res: np.ndarray) -> float:
    """Computes Least axis length feature.
    This feature refers to "Fmorph_pca_least" (id = 7J51) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Least axis length feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, xyz_morph, _, _, _ = get_variables(vol, mask_int, mask_morph, res)
    [_, _, least] = get_axis_lengths(xyz_morph)
    pca_least = 4 * np.sqrt(least)

    return pca_least  # Least axis length

def pca_elongation(vol: np.ndarray, 
                   mask_int: np.ndarray, 
                   mask_morph: np.ndarray, 
                   res: np.ndarray) -> float:
    """Computes Elongation feature.
    This feature refers to "Fmorph_pca_elongation" (id = Q3CK) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Elongation feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, xyz_morph, _, _, _ = get_variables(vol, mask_int, mask_morph, res)
    [major, minor, _] = get_axis_lengths(xyz_morph)
    pca_elongation = np.sqrt(minor / major)

    return pca_elongation  # Elongation

def pca_flatness(vol: np.ndarray, 
                 mask_int: np.ndarray, 
                 mask_morph: np.ndarray, 
                 res: np.ndarray) -> float:
    """Computes Flatness feature.
    This feature refers to "Fmorph_pca_flatness" (id = N17B) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Flatness feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, xyz_morph, _, _, _ = get_variables(vol, mask_int, mask_morph, res)
    [major, _, least] = get_axis_lengths(xyz_morph)
    pca_flatness = np.sqrt(least / major)

    return pca_flatness  # Flatness

def v_dens_aabb(vol: np.ndarray, 
                mask_int: np.ndarray, 
                mask_morph: np.ndarray, 
                res: np.ndarray) -> float:
    """Computes Volume density - axis-aligned bounding box feature.
    This feature refers to "Fmorph_v_dens_aabb" (id = PBX1) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Volume density - axis-aligned bounding box feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, _, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    volume = volume(vol, mask_int, mask_morph, res)
    xc_aabb = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
    yc_aabb = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
    zc_aabb = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
    v_aabb = xc_aabb * yc_aabb * zc_aabb
    v_dens_aabb = volume / v_aabb

    return v_dens_aabb  # Volume density - axis-aligned bounding box

def a_dens_aabb(vol: np.ndarray, 
                mask_int: np.ndarray, 
                mask_morph: np.ndarray, 
                res: np.ndarray) -> float:
    """Computes Area density - axis-aligned bounding box feature.
    This feature refers to "Fmorph_a_dens_aabb" (id = R59B) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Area density - axis-aligned bounding box feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, _, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    area = area(vol, mask_int, mask_morph, res)
    xc_aabb = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
    yc_aabb = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
    zc_aabb = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
    a_aabb = 2*xc_aabb*yc_aabb + 2*xc_aabb*zc_aabb + 2*yc_aabb*zc_aabb
    a_dens_aabb = area / a_aabb

    return a_dens_aabb  # Area density - axis-aligned bounding box

def v_dens_ombb(vol: np.ndarray, 
                mask_int: np.ndarray, 
                mask_morph: np.ndarray, 
                res: np.ndarray) -> float:
    """Computes Volume density - oriented minimum bounding box feature.
    Implementation of Chan and Tan's algorithm (C.K. Chan, S.T. Tan.
    Determination of the minimum bounding box of an
    arbitrary solid: an iterative approach.
    Comp Struc 79 (2001) 1433-1449
    This feature refers to "Fmorph_v_dens_ombb" (id = ZH1A) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Volume density - oriented minimum bounding box feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, _, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    volume = volume(vol, mask_int, mask_morph, res)
    bound_box_dims = min_oriented_bound_box(vertices)
    vol_bb = np.prod(bound_box_dims)
    v_dens_ombb = volume / vol_bb

    return v_dens_ombb  # Volume density - oriented minimum bounding box

def a_dens_ombb(vol: np.ndarray, 
                mask_int: np.ndarray, 
                mask_morph: np.ndarray, 
                res: np.ndarray) -> float:
    """Computes Area density - oriented minimum bounding box feature.
    Implementation of Chan and Tan's algorithm (C.K. Chan, S.T. Tan.
    Determination of the minimum bounding box of an
    arbitrary solid: an iterative approach.
    Comp Struc 79 (2001) 1433-1449
    This feature refers to "Fmorph_a_dens_ombb" (id = IQYR) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Area density - oriented minimum bounding box feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, _, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    area = area(vol, mask_int, mask_morph, res)

    bound_box_dims = min_oriented_bound_box(vertices)
    a_ombb = 2 * (bound_box_dims[0] * bound_box_dims[1] 
                + bound_box_dims[0] * bound_box_dims[2]
                + bound_box_dims[1] * bound_box_dims[2])
    a_dens_ombb = area / a_ombb

    return a_dens_ombb  # Area density - oriented minimum bounding box

def v_dens_aee(vol: np.ndarray, 
               mask_int: np.ndarray, 
               mask_morph: np.ndarray, 
               res: np.ndarray) -> float:
    """Computes Volume density - approximate enclosing ellipsoid feature.
    This feature refers to "Fmorph_v_dens_aee" (id = 6BDE) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args: 
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Volume density - approximate enclosing ellipsoid feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, xyz_morph, _, _, _ = get_variables(vol, mask_int, mask_morph, res)
    volume = volume(vol, mask_int, mask_morph, res)
    [major, minor, least] = get_axis_lengths(xyz_morph)
    a = 2*np.sqrt(major)
    b = 2*np.sqrt(minor)
    c = 2*np.sqrt(least)
    v_aee = (4*np.pi*a*b*c) / 3
    v_dens_aee = volume / v_aee

    return v_dens_aee  # Volume density - approximate enclosing ellipsoid

def a_dens_aee(vol: np.ndarray, 
               mask_int: np.ndarray, 
               mask_morph: np.ndarray, 
               res: np.ndarray) -> float:
    """Computes Area density - approximate enclosing ellipsoid feature.
    This feature refers to "Fmorph_a_dens_aee" (id = RDD2) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Area density - approximate enclosing ellipsoid feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, xyz_morph, _, _, _ = get_variables(vol, mask_int, mask_morph, res)
    area = area(vol, mask_int, mask_morph, res)
    [major, minor, least] = get_axis_lengths(xyz_morph)
    a = 2*np.sqrt(major)
    b = 2*np.sqrt(minor)
    c = 2*np.sqrt(least)
    a_aee = get_area_dens_approx(a, b, c, 20)
    a_dens_aee = area / a_aee

    return a_dens_aee  # Area density - approximate enclosing ellipsoid

def v_dens_mvee(vol: np.ndarray, 
                mask_int: np.ndarray, 
                mask_morph: np.ndarray, 
                res: np.ndarray) -> float:
    """Computes Volume density - minimum volume enclosing ellipsoid feature.
    Copyright (c) 2009, Nima Moshtagh
    http://www.mathworks.com/matlabcentral/fileexchange/
    9542-minimum-volume-enclosing-ellipsoid
    Subsequent singular value decomposition of matrix A and and
    taking the inverse of the square root of the diagonal of the
    sigma matrix will produce respective semi-axis lengths.
    Subsequent singular value decomposition of matrix A and
    taking the inverse of the square root of the diagonal of the
    sigma matrix will produce respective semi-axis lengths.
    This feature refers to "Fmorph_v_dens_mvee" (id = SWZ1) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Volume density - minimum volume enclosing ellipsoid feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, _, _, conv_hull = get_variables(vol, mask_int, mask_morph, res)
    volume = volume(vol, mask_int, mask_morph, res)
    p = np.stack((conv_hull.points[conv_hull.simplices[:, 0], 0],
                    conv_hull.points[conv_hull.simplices[:, 1], 1],
                    conv_hull.points[conv_hull.simplices[:, 2], 2]), axis=1)
    A, _ = minv.min_vol_ellipse(np.transpose(p), 0.01)
    # New semi-axis lengths
    _, Q, _ = np.linalg.svd(A)
    a = 1/np.sqrt(Q[2])
    b = 1/np.sqrt(Q[1])
    c = 1/np.sqrt(Q[0])
    v_mvee = (4*np.pi*a*b*c) / 3
    v_dens_mvee = volume / v_mvee

    return v_dens_mvee  # Volume density - minimum volume enclosing ellipsoid

def a_dens_mvee(vol: np.ndarray, 
                mask_int: np.ndarray, 
                mask_morph: np.ndarray, 
                res: np.ndarray) -> float:
    """Computes Area density - minimum volume enclosing ellipsoid feature.
    Copyright (c) 2009, Nima Moshtagh
    http://www.mathworks.com/matlabcentral/fileexchange/
    9542-minimum-volume-enclosing-ellipsoid
    Subsequent singular value decomposition of matrix A and and
    taking the inverse of the square root of the diagonal of the
    sigma matrix will produce respective semi-axis lengths.
    Subsequent singular value decomposition of matrix A and
    taking the inverse of the square root of the diagonal of the
    sigma matrix will produce respective semi-axis lengths.
    This feature refers to "Fmorph_a_dens_mvee" (id = BRI8) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Area density - minimum volume enclosing ellipsoid feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, _, _, conv_hull = get_variables(vol, mask_int, mask_morph, res)
    area = area(vol, mask_int, mask_morph, res)
    p = np.stack((conv_hull.points[conv_hull.simplices[:, 0], 0],
                    conv_hull.points[conv_hull.simplices[:, 1], 1],
                    conv_hull.points[conv_hull.simplices[:, 2], 2]), axis=1)
    A, _ = minv.min_vol_ellipse(np.transpose(p), 0.01)
    # New semi-axis lengths
    _, Q, _ = np.linalg.svd(A)
    a = 1/np.sqrt(Q[2])
    b = 1/np.sqrt(Q[1])
    c = 1/np.sqrt(Q[0])
    a_mvee = get_area_dens_approx(a, b, c, 20)
    a_dens_mvee = area / a_mvee

    return a_dens_mvee  # Area density - minimum volume enclosing ellipsoid

def v_dens_conv_hull(vol: np.ndarray, 
                     mask_int: np.ndarray, 
                     mask_morph: np.ndarray, 
                     res: np.ndarray) -> float:
    """Computes Volume density - convex hull feature.
    This feature refers to "Fmorph_v_dens_conv_hull" (id = R3ER) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Volume density - convex hull feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, _, _, conv_hull = get_variables(vol, mask_int, mask_morph, res)
    volume = volume(vol, mask_int, mask_morph, res)
    v_convex = conv_hull.volume
    v_dens_conv_hull = volume / v_convex

    return v_dens_conv_hull  # Volume density - convex hull

def a_dens_conv_hull(vol: np.ndarray, 
                     mask_int: np.ndarray, 
                     mask_morph: np.ndarray, 
                     res: np.ndarray) -> float:
    """Computes Area density - convex hull feature.
    This feature refers to "Fmorph_a_dens_conv_hull" (id = 7T7F) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Area density - convex hull feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, _, _, conv_hull = get_variables(vol, mask_int, mask_morph, res)
    area = area(vol, mask_int, mask_morph, res)
    v_convex = conv_hull.area
    a_dens_conv_hull = area / v_convex

    return a_dens_conv_hull  # Area density - convex hull

def integ_int(vol: np.ndarray, 
              mask_int: np.ndarray, 
              mask_morph: np.ndarray, 
              res: np.ndarray) -> float:
    """Computes Integrated intensity feature.
    This feature refers to "Fmorph_integ_int" (id = 99N0) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Integrated intensity feature.

    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    xgl_int, _, _, _, _, _, _ = get_variables(vol, mask_int, mask_morph, res)
    volume = volume(vol, mask_int, mask_morph, res)
    integ_int = np.mean(xgl_int) * volume

    return integ_int  # Integrated intensity

def moran_i(vol: np.ndarray, 
            mask_int: np.ndarray, 
            mask_morph: np.ndarray, 
            res: np.ndarray, 
            compute_moran_i: bool=False) -> float:
    """Computes Moran's I index feature.
    This feature refers to "Fmorph_moran_i" (id = N365) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).
        compute_moran_i (bool, optional): True to compute Moran's Index.

    Returns:
        float: Value of the Moran's I index feature.

    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)

    if compute_moran_i:
        vol_Mor = vol.copy()
        vol_Mor[mask_int == 0] = np.NaN
        moran_i = get_moran_i(vol_Mor, res)

    return moran_i  # Moran's I index

def geary_c(vol: np.ndarray, 
            mask_int: np.ndarray, 
            mask_morph: np.ndarray, 
            res: np.ndarray, 
            compute_geary_c: bool=False) -> float:
    """Computes Geary's C measure feature.
    This feature refers to "Fmorph_geary_c" (id = NPT7) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).
        compute_geary_c (bool, optional): True to compute Geary's C measure.

    Returns:
        float: Value of the Geary's C measure feature.

    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)

    if compute_geary_c:
        vol_Mor = vol.copy()
        vol_Mor[mask_int == 0] = np.NaN
        geary_c = get_geary_c(vol_Mor, res)

    return geary_c  # Geary's C measure
