#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.spatial as sc
import Code_Radiomics.ImageBiomarkers.getMesh
import Code_Radiomics.ImageBiomarkers.getMeshVolume
import Code_Radiomics.ImageBiomarkers.getMeshArea
import Code_Radiomics.ImageBiomarkers.getCOM
import Code_Radiomics.ImageBiomarkers.getMax3Ddiam
import Code_Radiomics.ImageBiomarkers.getAxisLengths
import Code_Radiomics.ImageBiomarkers.getAreaDensApprox
import Code_Radiomics.ImageBiomarkers.getMoranI
import Code_Radiomics.ImageBiomarkers.getGearyC
import Code_Radiomics.ImageBiomarkers.getOrientedBoundBox
import Code_Radiomics.ImageBiomarkers.MinVolEllipse.MinVolEllipse as minv


def getMorphFeatures(vol, maskInt, maskMorph, res, intensity=None):
    """Compute MorphFeatures.
    -------------------------------------------------------------------------
     - vol: 3D volume, NON-QUANTIZED, continous imaging intensity distribution
     - maskInt: Intensity mask
     - maskMorph: Morphological mask
     - res: [a,b,c] vector specfying the resolution of the volume in mm.
       # XYZ resolution (world), or JIK resolution (intrinsic matlab).
     - intensity (optional): If 'arbitrary', some feature will not be computed.
       If 'definite', all feature will be computed. If not present as an
       argument, all features will be computed. If 'filter', most features
       will not be computed, except some. The 'filter' option encompasses
       'arbitrary', must is even more stringent. Please see below.

    REFERENCES
    [1] https://arxiv.org/abs/1612.07003 (make it formal with authors, etc.)
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
    if intensity is None:
        definite = True
        im_filter = False
    elif intensity == 'arbitrary':
        definite = False
        im_filter = False
    elif intensity == 'definite':
        definite = True
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
    maskInt = maskInt.copy()
    maskInt = np.pad(maskInt, pad_width=1, mode="constant",
                     constant_values=0.0)
    maskMorph = maskMorph.copy()
    maskMorph = np.pad(maskMorph, pad_width=1, mode="constant",
                       constant_values=0.0)

    # GETTING IMPORTANT VARIABLES
    Xgl_int = np.reshape(vol, np.size(vol), order='F')[np.where(
        np.reshape(maskInt, np.size(maskInt), order='F') == 1)[0]].copy()
    Xgl_morph = np.reshape(vol, np.size(vol), order='F')[np.where(
        np.reshape(maskMorph, np.size(maskMorph), order='F') == 1)[0]].copy()
    # XYZ refers to [Xc,Yc,Zc] in ref. [1].
    XYZ_int, _, _ = Code_Radiomics.ImageBiomarkers.getMesh.getMesh(maskInt, res)
    # XYZ refers to [Xc,Yc,Zc] in ref. [1].
    XYZ_morph, faces, vertices = Code_Radiomics.ImageBiomarkers.getMesh.getMesh(
        maskMorph, res)
    # [X,Y,Z] points of the convex hull.
    # convHull Matlab is convHull.simplices
    convHull = sc.ConvexHull(vertices)

    # STARTING COMPUTATION

    if im_filter is not True:
        # In mm^3
        volume = Code_Radiomics.ImageBiomarkers.getMeshVolume.getMeshVolume(faces, vertices)
        morph['Fmorph_vol'] = volume  # Volume

        # Approximate Volume
        morph['Fmorph_approx_vol'] = np.sum(maskMorph[:]) * np.prod(res)

        # Surface area
        # In mm^2
        area = Code_Radiomics.ImageBiomarkers.getMeshArea.getMeshArea(faces, vertices)
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
        morph['Fmorph_asphericity'] = ((area**3)/(36*np.pi*volume**2))**(
            1/3) - 1

        # Centre of mass shift
        morph['Fmorph_com'] = Code_Radiomics.ImageBiomarkers.getCOM.getCOM(
            Xgl_int, Xgl_morph, XYZ_int, XYZ_morph)

        # Maximum 3D diameter
        #morph['Fmorph_diam'] = Code_Radiomics.ImageBiomarkers.getMax3Ddiam.getMax3Ddiam
        morph['Fmorph_diam'] = np.max(
            sc.distance.pdist(convHull.points[convHull.vertices]))

        # Major axis length
        [major, minor, least] = Code_Radiomics.ImageBiomarkers.getAxisLengths.getAxisLengths(
            XYZ_morph)
        morph['Fmorph_pca_major'] = 4*np.sqrt(major)

        # Minor axis length
        morph['Fmorph_pca_minor'] = 4*np.sqrt(minor)

        # Least axis length
        morph['Fmorph_pca_least'] = 4*np.sqrt(least)

        # Elongation
        morph['Fmorph_pca_elongation'] = np.sqrt(minor/major)

        # Flatness
        morph['Fmorph_pca_flatness'] = np.sqrt(least/major)

        # Volume density - axis-aligned bounding box
        Xc_aabb = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
        Yc_aabb = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
        Zc_aabb = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
        Vaabb = Xc_aabb * Yc_aabb * Zc_aabb
        morph['Fmorph_v_dens_aabb'] = volume / Vaabb

        # Area density - axis-aligned bounding box
        Aaabb = 2*Xc_aabb*Yc_aabb + 2*Xc_aabb*Zc_aabb + 2*Yc_aabb*Zc_aabb
        morph['Fmorph_a_dens_aabb'] = area / Aaabb

        # Volume density - oriented minimum bounding box
        # Implementation of Chan and Tan's algorithm (C.K. Chan, S.T. Tan.
        # Determination of the minimum bounding box of an
        # arbitrary solid: an iterative approach.
        # Comp Struc 79 (2001) 1433-1449
        boundBoxDims = Code_Radiomics.ImageBiomarkers.getOrientedBoundBox.minOrientedBoundBox(
            vertices)
        volBB = np.prod(boundBoxDims)
        morph['Fmorph_v_dens_ombb'] = volume / volBB

        # Area density - oriented minimum bounding box
        Aombb = 2 * (boundBoxDims[0]*boundBoxDims[1] +
                     boundBoxDims[0]*boundBoxDims[2] +
                     boundBoxDims[1]*boundBoxDims[2])
        morph['Fmorph_a_dens_ombb'] = area / Aombb

        # Volume density - approximate enclosing ellipsoid
        a = 2*np.sqrt(major)
        b = 2*np.sqrt(minor)
        c = 2*np.sqrt(least)
        Vaee = (4*np.pi*a*b*c)/3
        morph['Fmorph_v_dens_aee'] = volume / Vaee

        # Area density - approximate enclosing ellipsoid
        Aaee = Code_Radiomics.ImageBiomarkers.getAreaDensApprox.getAreaDensApprox(a, b, c, 20)
        morph['Fmorph_a_dens_aee'] = area / Aaee

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
        A, _ = minv.MinVolEllipse(np.transpose(P), 0.01)
        # New semi-axis lengths
        _, Q, _ = np.linalg.svd(A)
        a = 1/np.sqrt(Q[2])
        b = 1/np.sqrt(Q[1])
        c = 1/np.sqrt(Q[0])
        Vmvee = (4*np.pi*a*b*c)/3
        morph['Fmorph_v_dens_mvee'] = volume/Vmvee

        # Area density - minimum volume enclosing ellipsoid
        # Using a new set of (a,b,c), see Volume density - minimum
        # volume enclosing ellipsoid
        Amvee = Code_Radiomics.ImageBiomarkers.getAreaDensApprox.getAreaDensApprox(
            a, b, c, 20)
        morph['Fmorph_a_dens_mvee'] = area / Amvee

        # Volume density - convex hull
        Vconvex = convHull.volume
        morph['Fmorph_v_dens_conv_hull'] = volume / Vconvex

        # Area density - convex hull
        Aconvex = convHull.area
        morph['Fmorph_a_dens_conv_hull'] = area / Aconvex

    # Integrated intensity
    if definite:
        morph['Fmorph_integ_int'] = np.mean(Xgl_int) * volume

    # Moran's I index
    #vol_Mor = vol.copy()
    #vol_Mor[maskInt == 0] = np.NaN
    #morph['Fmorph_moran_i'] = Code_Radiomics.ImageBiomarkers.getMoranI.getMoranI(vol_Mor,res)

    # Geary's C measure
    #morph['Fmorph_geary_c'] = Code_Radiomics.ImageBiomarkers.getGearyC.getGearyC(vol_Mor,res)

    return morph
