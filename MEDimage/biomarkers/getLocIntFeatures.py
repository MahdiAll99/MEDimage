#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import Code_Radiomics.ImageBiomarkers.getLocPeak
import Code_Radiomics.ImageBiomarkers.getGlobPeak


def getLocIntFeatures(imgObj, roiObj, res, intensity=None):
    """Compute LocIntFeatures.
    -------------------------------------------------------------------------
    - imgObj: Continous image intentisity distribution, with no NaNs
      outside the ROI
    - roiObj: Mask defining the ROI
    - res: [a,b,c] vector specfying the resolution of the volume in mm.  %
      XYZ resolution (world), or JIK resolution (intrinsic matlab).
    - intensity (optional): If 'arbitrary', some feature will not be computed.
      If 'definite', all feature will be computed. If not present as an
      argument, all features will be computed. Here, 'filter' is the same as
      'arbitrary'.
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
    # INTIALIZATION

    if intensity is None:
        definite = True
    elif intensity == 'arbitrary':
        definite = False
    elif intensity == 'definite':
        definite = True
    elif intensity == 'filter':
        definite = False
    else:
        raise ValueError('Fourth argument must either be "arbitrary" or \
                         "definite" or "filter"')

    locInt = {'Floc_peak_local': [], 'Floc_peak_global': []}

    # Local grey level peak
    if definite:
        locInt['Floc_peak_local'] = (Code_Radiomics.ImageBiomarkers.getLocPeak.getLocPeak(
            imgObj, roiObj, res))

        # NEEDS TO BE VECTORIZED FOR FASTER CALCULATION! OR
        # SIMPLY JUST CONVOLUTE A 3D AVERAGING FILTER!
        # locInt['Floc_peak_global'] = (Code_Radiomics.ImageBiomarkers.getGlobPeak.getGlobPeak(
        #        imgObj,roiObj,res))

    return locInt
