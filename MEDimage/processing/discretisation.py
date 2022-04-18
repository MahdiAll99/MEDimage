#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
import numpy as np
from Code_Radiomics.ImageProcessing.equalization import equalization


def discretisation(vol_RE, discr_type, nQ=None, userSetMinVal=None, ivh=""):
    """
    -------------------------------------------------------------------------
    --> Use useSetValue as the minimum of range re-segmentation. This is the
    only way to create comparable textures using FBS discretisation. For FBN
    discretisation, this value has no importance as an argument to the
    function and will not be used.
    --> Last argument is optional and MUST BE SET TO 'ivh' FOR IVH FEATURES!

    IMPORTANT --> FOR 'FBS' TYPE, IT IS ASSUMED THAT RE-SEGMENTATION WITH
                  PROPER RANGE WAS ALREADY PERFORMED.
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

    # AZ: NOTE: the "type" variable that appeared in the MATLAB source code
    # matches the name of a standard python function. I have therefore renamed
    # this variable "discr_type"

    # PARSING ARGUMENTS
    volQuant_RE = deepcopy(vol_RE)

    if nQ is None:
        return None

    if not isinstance(nQ, float):
        nQ = float(nQ)

    if discr_type not in ["FBS", "FBN", "FBSequal", "FBNequal"]:
        raise ValueError(
            "discr_type must either be \"FBS\", \"FBN\", \"FBSequal\" or \"FBNequal\".")

    # DISCRETISATION
    if discr_type in ["FBS", "FBSequal"]:
        if userSetMinVal is not None:
            minVal = deepcopy(userSetMinVal)
        else:
            minVal = np.nanmin(volQuant_RE)
    else:
        minVal = np.nanmin(volQuant_RE)

    maxVal = np.nanmax(volQuant_RE)

    if discr_type == "FBS":
        wb = nQ
        wd = wb
        volQuant_RE = np.floor((volQuant_RE - minVal) / wb) + 1.0
    elif discr_type == "FBN":
        wb = (maxVal - minVal) / nQ
        wd = 1.0
        volQuant_RE = np.floor(
            nQ * ((volQuant_RE - minVal)/(maxVal - minVal))) + 1.0
        volQuant_RE[volQuant_RE == np.nanmax(volQuant_RE)] = nQ
    elif discr_type == "FBSequal":
        wb = nQ
        wd = wb
        volQuant_RE = equalization(volQuant_RE)
        volQuant_RE = np.floor((volQuant_RE - minVal) / wb) + 1.0
    elif discr_type == "FBNequal":
        wb = (maxVal - minVal) / nQ
        wd = 1.0
        volQuant_RE = volQuant_RE.astype(np.float32)
        volQuant_RE = equalization(volQuant_RE)
        volQuant_RE = np.floor(
            nQ * ((volQuant_RE - minVal)/(maxVal - minVal))) + 1.0
        volQuant_RE[volQuant_RE == np.nanmax(volQuant_RE)] = nQ
    if ivh == "ivh" and discr_type in ["FBS", "FBSequal"]:
        volQuant_RE = minVal + (volQuant_RE - 0.5) * wb

    return volQuant_RE, wd
