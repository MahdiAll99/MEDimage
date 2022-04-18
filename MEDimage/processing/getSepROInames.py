#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from Code_Utilities.strfind import strfind


def getSepROInames(nameROIin, delimiters):
    """
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
    WORKS ONLY FOR DELIMITERS "+" and "-"
    """
    # EX:
    #nameROIin = '{GTV-1}'
    #delimiters = ['\\+','\\-']

    # FINDING "+" and "-"
    indPlus = strfind(string=nameROIin, pattern=delimiters[0])
    vectPlus = np.ones(len(indPlus))
    indMinus = strfind(string=nameROIin, pattern=delimiters[1])
    vectMinus = np.ones(len(indMinus)) * -1
    ind = np.argsort(np.hstack((indPlus, indMinus)))
    vectPlusMinus = np.hstack((vectPlus, vectMinus))[ind]
    ind = np.hstack((indPlus, indMinus))[ind].astype(int)
    nDelim = np.size(vectPlusMinus)

    # MAKING SURE "+" and "-" ARE NOT INSIDE A ROIname
    indStart = strfind(string=nameROIin, pattern="{")
    nROI = len(indStart)
    indStop = strfind(string=nameROIin, pattern="}")
    indKeep = np.ones(nDelim, dtype=np.bool)
    for d in np.arange(nDelim):
        for r in np.arange(nROI):
             # Thus not indise a ROI name
            if (indStop[r] - ind[d]) > 0 and (ind[d] - indStart[r]) > 0:
                indKeep[d] = False
                break

    ind = ind[indKeep]
    vectPlusMinus = vectPlusMinus[indKeep]

    # PARSING ROI NAMES
    if ind.size == 0:
        # Excluding the "{" and "}" at the start and end of the ROIname
        nameROIout = [nameROIin[1:-1]]
    else:
        nInd = len(ind)
        # Excluding the "{" and "}" at the start and end of the ROIname
        nameROIout = [nameROIin[1:(ind[0]-1)]]
        for i in np.arange(start=1, stop=nInd):
            # Excluding the "{" and "}" at the start and end of the ROIname
            nameROIout += [nameROIin[(ind[i-1]+2):(ind[i]-1)]]
        nameROIout += [nameROIin[(ind[-1]+2):-1]]

    return nameROIout, vectPlusMinus
