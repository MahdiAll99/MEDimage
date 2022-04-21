#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List, Union

import numpy as np
from ..utils.strfind import strfind


def getSepROInames(nameROIin, delimiters) -> Union[List[str], List[int]]:
    """Seperated ROI names present in the given ROI name. An ROI name can
    have multiple ROI names seperated with curly brackets and delimeters.

    Note:
        WORKS ONLY FOR DELIMITERS "+" and "-".

    Args:
        nameROIin (str): Name of ROI that will be extracted from the imagign volume.
        delimiters (List): List of delimeters of "+" and "-".

    Returns:
        List: List ROI names seperated and excluding curly brackets({}).
        List: List of 1's and -1's that defines the regions that will
            included and/or excluded in/from the imaging data.

    Examples:
        >>> getSepROInames('{ED}+{ET}', ['+', '-'])
        ['ED', 'ET'], [1]
        >>> getSepROInames('{ED}-{ET}', ['+', '-'])
        ['ED', 'ET'], [-1]
    
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
