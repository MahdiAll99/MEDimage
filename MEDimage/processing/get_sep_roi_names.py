#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List, Tuple

import numpy as np

from ..utils.strfind import strfind


def get_sep_roi_names(nameROIin, delimiters) -> Tuple[List[int], np.ndarray]:
    """Seperated ROI names present in the given ROI name. An ROI name can
    have multiple ROI names seperated with curly brackets and delimeters.

    Note:
        WORKS ONLY FOR DELIMITERS "+" and "-".

    Args:
        nameROIin (str): Name of ROIs that will be extracted from the imagign volume.
            Separated with curly brackets and delimeters. Ex: '{ED}+{ET}'.
        delimiters (List): List of delimeters of "+" and "-".

    Returns:
        List[int]: List of ROI names seperated and excluding curly brackets.
        ndarray: array of 1's and -1's that defines the regions that will
            included and/or excluded in/from the imaging data.

    Examples:
        >>> get_sep_roi_names('{ED}+{ET}', ['+', '-'])
        ['ED', 'ET'], [1]
        >>> get_sep_roi_names('{ED}-{ET}', ['+', '-'])
        ['ED', 'ET'], [-1]
    
    """
    # EX:
    #nameROIin = '{GTV-1}'
    #delimiters = ['\\+','\\-']

    # FINDING "+" and "-"
    ind_plus = strfind(string=nameROIin, pattern=delimiters[0])
    vect_plus = np.ones(len(ind_plus))
    ind_minus = strfind(string=nameROIin, pattern=delimiters[1])
    vect_minus = np.ones(len(ind_minus)) * -1
    ind = np.argsort(np.hstack((ind_plus, ind_minus)))
    vect_plus_minus = np.hstack((vect_plus, vect_minus))[ind]
    ind = np.hstack((ind_plus, ind_minus))[ind].astype(int)
    n_delim = np.size(vect_plus_minus)

    # MAKING SURE "+" and "-" ARE NOT INSIDE A ROIname
    ind_start = strfind(string=nameROIin, pattern="{")
    n_roi = len(ind_start)
    ind_stop = strfind(string=nameROIin, pattern="}")
    ind_keep = np.ones(n_delim, dtype=np.bool)
    for d in np.arange(n_delim):
        for r in np.arange(n_roi):
             # Thus not indise a ROI name
            if (ind_stop[r] - ind[d]) > 0 and (ind[d] - ind_start[r]) > 0:
                ind_keep[d] = False
                break

    ind = ind[ind_keep]
    vect_plus_minus = vect_plus_minus[ind_keep]

    # PARSING ROI NAMES
    if ind.size == 0:
        # Excluding the "{" and "}" at the start and end of the ROIname
        name_roi_out = [nameROIin[1:-1]]
    else:
        nInd = len(ind)
        # Excluding the "{" and "}" at the start and end of the ROIname
        name_roi_out = [nameROIin[1:(ind[0]-1)]]
        for i in np.arange(start=1, stop=nInd):
            # Excluding the "{" and "}" at the start and end of the ROIname
            name_roi_out += [nameROIin[(ind[i-1]+2):(ind[i]-1)]]
        name_roi_out += [nameROIin[(ind[-1]+2):-1]]

    return name_roi_out, vect_plus_minus
