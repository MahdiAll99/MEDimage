#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from Code_Radiomics.ImageProcessing.getSepROInames import getSepROInames


def findContour(MEDImg, nameROI, nameStructureSet=None):
    """
    -------------------------------------------------------------------------
    THIRD ARGUMENT IS OPTIONAL AND IS MEANT TO BE USED ONLY FOR RTstructs.
    HOWEVER, IF TWO ROIs WITH THE SAME NAME BUT
    FROM DIFFERENT STRUCTURE SETS ARE PRESENT, THE ALGORITHM WILL JUST OUTPUT
    THE FIRST CONTOUR IT FINDS, SO BEWARE. BETTER TO ALWAYS SUPPLY A
    STRUCTURE SET NAME.
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

    delimiters = ["\+", "\-"]
    if hasattr(MEDImg,'niii'):  # Used by default if present
        nContourData = len(MEDImg.Data[1]['nii']['mask'])
    elif hasattr(MEDImg,'nrrd'):  # Used by default if present
        nContourData = len(MEDImg.Data[1]['nrrd']['mask'])
    elif hasattr(MEDImg,'img'):  # Used as second default if present
        nContourData = len(MEDImg.Data[1]['img']['mask'])
    else:  # Otherwise we use DICOM data
        nContourData = len(MEDImg.Data[1]['scan']['contour'])

    nameROI, vectPlusMinus = getSepROInames(nameROI, delimiters)
    contourNumber = np.zeros(len(nameROI))

    if nameStructureSet is None:
        nameStructureSet = []

    if nameStructureSet:
        nameStructureSet, _ = getSepROInames(nameStructureSet, delimiters)
        if len(nameROI) != len(nameStructureSet):
            raise ValueError(
                "The numbers of defined ROI names and Structure Set names are not the same")

    for i in range(0, len(nameROI)):
        for j in range(0, nContourData):
            if hasattr(MEDImg,'niii'):  # Used by default if present
                nameTemp = MEDImg.Data[1]['nii']['mask'][j]['name']
            elif hasattr(MEDImg,'nrrd'):  # Used as second default if present
                nameTemp = MEDImg.Data[1]['nrrd']['mask'][j]['name']
            elif hasattr(MEDImg,'img'):  # Used as third default if present
                nameTemp = MEDImg.Data[1]['img']['mask'][j]['name']
            else:  # Otherwise we use DICOM data
                nameTemp = MEDImg.Data[1]['scan']['contour'][j]['name']
            if nameTemp == nameROI[i]:
                if nameStructureSet:
                    # FOR DICOM + RTSTRUCT
                    nameSetTemp = MEDImg.Data[1]['scan']['contour'][j]['nameSet']
                    if nameSetTemp == nameStructureSet[i]:
                        contourNumber[i] = j
                        break
                else:
                    contourNumber[i] = j
                    break

    nROI = np.size(contourNumber)
    contourString = str(contourNumber[0].astype(int))
    for i in range(1, nROI):
        if vectPlusMinus[i-1] == 1:
            sign = '+'
        elif vectPlusMinus[i-1] == -1:
            sign = '-'
        contourString = contourString + sign + \
            str(contourNumber[i].astype(int))

    return contourString
