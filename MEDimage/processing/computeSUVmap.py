#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def computeSUVmap(rawPET, dicomH):
    """ 
    -------------------------------------------------------------------------
    DESCRIPTION:
    This function computes the SUVmap of a raw input PET volume. It is
    assumed that the calibration factor was applied beforehand to the PET
    volume (e.g., rawPET = rawPET*RescaleSlope + RescaleIntercept).
    -------------------------------------------------------------------------
    INPUTS:
    - rawPET: 3D array representing the PET volume in raw format.
    - dicomH: DICOM header of one of the corresponding slice of 'rawPET'.
    -------------------------------------------------------------------------
    OUTPUTS:
    - SUVmap: 'rawPET' converted to SUVs (standard uptake values).
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

    # Get patient weight
    if pydicom_has_tag(dcm_seq=dicomH, tag=(0x0010, 0x1030)):
        weight = get_pydicom_meta_tag(dcm_seq=dicomH, tag=(0x0010, 0x1030),
                                      tag_type="float") * 1000.0  # in grams
    else:
        weight = None
    if weight is None:
        weight = 75000.0  # estimation
    try:
        # Get Scan time
        scantime = dcm_hhmmss(dateStr=get_pydicom_meta_tag(
            dcm_seq=dicomH, tag=(0x0008, 0x0032), tag_type="str"))
        # Start Time for the Radiopharmaceutical Injection
        injection_time = dcm_hhmmss(dateStr=get_pydicom_meta_tag(
            dcm_seq=dicomH[0x0054, 0x0016][0],
            tag=(0x0018, 0x1072), tag_type="str"))
        # Half Life for Radionuclide
        half_life = get_pydicom_meta_tag(
            dcm_seq=dicomH[0x0054, 0x0016][0],
            tag=(0x0018, 0x1075), tag_type="float")
        # Total dose injected for Radionuclide
        injected_dose = get_pydicom_meta_tag(
            dcm_seq=dicomH[0x0054, 0x0016][0],
            tag=(0x0018, 0x1074), tag_type="float")
        # Calculate decay
        decay = np.exp(-np.log(2)*(scantime-injection_time)/half_life)
        # Calculate the dose decayed during procedure
        injected_dose_decay = injected_dose*decay  # in Bq
    except KeyError:
        # 90 min waiting time, 15 min preparation
        decay = np.exp(-np.log(2)*(1.75*3600)/6588)
        injected_dose_decay = 420000000 * decay  # 420 MBq

    # Calculate SUV
    SUVmap = rawPET * weight / injected_dose_decay

    return SUVmap


def dcm_hhmmss(dateStr):
    # Converts to seconds
    if not isinstance(dateStr, str):
        dateStr = str(dateStr)
    hh = float(dateStr[0:2])
    mm = float(dateStr[2:4])
    ss = float(dateStr[4:6])
    totSec = hh*60.0*60.0 + mm*60.0 + ss
    return totSec


def pydicom_has_tag(dcm_seq, tag):
    # Checks if tag exists
    return get_pydicom_meta_tag(dcm_seq, tag, test_tag=True)


def get_pydicom_meta_tag(dcm_seq, tag, tag_type=None, default=None,
                         test_tag=False):
    # Reads dicom tag
    # Initialise with default
    tag_value = default
    # Read from header using simple itk
    try:
        tag_value = dcm_seq[tag].value
    except KeyError:
        if test_tag:
            return False
    if test_tag:
        return True
    # Find empty entries
    if tag_value is not None:
        if tag_value == "":
            tag_value = default
    # Cast to correct type (meta tags are usually passed as strings)
    if tag_value is not None:
        # String
        if tag_type == "str":
            tag_value = str(tag_value)
        # Float
        elif tag_type == "float":
            tag_value = float(tag_value)
        # Multiple floats
        elif tag_type == "mult_float":
            tag_value = [float(str_num) for str_num in tag_value]
        # Integer
        elif tag_type == "int":
            tag_value = int(tag_value)
        # Multiple floats
        elif tag_type == "mult_int":
            tag_value = [int(str_num) for str_num in tag_value]
        # Boolean
        elif tag_type == "bool":
            tag_value = bool(tag_value)

    return tag_value
