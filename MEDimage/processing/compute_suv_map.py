#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pydicom


def compute_suv_map(raw_pet: np.ndarray,
                    dicom_h: pydicom.Dataset) -> np.ndarray:
    """Computes the suv_map of a raw input PET volume. It is assumed that
    the calibration factor was applied beforehand to the PET volume.
    **E.g: raw_pet = raw_pet*RescaleSlope + RescaleIntercept.**

    Args:
        raw_pet (ndarray):3D array representing the PET volume in raw format.
        dicom_h (pydicom.dataset.FileDataset): DICOM header of one of the
            corresponding slice of ``raw_pet``.

    Returns:
        ndarray: ``raw_pet`` converted to SUVs (standard uptake values).
    """
    def dcm_hhmmss(date_str: str) -> float:
        """"Converts to seconds

        Args:
            date_str (str): date string

        Returns:
            float: total seconds
        """
        # Converts to seconds
        if not isinstance(date_str, str):
            date_str = str(date_str)
        hh = float(date_str[0:2])
        mm = float(date_str[2:4])
        ss = float(date_str[4:6])
        tot_sec = hh*60.0*60.0 + mm*60.0 + ss
        return tot_sec

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

    # Get patient weight
    if pydicom_has_tag(dcm_seq=dicom_h, tag=(0x0010, 0x1030)):
        weight = get_pydicom_meta_tag(dcm_seq=dicom_h, tag=(0x0010, 0x1030),
                                      tag_type="float") * 1000.0  # in grams
    else:
        weight = None
    if weight is None:
        weight = 75000.0  # estimation
    try:
        # Get Scan time
        scantime = dcm_hhmmss(date_str=get_pydicom_meta_tag(
            dcm_seq=dicom_h, tag=(0x0008, 0x0032), tag_type="str"))
        # Start Time for the Radiopharmaceutical Injection
        injection_time = dcm_hhmmss(date_str=get_pydicom_meta_tag(
            dcm_seq=dicom_h[0x0054, 0x0016][0],
            tag=(0x0018, 0x1072), tag_type="str"))
        # Half Life for Radionuclide
        half_life = get_pydicom_meta_tag(
            dcm_seq=dicom_h[0x0054, 0x0016][0],
            tag=(0x0018, 0x1075), tag_type="float")
        # Total dose injected for Radionuclide
        injected_dose = get_pydicom_meta_tag(
            dcm_seq=dicom_h[0x0054, 0x0016][0],
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
    suv_map = raw_pet * weight / injected_dose_decay

    return suv_map
