#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def get_patient_id_from_scan_name(rad_name: str) -> str:
    """
    Finds the patient id from the given string

    Args:
        rad_name(str): Name of a scan or a radiomics structure
    
    Returns:
        str: patient id

    Example:
        >>> get_patient_id_from_scan_name('STS-McGill-001__T1(tumourAndEdema).MRscan')
        STS-McGill-001
    """
    ind_double_under = rad_name.find('__')
    patientID = rad_name[:ind_double_under]

    return patientID
