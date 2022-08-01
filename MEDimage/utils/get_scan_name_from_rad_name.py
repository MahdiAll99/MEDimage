#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def get_scan_name_from_rad_name(rad_name: str) -> str:
    """Finds the imaging scan name from thr radiomics structure name

    Args:
        rad_name (str): radiomics structure name.
    
    Returns:
        str: String of the imaging scan name
    
    Example:
        >>> get_scan_name_from_rad_name('STS-McGill-001__T1(tumourAndEdema).MRscan')
        'T1'
    """
    ind_double_under = rad_name.find('__')
    ind_open_par = rad_name.find('(')
    scan_name = rad_name[ind_double_under + 2:ind_open_par]

    return scan_name