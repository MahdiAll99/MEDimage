#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Dict, Tuple

import numpy as np
from MEDimage.MEDscan import MEDscan

from ..biomarkers.utils import find_i_x, find_v_x


def init_ivh(medscan: MEDscan, 
             vol: np.ndarray, 
             vol_int_re: np.ndarray, 
             wd: int, 
             user_set_range: np.ndarray=None) -> Tuple[np.ndarray, np.ndarray, np.integer, np.integer]:
    """Computes Intensity-volume Histogram Features.

    Note:
        For the input volume:

        - Naturally discretised volume can be kept as it is (e.g. HU values of CT scans) 
        - All other volumes with continuous intensity distribution should be \
            quantized (e.g., nBins = 100), with levels = [min, ..., max]

    Args:
        medscan (MEDscan): MEDscan instance.
        vol (ndarray): 3D volume, QUANTIZED, with NaNs outside the region of interest
        vol_int_re (ndarray): 3D volume, with NaNs outside the region of interest
        wd (int): Discretisation width.
        user_set_range (ndarray, optional): 1-D array with shape (1,2) of the intensity range.

    Returns:
        Dict: Dict of the Intensity Histogram Features.
    """
    try:
        if 'type' in medscan.params.process.ivh and medscan.params.process.ivh:
            # PET example case (definite intensity units -- continuous case)
            if medscan.params.process.ivh['type'] == 'FBS' or medscan.params.process.ivh['type'] == 'FBSequal':
                range_fbs = [0, 0]
                if not medscan.params.process.im_range:
                    range_fbs[0] = np.nanmin(vol_int_re)
                    range_fbs[1] = np.nanmax(vol_int_re)
                else:
                    if medscan.params.process.im_range[0] == -np.inf:
                        range_fbs[0] = np.nanmin(vol_int_re)
                    else:
                        range_fbs[0] = medscan.params.process.im_range[0]
                    if medscan.params.process.im_range[1] == np.inf:
                        range_fbs[1] = np.nanmax(vol_int_re)
                    else:
                        range_fbs[1] = medscan.params.process.im_range[1]
                # In this case, wd = wb (see discretisation.m)
                range_fbs[0] = range_fbs[0] + 0.5*wd
                # In this case, wd = wb (see discretisation.m)
                range_fbs[1] = range_fbs[1] - 0.5*wd
                user_set_range = range_fbs

            else:  # MRI example case (arbitrary intensity units)
                user_set_range = None

        else:  # CT example case (definite intensity units -- discrete case)
            user_set_range = medscan.params.process.im_range

        # INITIALIZATION
        X = vol[~np.isnan(vol[:])]

        if (vol is not None) & (wd is not None) & (user_set_range is not None):
            if user_set_range:
                min_val = user_set_range[0]
                max_val = user_set_range[1]
            else:
                min_val = np.min(X)
                max_val = np.max(X)
        else:
            min_val = np.min(X)
            max_val = np.max(X)

        if max_val == np.inf:
            max_val = np.max(X)

        if min_val == -np.inf:
            min_val = np.min(X)

        # Vector of grey-levels.
        # Values are generated within the half-open interval [min_val,max_val+wd)
        levels = np.arange(min_val, max_val + wd, wd)
        n_g = levels.size
        n_v = X.size
    
    except Exception as e:
        message = f'PROBLEM WITH INITIALIZATION OF INTENSITY-VOLUME HISTOGRAM FEATURES \n {e}'
        medscan.radiomics.image['intVolHist_3D'][medscan.params.radiomics.ivh_name].update(
            {'error': 'ERROR_INITIALIZATION'})
        logging.error(message)
        print(message)

    return X, levels, n_g, n_v

def extract_all(medscan: MEDscan, 
                vol: np.ndarray, 
                vol_int_re: np.ndarray, 
                wd: int, 
                user_set_range: np.ndarray=None) -> Dict:
    """Computes Intensity-volume Histogram Features.
    This features refer to Intensity-volume histogram family in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Note:
        For the input volume, naturally discretised volume can be kept as it is (e.g. HU values of CT scans).
        All other volumes with continuous intensity distribution should be
        quantized (e.g., nBins = 100), with levels = [min, ..., max]

    Args:
        medscan (MEDscan): MEDscan instance.
        vol (ndarray): 3D volume, QUANTIZED, with NaNs outside the region of interest
        vol_int_re (ndarray): 3D volume, with NaNs outside the region of interest
        wd (int): Discretisation width.
        user_set_range (ndarray, optional): 1-D array with shape (1,2) of the intensity range.

    Returns:
        Dict: Dict of the Intensity Histogram Features.
    """
    try:
        # Retrieve relevant parameters from init_ivh() method.
        X, levels, n_g, n_v = init_ivh(medscan, vol, vol_int_re, wd, user_set_range)

        # Initialization of final structure (Dictionary) containing all features.
        int_vol_hist = {'Fivh_V10': [],
                    'Fivh_V90': [],
                    'Fivh_I10': [],
                    'Fivh_I90': [],
                    'Fivh_V10minusV90': [],
                    'Fivh_I10minusI90': [],
                    'Fivh_auc': []
                    }

        # Calculating fractional volume
        fract_vol = np.zeros(n_g)
        for i in range(0, n_g):
            fract_vol[i] = 1 - np.sum(X < levels[i])/n_v

        # Calculating intensity fraction
        fract_int = (levels - np.min(levels)) / (np.max(levels) - np.min(levels))

        # Volume at intensity fraction 10
        v10 = find_v_x(fract_int, fract_vol, 10)
        int_vol_hist['Fivh_V10'] = v10

        # Volume at intensity fraction 90
        v90 = find_v_x(fract_int, fract_vol, 90)
        int_vol_hist['Fivh_V90'] = v90

        # Intensity at volume fraction 10
        #   For initial arbitrary intensities,
        #   we will always be discretising (1000 bins).
        #   So intensities are definite here.
        i10 = find_i_x(levels, fract_vol, 10)
        int_vol_hist['Fivh_I10'] = i10

        # Intensity at volume fraction 90
        #   For initial arbitrary intensities,
        #   we will always be discretising (1000 bins).
        #   So intensities are definite here.
        i90 = find_i_x(levels, fract_vol, 90)
        int_vol_hist['Fivh_I90'] = i90

        # Volume at intensity fraction difference v10-v90
        int_vol_hist['Fivh_V10minusV90'] = v10 - v90

        # Intensity at volume fraction difference i10-i90
        #   For initial arbitrary intensities,
        #   we will always be discretising (1000 bins).
        #   So intensities are definite here.
        int_vol_hist['Fivh_I10minusI90'] = i10 - i90

        # Area under IVH curve
        int_vol_hist['Fivh_auc'] = np.trapz(fract_vol) / (n_g - 1)

    except Exception as e:
        message = f'PROBLEM WITH COMPUTATION OF INTENSITY-VOLUME HISTOGRAM FEATURES \n {e}'
        medscan.radiomics.image['intVolHist_3D'][medscan.params.radiomics.ivh_name].update(
            {'error': 'ERROR_COMPUTATION'})
        logging.error(message)
        print(message)

    return int_vol_hist

def v10(medscan: MEDscan, 
        vol: np.ndarray, 
        vol_int_re: np.ndarray, 
        wd: int, 
        user_set_range: np.ndarray=None) -> float:
    """Computes Volume at intensity fraction 10 feature.
    This feature refers to "Fivh_V10" (ID = BC2M) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.


    Args:
        medscan (MEDscan): MEDscan instance.
        vol (ndarray): 3D volume, QUANTIZED, with NaNs outside the region of interest
        vol_int_re (ndarray): 3D volume, with NaNs outside the region of interest
        wd (int): Discretisation width.
        user_set_range (ndarray, optional): 1-D array with shape (1,2) of the intensity range.

    Returns:
        float: Volume at intensity fraction 10 feature.
    """
    try:
        # Retrieve relevant parameters from init_ivh() method.
        X, levels, n_g, n_v = init_ivh(medscan, vol, vol_int_re, wd, user_set_range)

        # Calculating fractional volume
        fract_vol = np.zeros(n_g)
        for i in range(0, n_g):
            fract_vol[i] = 1 - np.sum(X < levels[i]) / n_v

        # Calculating intensity fraction
        fract_int = (levels - np.min(levels))/(np.max(levels) - np.min(levels))

        # Volume at intensity fraction 10
        v10 = find_v_x(fract_int, fract_vol, 10)

    except Exception as e:
        message = f'PROBLEM WITH COMPUTATION OF V10 FEATURE \n {e}'
        medscan.radiomics.image['intVolHist_3D'][medscan.params.radiomics.ivh_name].update(
            {'error': 'ERROR_V10'})
        logging.error(message)
        print(message)

    return v10

def v90(medscan: MEDscan, 
        vol: np.ndarray, 
        vol_int_re: np.ndarray, 
        wd: int, 
        user_set_range: np.ndarray=None) -> float:
    """Computes Volume at intensity fraction 90 feature.
    This feature refers to "Fivh_V90" (ID = BC2M) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        medscan (MEDscan): MEDscan instance.
        vol (ndarray): 3D volume, QUANTIZED, with NaNs outside the region of interest
        vol_int_re (ndarray): 3D volume, with NaNs outside the region of interest
        wd (int): Discretisation width.
        user_set_range (ndarray, optional): 1-D array with shape (1,2) of the intensity range.

    Returns:
        float: Volume at intensity fraction 90 feature.
    """
    try:
        # Retrieve relevant parameters from init_ivh() method.
        X, levels, n_g, n_v = init_ivh(medscan, vol, vol_int_re, wd, user_set_range)

        # Calculating fractional volume
        fract_vol = np.zeros(n_g)
        for i in range(0, n_g):
            fract_vol[i] = 1 - np.sum(X < levels[i]) / n_v

        # Calculating intensity fraction
        fract_int = (levels - np.min(levels)) / (np.max(levels) - np.min(levels))

        # Volume at intensity fraction 90
        v90 = find_v_x(fract_int, fract_vol, 90)

    except Exception as e:
        message = f'PROBLEM WITH COMPUTATION OF V90 FEATURE \n {e}'
        medscan.radiomics.image['intVolHist_3D'][medscan.params.radiomics.ivh_name].update(
            {'error': 'ERROR_V90'})
        logging.error(message)
        print(message)

    return v90

def i10(medscan: MEDscan, 
        vol: np.ndarray, 
        vol_int_re: np.ndarray, 
        wd: int, 
        user_set_range: np.ndarray=None) -> float:
    """Computes Intensity at volume fraction 10 feature.
    This feature refers to "Fivh_I10" (ID = GBPN) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        medscan (MEDscan): MEDscan instance.
        vol (ndarray): 3D volume, QUANTIZED, with NaNs outside the region of interest
        vol_int_re (ndarray): 3D volume, with NaNs outside the region of interest
        wd (int): Discretisation width.
        user_set_range (ndarray, optional): 1-D array with shape (1,2) of the intensity range.

    Returns:
        float: Intensity at volume fraction 10 feature.
    """
    try:
        # Retrieve relevant parameters from init_ivh() method.
        X, levels, n_g, n_v = init_ivh(medscan, vol, vol_int_re, wd, user_set_range)

        # Calculating fractional volume
        fract_vol = np.zeros(n_g)
        for i in range(0, n_g):
            fract_vol[i] = 1 - np.sum(X < levels[i]) / n_v

        # Intensity at volume fraction 10
        #   For initial arbitrary intensities,
        #   we will always be discretising (1000 bins).
        #   So intensities are definite here.
        i10 = find_i_x(levels, fract_vol, 10)

    except Exception as e:
        message = f'PROBLEM WITH COMPUTATION OF I10 FEATURE \n {e}'
        medscan.radiomics.image['intVolHist_3D'][medscan.params.radiomics.ivh_name].update(
            {'error': 'ERROR_I10'})
        logging.error(message)
        print(message)

    return i10

def i90(medscan: MEDscan, 
        vol: np.ndarray, 
        vol_int_re: np.ndarray, 
        wd: int, 
        user_set_range: np.ndarray=None) -> float:
    """Computes Intensity at volume fraction 90 feature.
    This feature refers to "Fivh_I90" (ID = GBPN) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        medscan (MEDscan): MEDscan instance.
        vol (ndarray): 3D volume, QUANTIZED, with NaNs outside the region of interest
        vol_int_re (ndarray): 3D volume, with NaNs outside the region of interest
        wd (int): Discretisation width.
        user_set_range (ndarray, optional): 1-D array with shape (1,2) of the intensity range.

    Returns:
        float: Intensity at volume fraction 90 feature.
    """
    try:
        # Retrieve relevant parameters from init_ivh() method.
        X, levels, n_g, n_v = init_ivh(medscan, vol, vol_int_re, wd, user_set_range)

        # Calculating fractional volume
        fract_vol = np.zeros(n_g)
        for i in range(0, n_g):
            fract_vol[i] = 1 - np.sum(X < levels[i]) / n_v

        # Intensity at volume fraction 90
        #   For initial arbitrary intensities,
        #   we will always be discretising (1000 bins).
        #   So intensities are definite here.
        i90 = find_i_x(levels, fract_vol, 90)

    except Exception as e:
        message = f'PROBLEM WITH COMPUTATION OF I90 FEATURE \n {e}'
        medscan.radiomics.image['intVolHist_3D'][medscan.params.radiomics.ivh_name].update(
            {'error': 'ERROR_I90'})
        logging.error(message)
        print(message)

    return i90

def v10_minus_v90(medscan: MEDscan, 
                vol: np.ndarray, 
                vol_int_re: np.ndarray, 
                wd: int, 
                user_set_range: np.ndarray=None) -> float:
    """Computes Volume at intensity fraction difference v10-v90
    This feature refers to "Fivh_V10minusV90" (ID = DDTU) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        medscan (MEDscan): MEDscan instance.
        vol (ndarray): 3D volume, QUANTIZED, with NaNs outside the region of interest
        vol_int_re (ndarray): 3D volume, with NaNs outside the region of interest
        wd (int): Discretisation width.
        user_set_range (ndarray, optional): 1-D array with shape (1,2) of the intensity range.

    Returns:
        float: Volume at intensity fraction difference v10-v90 feature.
    """
    try:
        # Retrieve relevant parameters from init_ivh() method.
        X, levels, n_g, n_v = init_ivh(medscan, vol, vol_int_re, wd, user_set_range)

        # Calculating fractional volume
        fract_vol = np.zeros(n_g)
        for i in range(0, n_g):
            fract_vol[i] = 1 - np.sum(X < levels[i]) / n_v

        # Calculating intensity fraction
        fract_int = (levels - np.min(levels)) / (np.max(levels) - np.min(levels))

        # Volume at intensity fraction 10
        v10 = find_v_x(fract_int, fract_vol, 10)

        # Volume at intensity fraction 90
        v90 = find_v_x(fract_int, fract_vol, 90)

    except Exception as e:
        message = f'PROBLEM WITH COMPUTATION OF V10minusV90 FEATURE \n {e}'
        medscan.radiomics.image['intVolHist_3D'][medscan.params.radiomics.ivh_name].update(
            {'error': 'ERROR_V10minusV90'})
        logging.error(message)
        print(message)

    return v10 - v90

def i10_minus_i90(medscan: MEDscan, 
                vol: np.ndarray, 
                vol_int_re: np.ndarray, 
                wd: int, 
                user_set_range: np.ndarray=None) -> float:
    """Computes Intensity at volume fraction difference i10-i90
    This feature refers to "Fivh_I10minusI90" (ID = CNV2) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        medscan (MEDscan): MEDscan instance.
        vol (ndarray): 3D volume, QUANTIZED, with NaNs outside the region of interest
        vol_int_re (ndarray): 3D volume, with NaNs outside the region of interest
        wd (int): Discretisation width.
        user_set_range (ndarray, optional): 1-D array with shape (1,2) of the intensity range.

    Returns:
        float: Intensity at volume fraction difference i10-i90 feature.
    """
    try:
        # Retrieve relevant parameters from init_ivh() method.
        X, levels, n_g, n_v = init_ivh(medscan, vol, vol_int_re, wd, user_set_range)

        # Calculating fractional volume
        fract_vol = np.zeros(n_g)
        for i in range(0, n_g):
            fract_vol[i] = 1 - np.sum(X < levels[i]) / n_v

        # Intensity at volume fraction 10
        #   For initial arbitrary intensities,
        #   we will always be discretising (1000 bins).
        #   So intensities are definite here.
        i10 = find_i_x(levels, fract_vol, 10)

        # Intensity at volume fraction 90
        #   For initial arbitrary intensities,
        #   we will always be discretising (1000 bins).
        #   So intensities are definite here.
        i90 = find_i_x(levels, fract_vol, 90)

    except Exception as e:
        message = f'PROBLEM WITH COMPUTATION OF I10minusI90 FEATURE \n {e}'
        medscan.radiomics.image['intVolHist_3D'][medscan.params.radiomics.ivh_name].update(
            {'error': 'ERROR_I10minusI90'})
        logging.error(message)
        print(message)

    return i10 - i90

def auc(medscan: MEDscan, 
                vol: np.ndarray, 
                vol_int_re: np.ndarray, 
                wd: int, 
                user_set_range: np.ndarray=None) -> float:
    """
    Computes Area under IVH curve.
    This feature refers to "Fivh_auc" (ID = 9CMM) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Note:
        For the input volume:

            * Naturally discretised volume can be kept as it is (e.g. HU values of CT scans)
            * All other volumes with continuous intensity distribution should be \
            quantized (e.g., nBins = 100), with levels = [min, ..., max]

    Args:
        medscan(MEDscan): MEDscan instance.
        vol(ndarray): 3D volume, QUANTIZED, with NaNs outside the region of interest
        vol_int_re(ndarray): 3D volume, with NaNs outside the region of interest
        wd(int): Discretisation width.
        user_set_range(ndarray, optional): 1-D array with shape (1,2) of the intensity range.

    Returns:
        float: Area under IVH curve feature.
    """
    try:
        # Retrieve relevant parameters from init_ivh() method.
        X, levels, n_g, n_v = init_ivh(medscan, vol, vol_int_re, wd, user_set_range)

        # Calculating fractional volume
        fract_vol = np.zeros(n_g)
        for i in range(0, n_g):
            fract_vol[i] = 1 - np.sum(X < levels[i]) / n_v

        # Area under IVH curve
        auc = np.trapz(fract_vol) / (n_g - 1)

    except Exception as e:
        message = f'PROBLEM WITH COMPUTATION OF AUC FEATURE \n {e}'
        medscan.radiomics.image['intVolHist_3D'][medscan.params.radiomics.ivh_name].update(
            {'error': 'ERROR_AUC'})
        logging.error(message)
        print(message)

    return auc
