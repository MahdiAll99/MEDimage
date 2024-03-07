#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import random
from json import load
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from ..utils.get_patient_id_from_scan_name import get_patient_id_from_scan_name
from ..utils.initialize_features_names import initialize_features_names


def create_radiomics_table(radiomics_files_paths: List, image_space: str, log_file: Union[str, Path]) -> Dict:
    """
    Creates a dictionary with a csv and other information

    Args:
        radiomics_files_paths(List): List of paths to the radiomics JSON files.
        image_space(str): String of the image space that contains the extracted features
        log_file(Union[str, Path]): Path to logging file.

    Returns:
        Dict: Dictionary containing the extracted radiomics and other info (patientID, feature names...)
    """
    if log_file:
        # Setting up logging settings
        logging.basicConfig(filename=log_file, level=logging.DEBUG)
    
    # INITIALIZATIONS OF RADIOMICS STRUCTURES
    n_files = len(radiomics_files_paths)
    patientID = [0] * n_files
    rad_structs = [0] * n_files
    file_open = [False] * n_files

    for f in range(n_files):
        with open(radiomics_files_paths[f], "r") as fp: 
            radStruct = load(fp)
        rad_structs[f] = radStruct
        file_open[f] = True
        patientID[f] = get_patient_id_from_scan_name(radiomics_files_paths[f].stem)

    # INITIALIZE FEATURE NAMES
    logging.info(f"\nnFiles: {n_files}")
    non_text_cell = []
    text_cell = []
    while len(non_text_cell) == 0 and len(text_cell) == 0:
        try:
            rand_patient = np.floor(n_files * random.uniform(0, 1)).astype(int)
            with open(radiomics_files_paths[rand_patient], "r") as fp: 
                radiomics_struct = load(fp)

            # IMAGE SPACE STRUCTURE --> .morph, .locInt, ...,  .texture
            image_space_struct = radiomics_struct[image_space]
            non_text_cell, text_cell = initialize_features_names(image_space_struct)
        except:
            pass

    # CREATE TABLE DATA
    features_name_dict = {}
    str_table = ''
    str_names = '||'
    count_var = 0

    # Non-texture features
    for im_type in range(len(non_text_cell[0])):
        for param in range(len(non_text_cell[2][im_type])):
            for feat in range(len(non_text_cell[1][im_type])):
                count_var = count_var + 1
                feature_name = 'radVar' + str(count_var)
                features_name_dict.update({feature_name: [0] * n_files})
                real_name_feature = non_text_cell[0][im_type] + '__' + \
                    non_text_cell[1][im_type][feat] + '__' + \
                    non_text_cell[2][im_type][param]
                str_table = str_table + feature_name + ','
                str_names = str_names + feature_name + ':' + real_name_feature + '||'

                for f in range(n_files):
                    if file_open[f]:
                        try:
                            val = rad_structs[f][image_space][
                                non_text_cell[0][im_type]][
                                non_text_cell[2][im_type][param]][
                                non_text_cell[1][im_type][feat]]
                        except:
                            val = np.NaN
                        if type(val) in [str, list]:
                            val = np.NaN
                    else:
                        val = np.NaN
                    features_name_dict[feature_name][f] = val

    # Texture features
    for im_type in range(len(text_cell[0])):
        for param in range(len(text_cell[2][im_type])):
            for feat in range(len(text_cell[1][im_type])):
                count_var = count_var + 1
                feature_name = 'radVar' + str(count_var)
                features_name_dict.update({feature_name: [0] * n_files})
                real_name_feature = text_cell[0][im_type] + '__' + \
                    text_cell[1][im_type][feat] + '__' + \
                    text_cell[2][im_type][param]
                str_table = str_table + feature_name + ','
                str_names = str_names + feature_name + ':' + real_name_feature + '||'
                for f in range(n_files):
                    if file_open[f]:
                        try:
                            val = rad_structs[f][image_space]['texture'][
                                text_cell[0][im_type]][
                                text_cell[2][im_type][param]][
                                text_cell[1][im_type][feat]]
                        except:
                            val = np.NaN
                        if type(val) in [str, list]:
                            val = np.NaN
                    else:
                        val = np.NaN
                    features_name_dict[feature_name][f] = val

    radiomics_table_dict = {
        'Table': pd.DataFrame(features_name_dict, index=patientID),
        'Properties': {'UserData': str_names,
                       'RowNames': patientID,
                       'DimensionNames': ['PatientID', 'Variables'],
                       'VariableNames': [key for key in features_name_dict.keys()]
                       }}

    return radiomics_table_dict
