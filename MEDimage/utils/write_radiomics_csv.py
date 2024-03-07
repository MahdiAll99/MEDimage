#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Union

import numpy as np


def write_radiomics_csv(path_radiomics_table: Union[Path, str]) -> None:
    """
    Loads a radiomics structure (dict with radiomics features) to convert it to a CSV file and save it.

    Args:
        path_radiomics_table(Union[Path, str]): path to the radiomics dict.
    
    Returns:
        None.
    """

    # INITIALIZATION
    path_radiomics_table = Path(path_radiomics_table)
    path_to_table = path_radiomics_table.parent
    name_table = path_radiomics_table.stem

    # LOAD RADIOMICS TABLE
    radiomics_table_dict = np.load(path_radiomics_table, allow_pickle=True)[0]

    # WRITE RADIOMICS TABLE IN CSV FORMAT
    csv_name = name_table + '.csv'
    csv_path = path_to_table / csv_name
    radiomics_table_dict['Table'] = radiomics_table_dict['Table'].fillna(value='NaN')
    radiomics_table_dict['Table'] = radiomics_table_dict['Table'].sort_index()
    radiomics_table_dict['Table'].to_csv(csv_path,
                                       sep=',',
                                       encoding='utf-8',
                                       index=True,
                                       index_label=radiomics_table_dict['Properties']['DimensionNames'][0])

    # WRITE DEFINITIONS.TXT
    txt_name = name_table + '.txt'
    txt_Path = path_to_table / txt_name

    # WRITE THE CSV
    fid = open(txt_Path, 'w')
    fid.write(radiomics_table_dict['Properties']['UserData'])
    fid.close()
