from typing import List

import numpy as np


def get_full_rad_names(str_user_data: str, rad_var_ids: List):
    """
    Returns the full real names of the radiomics variables (sequential names are not very informative)
    Args:
        str_user_data: string containing the full rad names
        rad_var_ids: can get it by doing table.column.values
    
    Returns:
        List: List of full radiomic names.
    """
    full_rad_names = np.array([])
    for rad_var in rad_var_ids:
        ind_var = int(rad_var[6:])
        full_rad_names = np.append(full_rad_names, str_user_data.split('||')[ind_var].split(':')[1])

    return full_rad_names
