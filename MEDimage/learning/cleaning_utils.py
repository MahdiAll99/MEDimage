import random
from typing import List

import numpy as np
import pandas as pd


def check_min_n_per_cat(
        variable_table: pd.DataFrame, 
        var_names: List[str], 
        min_n_per_cat: float, 
        type: str) -> pd.DataFrame:
    """
    This Function is different from matlab, it takes the whole variable_table
    and the name of the var_of_type to fit the way pandas works

    Args:
        variable_table (pd.DataFrame): Table of variables.
        var_names (list): List of variable names.
        min_n_per_cat (float): Minimum number of observations per category.
        type (str): Type of variable.
    
    Returns:
        pd.DataFrame: Table of variables with categories under ``min_n_per_cat``.
    """

    for name in var_names:
        table = variable_table[var_names]
        cats = pd.Categorical(table[name]).categories
        for cat in cats:
            flag_cat = (table == cat)
            if sum(flag_cat[name]) < min_n_per_cat:
                if type == 'hcategorical':
                    table.mask(flag_cat, np.nan)
                if type == 'icategorical':
                    table.mask(flag_cat, '')
        variable_table[var_names] = table

    return variable_table

def check_max_percent_cat(variable_table, var_names, max_percent_cat) -> pd.Series:
    """
    This Function is different from matlab, it takes the whole variable_table
    and the name of the var_of_type to fit the way pandas works

    Args:
        variable_table (pd.DataFrame): Table of variables.
        var_names (list): List of variable names.
        max_percent_cat (float): Maximum number of observations per category.
    
    Returns:
        pd.DataFrame: Table of variables with categories over ``max_percent_cat``.
    """

    n_observation = variable_table.shape[0]
    flag_var_out = pd.Series(np.zeros(var_names.size, dtype=bool))
    n = 0
    for name in var_names:
        cats = pd.Categorical(variable_table[name]).categories
        for cat in cats:
            if (variable_table[name] == cat).sum()/n_observation > max_percent_cat:
                flag_var_out[n] = True
                break
        n += 1
    return flag_var_out

def one_hot_encode_table(variable_table: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a table of categorical variables into a table of one-hot encoded variables.

    Args:
        variable_table (pd.DataFrame): Table of variables.
    
    Returns:
        variable_table (pd.DataFrame): Table of variables with one-hot encoded variables.
    """

    #INITIALIZATION
    var_icat = variable_table.Properties['userData']['variables']['icategorical']
    n_var_icat = var_icat.size
    if n_var_icat == 0:
        return variable_table

    # ONE-HOT ENCODING
    for var_name in var_icat:
        categories = variable_table[var_name].unique()
        categories = np.asarray(list(filter(lambda v: v == v, categories)))  # get rid of nan
        categories.sort()
        n_categories = categories.size
        name_encoded = []
        position_to_add = variable_table.columns.get_loc(var_name)+1
        if n_categories == 2:
            n_categories = 1
        for c in range(n_categories):
            cat = categories[c]
            new_name = f"{var_name}__{cat}"
            data_to_add = (variable_table[var_name] == cat).astype(int)
            variable_table.insert(loc=position_to_add, column=new_name, value=data_to_add)
            name_encoded.append(new_name)
        variable_table.Properties['userData']['variables']["one_hot"] = dict()
        variable_table.Properties['userData']['variables']["one_hot"][var_name] = name_encoded
        variable_table = variable_table.drop(var_name, axis=1)

    # UPDATING THE VARIABLE TYPES
    variable_table.Properties['userData']['variables']["icategorical"] = np.array([])
    variable_table.Properties['userData']['variables']["hcategorical"] = np.append([], name_encoded)
    return variable_table
