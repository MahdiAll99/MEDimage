from importlib.resources import path


def get_institutions_from_ids(patient_ids: path):
    """This function extracts the institution strings from a cell of patient IDs.

    Args:
        patient_ids (path): Full path to the where a given variable data CSV 
                            table is stored.
                            --> Ex: {'Cervix-UCSF-005';'Cervix-CEM-010'}
    
    Returns:
        institution_catVector: Categorical vector, specifying the institution
                               of each patient_id entry in "patient_ids".
                               --> Ex: {UCSF;CEM}
    """
    return patient_ids.str.rsplit('-', expand=True)[1]
