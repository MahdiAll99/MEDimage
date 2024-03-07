import pandas as pd


def get_institutions_from_ids(patient_ids):
    """
    Extracts the institution strings from a cell of patient IDs.

    Args:
        patient_ids (Any): Patient ID (string, list of strings or pandas Series). Ex: 'Cervix-CEM-010'.
    
    Returns:
        str: Categorical vector, specifying the institution of each patient_id entry in "patient_ids". Ex: 'CEM'.
    """
    if isinstance(patient_ids, list):
        patient_ids = pd.Series(patient_ids)
    return patient_ids.str.rsplit('-', expand=True)[1]
