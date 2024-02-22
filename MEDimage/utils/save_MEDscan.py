import pickle
from pathlib import Path

from ..MEDscan import MEDscan


def save_MEDscan(medscan: MEDscan,
                  path_save: Path) -> str:
    """Saves MEDscan class instance in a pickle object
    
    Args:
        medscan (MEDscan): MEDscan instance
        path_save (Path): MEDscan instance saving paths
    
    Returns:
        None.
    """

    series_description = medscan.series_description.translate({ord(ch): '-' for ch in '/\\ ()&:*'})
    name_id = medscan.patientID
    name_id = name_id.translate({ord(ch): '-' for ch in '/\\ ()&:*'})

    # final saving name
    name_complete = name_id + '__' + series_description + '.' + medscan.type + '.npy'
    
    # save
    with open(path_save / name_complete,'wb') as f: 
        pickle.dump(medscan, f)

    return name_complete
