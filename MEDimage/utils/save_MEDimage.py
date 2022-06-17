import os
from pathlib import Path
import pickle

from MEDimage.MEDimage import MEDimage


def save_MEDimage(MEDimg: MEDimage, series_description: str, path_save: Path) -> None:
    """
    Saves MEDimage class instance in a pickle object
    
    Args:
        MEDimg (MEDimage): MEDimage instance
        series_description (str): field of DICOM headers of imaging 
            volume with TAG: (0008,103E). For ex: 'T1'
        path_save (Path): MEDimage instance saving paths
    
    Returns:
        None.
    """

    series_description = series_description.translate({ord(ch): '-' for ch in '/\\ ()&:*'})
    name_id = MEDimg.patient_id
    name_id = name_id.translate({ord(ch): '-' for ch in '/\\ ()&:*'})
    MEDimg.format = "npy"

    # final saving name
    name_complete = name_id + '__' + series_description + '.' + MEDimg.type + '.npy'
    
    # save
    with open(path_save / name_complete,'wb') as f: 
        pickle.dump(MEDimg, f)
