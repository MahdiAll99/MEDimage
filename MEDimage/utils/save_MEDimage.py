import os
from pathlib import Path
import pickle

from MEDimage.MEDimage import MEDimage


def save_MEDimage(MEDimg: MEDimage, SeriesDescription: str, pathSave: Path) -> None:
    """
    Saves MEDimage class instance in a pickle object
    
    Args:
        MEDimg (MEDimage): MEDimage instance
        SeriesDescription (str): field of DICOM headers of imaging 
            volume with TAG: (0008,103E). For ex: 'T1'
        pathSave (Path): MEDimage instance saving paths
    
    Returns:
        None.
    """

    SeriesDescription = SeriesDescription.translate({ord(ch): '-' for ch in '/\\ ()&:*'})
    nameID = MEDimg.patientID
    nameID = nameID.translate({ord(ch): '-' for ch in '/\\ ()&:*'})
    MEDimg.format = "npy"

    # final saving name
    nameComplete = nameID + '__' + SeriesDescription + '.' + MEDimg.type + '.npy'
    
    # save
    with open(pathSave / nameComplete,'wb') as f: 
        pickle.dump(MEDimg, f)
