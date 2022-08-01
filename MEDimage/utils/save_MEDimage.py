import pickle
from pathlib import Path

from MEDimage.MEDimage import MEDimage


def save_MEDimage(MEDimg: MEDimage,
                  path_save: Path) -> None:
    """Saves MEDimage class instance in a pickle object
    
    Args:
        MEDimg (MEDimage): MEDimage instance
        path_save (Path): MEDimage instance saving paths
    
    Returns:
        None.
    """

    series_description = MEDimg.series_description.translate({ord(ch): '-' for ch in '/\\ ()&:*'})
    name_id = MEDimg.patientID
    name_id = name_id.translate({ord(ch): '-' for ch in '/\\ ()&:*'})

    # final saving name
    name_complete = name_id + '__' + series_description + '.' + MEDimg.type + '.npy'
    
    # save
    with open(path_save / name_complete,'wb') as f: 
        pickle.dump(MEDimg, f)
