import argparse
import os
from pathlib import Path
from typing import Union

import pydicom


def main(path_dataset: Union[str, Path]) -> None:
    """Process the dataset.
    Args:
        path_dataset(Union[str, Path]): Path to the dataset.
    
    Returns:
        None.
    """
    path_dataset = Path(path_dataset)
    # get all sub-folders
    folders_patient_id = [f for f in path_dataset.iterdir() if f.is_dir()]
    for folder_patient_id in folders_patient_id:
        # get all sub-folders
        folders_imaging_summary = [f for f in folder_patient_id.iterdir() if f.is_dir()]
        for folder_imaging_summary in folders_imaging_summary:
            # get all dicom files
            dicom_files = os.listdir(folder_imaging_summary)
            for file in dicom_files:
                if pydicom.misc.is_dicom(folder_imaging_summary / file):
                    # update the dicom header fields
                    dicom_header = pydicom.dcmread(str(folder_imaging_summary / file))
                    dicom_header.PatientID = os.path.basename(folder_patient_id)
                    if 'SeriesDescription' in dicom_header:
                        dicom_header.SeriesDescription = os.path.basename(folder_imaging_summary)
                    dicom_header.save_as(str(folder_imaging_summary / file))

if __name__ == "__main__":
    # setting up arguments:
    parser = argparse.ArgumentParser(description='Re-organize dataset to follow MEDimage package conventions.')
    parser.add_argument("--path-dataset", required=True, help="Path to your dataset folder.")
    args = parser.parse_args()

    # main
    main(args.path_dataset)
