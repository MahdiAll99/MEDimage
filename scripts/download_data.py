import argparse
import os
import shutil
import zipfile

import wget


def main(no_sts: bool) -> None:
    """
    Downloads MEDimage data for testing, tutorials and demo and organizes it in the right folders.

    Args:
        no_sts (bool): if ``True`` will not download the STS data (large size). We recommend you set it to
            ``False`` so you can be able to run all the tutorials.
    
    Returns:
        None.
    """
    # download no-sts data (IBSI and glioma test data) 
    wget.download(
        "https://sandbox.zenodo.org/record/1094555/files/MEDimage-Dataset-No-STS.zip?download=1",
        out=os.getcwd())

    with zipfile.ZipFile(os.getcwd() + "/MEDimage-Dataset-No-STS.zip", 'r') as zip_ref:
        zip_ref.extractall(os.getcwd())
    # organize data in the right folders
    # ibsi tests data
    shutil.move(os.getcwd() + "/ibsi-test-data" + "/CTimage", os.getcwd() + "/notebooks" + "/ibsi" + "/data")
    shutil.move(os.getcwd() + "/ibsi-test-data" + "/Filters", os.getcwd() + "/notebooks" + "/ibsi" + "/data")
    shutil.move(os.getcwd() + "/ibsi-test-data" + "/Phantom", os.getcwd() + "/notebooks" + "/ibsi" + "/data")
    shutil.rmtree(os.getcwd() + "/ibsi-test-data")
    # tutorials data
    shutil.move(os.getcwd() + "/tutorials-data" + "/DICOM", os.getcwd() + "/notebooks" + "/tutorial" + "/data")
    shutil.move(os.getcwd() + "/tutorials-data" + "/IBSI-CT", os.getcwd() + "/notebooks" + "/tutorial" + "/data")
    shutil.move(os.getcwd() + "/tutorials-data" + "/NIfTI", os.getcwd() + "/notebooks" + "/tutorial" + "/data")
    shutil.rmtree(os.getcwd() + "/tutorials-data")

    # download sts data (multi-scans tutorial data)
    if not no_sts:
        wget.download(
            "https://sandbox.zenodo.org/record/1094555/files/MEDimage-Dataset-STS-McGill-001-005.zip?download=1",
            out=os.getcwd())
        # unzip data
        with zipfile.ZipFile(os.getcwd() + "/MEDimage-Dataset-STS-McGill-001-005.zip", 'r') as zip_ref:
            zip_ref.extractall(os.getcwd())
        # organize data in the right folder
        shutil.move(os.getcwd() + "/STS-McGill-001-005", 
                    os.getcwd() + "/notebooks" + "/tutorial" + "/data" + "/DICOM-STS")


if __name__ == "__main__":
    # setting up arguments:
    parser = argparse.ArgumentParser(description='Download dataset "\
        "for MEDimage package tests, tutorials and other demos.')
    parser.add_argument("--no-sts", default=False, action='store_false',
                    help="If specified, will not download STS data (Used in tutorials).")
    args = parser.parse_args()

    # main
    main(args.no_sts)
