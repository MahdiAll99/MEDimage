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
    try:
        wget.download(
        "https://sandbox.zenodo.org/record/1094555/files/MEDimage-Dataset-No-STS.zip?download=1",
        out=os.getcwd())
    except Exception as e:
        print("MEDimage-Dataset-No-STS.zip download failed, error:", e)
    
    # unzip data
    try:
        with zipfile.ZipFile(os.getcwd() + "/MEDimage-Dataset-No-STS.zip", 'r') as zip_ref:
            zip_ref.extractall(os.getcwd())
        # delete zip file after extraction
        os.remove(os.getcwd() + "/MEDimage-Dataset-No-STS.zip")
    except Exception as e:
        print("MEDimage-Dataset-No-STS.zip extraction failed, error:", e)
    
    # Organize data in the right folders
    # ibsi tests data organization
    try:
        shutil.move(os.getcwd() + "/ibsi-test-data" + "/CTimage", 
                    os.getcwd() + "/notebooks" + "/ibsi" + "/data/CTimage")
    except Exception as e:
        print("Failed to move ibsi-test-data/CTimage folder, error:", e)
    try:
        shutil.move(os.getcwd() + "/ibsi-test-data" + "/Filters", 
                    os.getcwd() + "/notebooks" + "/ibsi" + "/data/Filters")
    except Exception as e:
        print("Failed to move ibsi-test-data/Filters folder, error:", e)
    try:
        shutil.move(os.getcwd() + "/ibsi-test-data" + "/Phantom", 
                    os.getcwd() + "/notebooks" + "/ibsi" + "/data/Phantom")
    except Exception as e:
        print("Failed to move ibsi-test-data/Phantom folder, error:", e)
    try:
        shutil.rmtree(os.getcwd() + "/ibsi-test-data")
    except Exception as e:
        print("Failed to delete ibsi-test-data folder, error:", e)
    
    # tutorials data organization
    try:
        shutil.move(os.getcwd() + "/tutorials-data" + "/DICOM", 
                    os.getcwd() + "/notebooks" + "/tutorial" + "/data/DICOM")
    except Exception as e:
        print("Failed to move tutorials-data/DICOM folder, error:", e)
    try:
        shutil.move(os.getcwd() + "/tutorials-data" + "/IBSI-CT", 
                    os.getcwd() + "/notebooks" + "/tutorial" + "/data/IBSI-CT")
    except Exception as e:
        print("Failed to move tutorials-data/IBSI-CT folder, error:", e)
    try:
        shutil.move(os.getcwd() + "/tutorials-data" + "/NIfTI", 
                    os.getcwd() + "/notebooks" + "/tutorial" + "/data/NIfTI")
    except Exception as e:
        print("Failed to move tutorials-data/NIfTI folder, error:", e)
    try:
        shutil.rmtree(os.getcwd() + "/tutorials-data")
    except Exception as e:
        print("Failed to remove tutorials-data folder, error:", e)

    # download sts data (multi-scans tutorial data)
    if not no_sts:
        # get data online
        try:
            wget.download(
            "https://sandbox.zenodo.org/record/1094555/files/MEDimage-Dataset-STS-McGill-001-005.zip?download=1",
            out=os.getcwd())
        except Exception as e:
            print("MEDimage-Dataset-STS-McGill-001-005.zip download failed, error:", e)
        
        # unzip data
        try:
            with zipfile.ZipFile(os.getcwd() + "/MEDimage-Dataset-STS-McGill-001-005.zip", 'r') as zip_ref:
                zip_ref.extractall(os.getcwd())
                # remove zip file after extraction
            os.remove(os.getcwd() + "/MEDimage-Dataset-STS-McGill-001-005.zip")
        except Exception as e:
            print("MEDimage-Dataset-STS-McGill-001-005.zip extraction failed, error:", e)
        
        # organize data in the right folder
        try:
            shutil.move(os.getcwd() + "/STS-McGill-001-005", 
                    os.getcwd() + "/notebooks" + "/tutorial" + "/data" + "/DICOM-STS")
        except Exception as e:
            print("Failed to move STS-McGill-001-005 folder, error:", e)


if __name__ == "__main__":
    # setting up arguments:
    parser = argparse.ArgumentParser(description='Download dataset "\
        "for MEDimage package tests, tutorials and other demos.')
    parser.add_argument("--no-sts", default=False, action='store_true',
                    help="If specified, will not download STS data (Used in tutorials).")
    args = parser.parse_args()

    # main
    main(args.no_sts)
