import argparse
import os
import shutil
import zipfile

import wget


def main(full_sts: bool, subset: bool) -> None:
    """
    Downloads MEDimage data for testing, tutorials and demo and organizes it in the right folders.

    Args:
        full_sts (bool): if ``True`` will not download the STS data (large size).
        subset (bool): if ``True`` will download a subset of the data (small size).
    
    Returns:
        None.
    """
    # download no-sts data (IBSI and glioma test data)
    print("\n================ Downloading first part of data ================")
    try:
        wget.download(
        "https://sandbox.zenodo.org/record/1106515/files/MEDimage-Dataset-No-STS.zip?download=1",
        out=os.getcwd())
    except Exception as e:
        print("MEDimage-Dataset-No-STS.zip download failed, error:", e)
    
    # unzip data
    print("\n================ Extracting first part of data  ================")
    try:
        with zipfile.ZipFile(os.getcwd() + "/MEDimage-Dataset-No-STS.zip", 'r') as zip_ref:
            zip_ref.extractall(os.getcwd())
        # delete zip file after extraction
        os.remove(os.getcwd() + "/MEDimage-Dataset-No-STS.zip")
    except Exception as e:
        print("MEDimage-Dataset-No-STS.zip extraction failed, error:", e)
    
    # Organize data in the right folders
    # ibsi tests data organization
    print("\n================== Organizing data in folders ==================")
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
    if full_sts:
        # get data online
        print("\n================ Downloading second part of data ================")
        try:
            wget.download(
            "https://sandbox.zenodo.org/record/1106516/files/MEDimage-STS-Dataset.zip?download=1",
            out=os.getcwd())
            pass
        except Exception as e:
            print("MEDimage-STS-Dataset.zip download failed, error:", e)
        
        # unzip data
        print("\n================ Extracting second part of data  ================")
        try:
            with zipfile.ZipFile(os.getcwd() + "/MEDimage-STS-Dataset.zip", 'r') as zip_ref:
                zip_ref.extractall(os.getcwd())
                # remove zip file after extraction
            os.remove(os.getcwd() + "/MEDimage-STS-Dataset.zip")
        except Exception as e:
            print("MEDimage-STS-Dataset.zip extraction failed, error:", e)
        
        # organize data in the right folder
        print("\n================== Organizing data in folders  ==================")
        try:
            shutil.move(os.getcwd() + "/DICOM-STS-Organized", 
                    os.getcwd() + "/notebooks" + "/tutorial" + "/data" + "/DICOM-STS")
        except Exception as e:
            print("Failed to move DICOM-STS-Organized folder, error:", e)
    elif subset:
        # get data online
        print("\n================ Downloading second part of data ================")
        try:
            wget.download(
            "https://sandbox.zenodo.org/record/1106515/files/MEDimage-Dataset-STS-McGill-001-005.zip?download=1",
            out=os.getcwd())
            pass
        except Exception as e:
            print("MEDimage-Dataset-STS-McGill-001-005.zip download failed, error:", e)
        
        # unzip data
        print("\n================ Extracting second part of data  ================")
        try:
            with zipfile.ZipFile(os.getcwd() + "/MEDimage-Dataset-STS-McGill-001-005.zip", 'r') as zip_ref:
                zip_ref.extractall(os.getcwd())
                # remove zip file after extraction
            os.remove(os.getcwd() + "/MEDimage-Dataset-STS-McGill-001-005.zip")
        except Exception as e:
            print("MEDimage-Dataset-STS-McGill-001-005.zip extraction failed, error:", e)
        
        # organize data in the right folder
        print("\n================== Organizing data in folders  ==================")
        try:
            shutil.move(os.getcwd() + "/STS-McGill-001-005", 
                    os.getcwd() + "/notebooks" + "/tutorial" + "/data" + "/DICOM-STS")
        except Exception as e:
            print("Failed to move STS-McGill-001-005 folder, error:", e)


if __name__ == "__main__":
    # setting up arguments:
    parser = argparse.ArgumentParser(description='Download dataset "\
        "for MEDimage package tests, tutorials and other demos.')
    parser.add_argument("--full-sts", default=False, action='store_true',
                    help="If specified, will download the full STS data used in tutorials. Defaults to False.")
    parser.add_argument("--subset", default=True, action='store_true',
                    help="If specified, will only download a subset of the STS sataset. Defaults to True.")
    args = parser.parse_args()

    # main
    main(args.full_sts, args.subset)
