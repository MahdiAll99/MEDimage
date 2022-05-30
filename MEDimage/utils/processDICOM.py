import json
import os
import warnings
from pathlib import Path

import pydicom
import pydicom.errors
import pydicom.misc
import ray
from tqdm import tqdm, trange

from .process_dicom_scan_files import process_dicom_scan_files as pdsf

warnings.simplefilter("ignore")
ray.init(local_mode=True, include_dashboard=True)


def processDICOM(pathRead: Path, 
                pathSave: Path = None, 
                nBatch: int = 2) -> None:
    """
    This function reads the DICOM content of all the sub-folder tree of a
    starting directory defined by 'pathRead'. It then organizes the data
    (files throughout the starting directory are associated by
    'SeriesInstanceUID') in the MEDimage class, and it finally computes the region of 
    interest (ROI) defined by an associated RTstruct, and save as well all the  REG, 
    RTdose and RTplan structures(if present in the sub-folder tree of the starting
    directory).
    All MEDimage classes hereby created are saved in 'pathSave' with a name
    varying with the variable 'nameSaveOption'.

    DIFFERENTIATION/ASSOCIATION OF DICOM FILES: 1)imaging, 2)RTstruct, 3)REG,
    4)RTdose, 5)RTplan.
    1) Imaging volumes are differentiated by the 'SeriesInstanceUID' field
    2) Association between a RTstruct and an imaging volume is performed with:
       - 'ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID' field, or
       - 'FrameOfReferenceUID' field.
    3) Association between a REG and an imaging volume is performed with:
       - 'RegistrationSequence[1].FrameOfReferenceUID' field, or
       - 'FrameOfReferenceUID' field.
    4) Association between a RTdose and an imaging volume is performed with:
       - 'FrameOfReferenceUID' field.
    5) Association between a RTplan and an imaging volume is performed with:
       - 'FrameOfReferenceUID' field.

    If multiple imaging volumes have the same 'FrameOfReferenceUID' field,
    they will all be assigned a RTstruct, REG, RTdose and RTplan possessing
    that same field.

    Args:
        pathRead (Path): String specifying the full path to the starting directory
            where the DICOM path to read are located.
        pathSave (Path, optional): Full path to the directory where to
            save all the MEDimage classes (if specified) created by the current function.
        nBatch (int, optional): Numerical value specifying the number of batch to use in the
            parallel computations (use 0 for serial).
        nameSaveOption: (int, optional). If this argument is not present (default)
            all MEDimage classes will be saved with a filename
            defined by the 'Modality' field present in the
            DICOM header of their corresponding imaging volume. If
            multiple volumes of the same modality are present for
            the same 'PatientID', the different volumes will be
            numerated (e.g. CT1, CT2, CT3, etc.). If this argument
            is present and set to 'folder', all MEDimage classes
            will be saved with a filename defined by the name of
            the folder in which all DICOM files of a given imaging
            volume are located (useful to keep track of meaningful
            names). The 'folder' option assumes that all imaging
            DICOM files of a given imaging volume are present in
            the same directory. Note: the 'folder' argument may
            lead to overwriting of data if not organized properly.
            Finally, if this argument is present and set to
            'modality',  all MEDimage classes will be saved with a
            filename defined by the 'Modality' field in the DICOM
            header, but without enumerating multiple volumes of
            the same modality and same 'PatientID' (this may lead
            to overwriting, but is faster if the user is sure
            that only one scan of each modality for each patient
            is present in 'pathRead'.
            --> Options: 
                - No argument
                - 'folder'
                - 'modality'
                - 'seriesDescription'

    Returns:
        None.
    """

    def findUIDcellIndex(uid, cell): 
        """
        LAMBDA FUNCTION
        substitution of the Matlab function findUIDcellIndex
        If not is present in cellString, create a new position
        in the cell for the new UID"""
        return [len(cell)] if uid not in cell else[i for i, e in enumerate(cell) if e == uid]

    def get_list_of_files(dirName):
        list_of_file = os.listdir(dirName)
        all_files = list()

        for entry in list_of_file:
            full_path = os.path.join(dirName, entry)
            if os.path.isdir(full_path):
                all_files = all_files + get_list_of_files(full_path)
            else:
                all_files.append(full_path)

        return all_files

    # INITIALIZATION
    # Full path, FrameOfReferenceUID and References SeriesInstanceUID of the RTstruct files.
    stackPathRS = []
    stackSeriesRS = []
    stackFrameRS = []

    # Cell of 'SeriesInstanceUID' of different imaging volumes (string).
    cellSeriesID = []

    # Cell of 'FrameOfReferenceUID' of different imaging volumes (string).
    # Cell index is associated to the index of cellSeriesID.
    cellFrameID = []

    # Cell of paths to the different imaging volumes.
    # Cell index is associated to the index of cellSeriesID.
    # Each cell in turn contains all the different path to the dicom
    # images of a given volume.
    cellPathImages = []

    # Cell of paths to the different RTstruct file (struct).
    # Cell index is associated to the index of cellSeriesID.
    cellPathRS = []

    # Cell of filenames for all created sData structures (string).
    # Cell index is associated to the index of cellSeriesID.
    nameSave = []

    # SCANNING ALL FOLDERS IN INITIAL DIRECTORY
    print('\n--> Scanning all folders in initial directory ... ', end='')
    p = Path(pathRead)
    e_rglob = '*.[!xlsx,!xls,!py,!.DS_Store,!csv,!.,!txt,!..,!TXT,!npy,!m,!CT.npy]*'

    if pathRead.is_dir():
        stackFolderTemp = list(p.rglob(e_rglob))
        stackFolder = [x for x in stackFolderTemp if not x.is_dir()]
    elif str(pathRead).find('json') != -1:
        with open(pathRead) as f:
            data = json.load(f)
            for value in data.values():
                stackFolderTemp = value
        directory_name = str(stackFolderTemp).replace("'", '').replace('[', '').replace(']', '')

        stackFolder = get_list_of_files(directory_name)

    for file in tqdm(stackFolder):
        if pydicom.misc.is_dicom(file):
            try:
                info = pydicom.dcmread(str(file))
                if info.Modality in ['MR', 'PT', 'CT']:
                    indSeriesID = findUIDcellIndex(
                        info.SeriesInstanceUID, cellSeriesID)[0]
                    if indSeriesID == len(cellSeriesID):  # New volume
                        cellSeriesID = cellSeriesID + [info.SeriesInstanceUID]
                        cellFrameID = cellFrameID + [info.FrameOfReferenceUID]
                        cellPathImages = cellPathImages + [[]]
                        cellPathRS = cellPathRS + [[]]
                        nameSave = nameSave + [[]]
                        nameSave[indSeriesID] = info.Modality
                    cellPathImages[indSeriesID] = cellPathImages[indSeriesID] + [file]
                elif info.Modality == 'RTSTRUCT':
                    stackPathRS = stackPathRS + [file]
                    try:
                        seriesUID = info.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[
                            0].RTReferencedSeriesSequence[0].SeriesInstanceUID
                    except:
                        seriesUID = 'NotFound'
                    stackSeriesRS = stackSeriesRS + [seriesUID]
                    try:
                        frameUID = info.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
                    except:
                        frameUID = info.FrameOfReferenceUID
                    stackFrameRS = stackFrameRS + [frameUID]
            except Exception:
                print(f'Error while reading: {file}\n')
                continue
    print('DONE')

    print('--> Associating all RT objects to imaging volumes')
    # ASSOCIATING ALL RTSTRUCT TO IMAGING VOLUMES
    nRS = len(stackPathRS)
    if nRS:
        for i in trange(0, nRS):
            try:
                indSeriesID = findUIDcellIndex(stackSeriesRS[i], cellSeriesID)
                for n in range(len(indSeriesID)):
                    cellPathRS[indSeriesID[n]] = (cellPathRS[indSeriesID[n]] + [stackPathRS[i]])
            except:
                indSeriesID = findUIDcellIndex(stackFrameRS[i], cellFrameID)
                for n in range(len(indSeriesID)):
                    cellPathRS[indSeriesID[n]] = (cellPathRS[indSeriesID[n]] + [stackPathRS[i]])

    print('DONE')

    # READING ALL IMAGES TO CREATE ALL MEDimage classes
    print('--> Reading all DICOM objects to create MEDimage classes')
    nScans = len(cellPathImages)

    if nBatch is None:
        nBatch = 1
    elif nScans < nBatch:
        nBatch = nScans

    # Distribute the first tasks to all workers
    ids = [pdsf.remote(cellPathImages[i], cellPathRS[i], pathSave)
           for i in range(nBatch)]

    nb_job_left = nScans - nBatch

    for _ in trange(nScans):
        _, not_ready = ray.wait(ids, num_returns=1)
        ids = not_ready

        # Distribute the remaining tasks
        if nb_job_left > 0:
            idx = nScans - nb_job_left
            ids.extend([pdsf.remote(cellPathImages[idx], cellPathRS[idx], pathSave)])
            nb_job_left -= 1

    print('DONE')
