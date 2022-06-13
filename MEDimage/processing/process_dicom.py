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


def process_dicom(path_read: Path, 
                path_save: Path = None, 
                n_batch: int = 2) -> None:
    """
    This function reads the DICOM content of all the sub-folder tree of a
    starting directory defined by 'path_read'. It then organizes the data
    (files throughout the starting directory are associated by
    'SeriesInstanceUID') in the MEDimage class, and it finally computes the region of 
    interest (ROI) defined by an associated RTstruct, and save as well all the  REG, 
    RTdose and RTplan structures(if present in the sub-folder tree of the starting
    directory).
    All MEDimage classes hereby created are saved in 'path_save' with a name
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
        path_read (Path): String specifying the full path to the starting directory
            where the DICOM path to read are located.
        path_save (Path, optional): Full path to the directory where to
            save all the MEDimage classes (if specified) created by the current function.
        n_batch (int, optional): Numerical value specifying the number of batch to use in the
            parallel computations (use 0 for serial).
        nameSaveOption: (int, optional). If this argument is not present (default)
            all MEDimage classes will be saved with a filename
            defined by the 'Modality' field present in the
            DICOM header of their corresponding imaging volume. If
            multiple volumes of the same modality are present for
            the same 'patient_id', the different volumes will be
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
            the same modality and same 'patient_id' (this may lead
            to overwriting, but is faster if the user is sure
            that only one scan of each modality for each patient
            is present in 'path_read'.
            --> Options: 
                - No argument
                - 'folder'
                - 'modality'
                - 'seriesDescription'

    Returns:
        None.
    """

    def find_uid_cell_index(uid, cell): 
        """
        LAMBDA FUNCTION
        substitution of the Matlab function find_uid_cell_index
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
    stack_path_rs = []
    stack_series_rs = []
    stack_frame_rs = []

    # Cell of 'SeriesInstanceUID' of different imaging volumes (string).
    cell_series_id = []

    # Cell of 'FrameOfReferenceUID' of different imaging volumes (string).
    # Cell index is associated to the index of cell_series_id.
    cell_frame_id = []

    # Cell of paths to the different imaging volumes.
    # Cell index is associated to the index of cell_series_id.
    # Each cell in turn contains all the different path to the dicom
    # images of a given volume.
    cell_path_images = []

    # Cell of paths to the different RTstruct file (struct).
    # Cell index is associated to the index of cell_series_id.
    cell_path_rs = []

    # Cell of filenames for all created sData structures (string).
    # Cell index is associated to the index of cell_series_id.
    name_save = []

    # SCANNING ALL FOLDERS IN INITIAL DIRECTORY
    print('\n--> Scanning all folders in initial directory ... ', end='')
    p = Path(path_read)
    e_rglob = '*.[!xlsx,!xls,!py,!.DS_Store,!csv,!.,!txt,!..,!TXT,!npy,!m,!CT.npy]*'

    if path_read.is_dir():
        stack_folder_temp = list(p.rglob(e_rglob))
        stack_folder = [x for x in stack_folder_temp if not x.is_dir()]
    elif str(path_read).find('json') != -1:
        with open(path_read) as f:
            data = json.load(f)
            for value in data.values():
                stack_folder_temp = value
        directory_name = str(stack_folder_temp).replace("'", '').replace('[', '').replace(']', '')

        stack_folder = get_list_of_files(directory_name)

    for file in tqdm(stack_folder):
        if pydicom.misc.is_dicom(file):
            try:
                info = pydicom.dcmread(str(file))
                if info.Modality in ['MR', 'PT', 'CT']:
                    ind_series_id = find_uid_cell_index(
                        info.SeriesInstanceUID, cell_series_id)[0]
                    if ind_series_id == len(cell_series_id):  # New volume
                        cell_series_id = cell_series_id + [info.SeriesInstanceUID]
                        cell_frame_id = cell_frame_id + [info.FrameOfReferenceUID]
                        cell_path_images = cell_path_images + [[]]
                        cell_path_rs = cell_path_rs + [[]]
                        name_save = name_save + [[]]
                        name_save[ind_series_id] = info.Modality
                    cell_path_images[ind_series_id] = cell_path_images[ind_series_id] + [file]
                elif info.Modality == 'RTSTRUCT':
                    stack_path_rs = stack_path_rs + [file]
                    try:
                        series_uid = info.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[
                            0].RTReferencedSeriesSequence[0].SeriesInstanceUID
                    except:
                        series_uid = 'NotFound'
                    stack_series_rs = stack_series_rs + [series_uid]
                    try:
                        frame_uid = info.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
                    except:
                        frame_uid = info.FrameOfReferenceUID
                    stack_frame_rs = stack_frame_rs + [frame_uid]
            except Exception:
                print(f'Error while reading: {file}\n')
                continue
    print('DONE')

    print('--> Associating all RT objects to imaging volumes')
    # ASSOCIATING ALL RTSTRUCT TO IMAGING VOLUMES
    n_rs = len(stack_path_rs)
    if n_rs:
        for i in trange(0, n_rs):
            try:
                ind_series_id = find_uid_cell_index(stack_series_rs[i], cell_series_id)
                for n in range(len(ind_series_id)):
                    cell_path_rs[ind_series_id[n]] = (cell_path_rs[ind_series_id[n]] + [stack_path_rs[i]])
            except:
                ind_series_id = find_uid_cell_index(stack_frame_rs[i], cell_frame_id)
                for n in range(len(ind_series_id)):
                    cell_path_rs[ind_series_id[n]] = (cell_path_rs[ind_series_id[n]] + [stack_path_rs[i]])

    print('DONE')

    # READING ALL IMAGES TO CREATE ALL MEDimage classes
    print('--> Reading all DICOM objects to create MEDimage classes')
    n_scans = len(cell_path_images)

    if n_batch is None:
        n_batch = 1
    elif n_scans < n_batch:
        n_batch = n_scans

    # Distribute the first tasks to all workers
    ids = [pdsf.remote(cell_path_images[i], cell_path_rs[i], path_save)
           for i in range(n_batch)]

    nb_job_left = n_scans - n_batch

    for _ in trange(n_scans):
        _, not_ready = ray.wait(ids, num_returns=1)
        ids = not_ready

        # Distribute the remaining tasks
        if nb_job_left > 0:
            idx = n_scans - nb_job_left
            ids.extend([pdsf.remote(cell_path_images[idx], cell_path_rs[idx], path_save)])
            nb_job_left -= 1

    print('DONE')
