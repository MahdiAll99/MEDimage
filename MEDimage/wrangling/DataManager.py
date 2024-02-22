import json
import logging
import os
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import List, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import pydicom.errors
import pydicom.misc
import ray
from nilearn import image
from numpyencoder import NumpyEncoder
from tqdm import tqdm, trange

from ..MEDscan import MEDscan
from ..processing.compute_suv_map import compute_suv_map
from ..processing.interpolation import interp_volume
from ..processing.segmentation import get_roi_from_indexes
from ..utils.get_file_paths import get_file_paths
from ..utils.get_patient_names import get_patient_names
from ..utils.imref import imref3d
from ..utils.json_utils import load_json, save_json
from ..utils.save_MEDscan import save_MEDscan
from .ProcessDICOM import ProcessDICOM


class DataManager(object):
    """Reads all the raw data (DICOM, NIfTI) content and organizes it in instances of the MEDscan class."""


    @dataclass
    class DICOM(object):
        """DICOM data management class that will organize data during the conversion to MEDscan class process"""
        stack_series_rs: List
        stack_path_rs: List
        stack_frame_rs: List
        cell_series_id: List
        cell_path_rs: List
        cell_path_images: List
        cell_frame_rs: List
        cell_frame_id: List


    @dataclass
    class NIfTI(object):
        """NIfTI data management class that will organize data during the conversion to MEDscan class process"""
        stack_path_images: List
        stack_path_roi: List
        stack_path_all: List


    @dataclass
    class Paths(object):
        """Paths management class that will organize the paths used in the processing"""
        _path_to_dicoms: List
        _path_to_niftis: List
        _path_csv: Union[Path, str]
        _path_save: Union[Path, str]
        _path_save_checks: Union[Path, str]
        _path_pre_checks_settings: Union[Path, str]

    def __init__(
            self, 
            path_to_dicoms: List = [],
            path_to_niftis: List = [],
            path_csv: Union[Path, str] = None,
            path_save: Union[Path, str] = None,
            path_save_checks: Union[Path, str] = None,
            path_pre_checks_settings: Union[Path, str] = None,
            save: bool = False,
            n_batch: int = 2
    ) -> None:
        """Constructor of the class DataManager.

        Args:
            path_to_dicoms (Union[Path, str], optional): Full path to the starting directory
                where the DICOM data is located.
            path_to_niftis (Union[Path, str], optional): Full path to the starting directory
                where the NIfTI is located.
            path_csv (Union[Path, str], optional): Full path to the CSV file containing the scans info list.
            path_save (Union[Path, str], optional): Full path to the directory where to save all the MEDscan classes.
            path_save_checks(Union[Path, str], optional): Full path to the directory where to save all 
                the pre-radiomics checks analysis results.
            path_pre_checks_settings(Union[Path, str], optional): Full path to the JSON file of the pre-checks analysis
                parameters.
            save (bool, optional): True to save the MEDscan classes in `path_save`.
            n_batch (int, optional): Numerical value specifying the number of batch to use in the
                parallel computations (use 0 for serial computation).
        
        Returns:
            None
        """
        # Convert all paths to Pathlib.Path
        if path_to_dicoms:
            path_to_dicoms = Path(path_to_dicoms)
        if path_to_niftis:
            path_to_niftis = Path(path_to_niftis)
        if path_csv:
            path_csv = Path(path_csv)
        if path_save:
            path_save = Path(path_save)
        if path_save_checks:
            path_save_checks = Path(path_save_checks)
        if path_pre_checks_settings:
            path_pre_checks_settings = Path(path_pre_checks_settings)

        self.paths = self.Paths(
                path_to_dicoms,
                path_to_niftis,
                path_csv,
                path_save,
                path_save_checks,
                path_pre_checks_settings,
        )
        self.save = save
        self.n_batch = n_batch
        self.__dicom = self.DICOM(
                stack_series_rs=[],
                stack_path_rs=[],
                stack_frame_rs=[],
                cell_series_id=[],
                cell_path_rs=[],
                cell_path_images=[],
                cell_frame_rs=[],
                cell_frame_id=[]
        )
        self.__nifti = self.NIfTI(
                stack_path_images=[],
                stack_path_roi=[],
                stack_path_all=[]
        )
        self.path_to_objects = []
        self.summary = {}
        self.csv_data = None
        self.__studies = []
        self.__institutions = []
        self.__scans = []
        self.__warned = False

    def __find_uid_cell_index(self, uid: Union[str, List[str]], cell: List[str]) -> List: 
        """Finds the cell with the same `uid`. If not is present in `cell`, creates a new position
        in the `cell` for the new `uid`.

        Args:
            uid (Union[str, List[str]]):  Unique identifier of the Series to find.
            cell (List[str]): List of Unique identifiers of the Series.

        Returns:
            Union[List[str], str]: List or string of the uid  
        """
        return [len(cell)] if uid not in cell else[i for i, e in enumerate(cell) if e == uid]

    def __get_list_of_files(self, dir_name: str) -> List:
        """Gets all files in the given directory

        Args:
            dir_name (str): directory name

        Returns:
            List: List of all files in the directory
        """
        list_of_file = os.listdir(dir_name)
        all_files = list()
        for entry in list_of_file:
            full_path = os.path.join(dir_name, entry)
            if os.path.isdir(full_path):
                all_files = all_files + self.__get_list_of_files(full_path)
            else:
                all_files.append(full_path)

        return all_files

    def __get_MEDscan_name_save(self, medscan: MEDscan) -> str:
        """Returns the name that will be used to save the MEDscan instance, based on the values of the attributes.

        Args:
            medscan(MEDscan): A MEDscan class instance.
        
        Returns:
            str: String of the name save.
        """
        series_description = medscan.series_description.translate({ord(ch): '-' for ch in '/\\ ()&:*'})
        name_id = medscan.patientID.translate({ord(ch): '-' for ch in '/\\ ()&:*'})
        # final saving name
        name_complete = name_id + '__' + series_description + '.' + medscan.type + '.npy'
        return name_complete

    def __associate_rt_stuct(self) -> None:
        """Associates the imaging volumes to their mask using UIDs

        Returns:
            None
        """
        print('--> Associating all RT objects to imaging volumes')
        n_rs = len(self.__dicom.stack_path_rs)
        if n_rs:
            for i in trange(0, n_rs):
                try: # PUT ALL THE DICOM PATHS WITH THE SAME UID IN THE SAME PATH LIST
                    self.__dicom.stack_series_rs = list(set(self.__dicom.stack_series_rs))
                    ind_series_id = self.__find_uid_cell_index(
                                                        self.__dicom.stack_series_rs[i], 
                                                        self.__dicom.cell_series_id)
                    for n in range(len(ind_series_id)):
                        if ind_series_id[n] < len(self.__dicom.cell_path_rs):
                            self.__dicom.cell_path_rs[ind_series_id[n]] += [self.__dicom.stack_path_rs[i]]
                except:
                    ind_series_id = self.__find_uid_cell_index(
                                                        self.__dicom.stack_frame_rs[i], 
                                                        self.__dicom.cell_frame_id)
                    for n in range(len(ind_series_id)):
                        if ind_series_id[n] < len(self.__dicom.cell_path_rs):
                            self.__dicom.cell_path_rs[ind_series_id[n]] += [self.__dicom.stack_path_rs[i]]
        print('DONE')

    def __read_all_dicoms(self) -> None:
        """Reads all the dicom files in the all the paths of the attribute `_path_to_dicoms`

        Returns:
            None
        """
        # SCANNING ALL FOLDERS IN INITIAL DIRECTORY
        print('\n--> Scanning all folders in initial directory...', end='')
        p = Path(self.paths._path_to_dicoms)
        e_rglob = '*.dcm'

        # EXTRACT ALL FILES IN THE PATH TO DICOMS
        if self.paths._path_to_dicoms.is_dir():
            stack_folder_temp = list(p.rglob(e_rglob))
            stack_folder = [x for x in stack_folder_temp if not x.is_dir()]
        elif str(self.paths._path_to_dicoms).find('json') != -1:
            with open(self.paths._path_to_dicoms) as f:
                data = json.load(f)
                for value in data.values():
                    stack_folder_temp = value
            directory_name = str(stack_folder_temp).replace("'", '').replace('[', '').replace(']', '')
            stack_folder = self.__get_list_of_files(directory_name)
        else:
            raise ValueError("The given dicom folder path either doesn't exist or not a folder.")
        # READ ALL DICOM FILES AND UPDATE ATTRIBUTES FOR FURTHER PROCESSING
        for file in tqdm(stack_folder):
            if pydicom.misc.is_dicom(file):
                try:
                    info = pydicom.dcmread(str(file))
                    if info.Modality in ['MR', 'PT', 'CT']:
                        ind_series_id = self.__find_uid_cell_index(
                                                        info.SeriesInstanceUID, 
                                                        self.__dicom.cell_series_id)[0]
                        if ind_series_id == len(self.__dicom.cell_series_id):  # New volume
                            self.__dicom.cell_series_id = self.__dicom.cell_series_id + [info.SeriesInstanceUID]
                            self.__dicom.cell_frame_id += [info.FrameOfReferenceUID]
                            self.__dicom.cell_path_images += [[]]
                            self.__dicom.cell_path_rs = self.__dicom.cell_path_rs + [[]]
                        self.__dicom.cell_path_images[ind_series_id] += [file]
                    elif info.Modality == 'RTSTRUCT':
                        self.__dicom.stack_path_rs += [file]
                        try:
                            series_uid = info.ReferencedFrameOfReferenceSequence[
                                        0].RTReferencedStudySequence[
                                        0].RTReferencedSeriesSequence[
                                        0].SeriesInstanceUID
                        except:
                            series_uid = 'NotFound'
                        self.__dicom.stack_series_rs += [series_uid]
                        try:
                            frame_uid = info.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
                        except:
                            frame_uid = info.FrameOfReferenceUID
                        self.__dicom.stack_frame_rs += [frame_uid]

                except Exception as e:
                    print(f'Error while reading: {file}, error: {e}\n')
                    continue
        print('DONE')

        # ASSOCIATE ALL VOLUMES TO THEIR MASK
        self.__associate_rt_stuct()

    def process_all_dicoms(self) -> Union[List[MEDscan], None]:
        """This function reads the DICOM content of all the sub-folder tree of a starting directory defined by
        `path_to_dicoms`. It then organizes the data (files throughout the starting directory are associated by
        'SeriesInstanceUID') in the MEDscan class including the region of  interest (ROI) defined by an
        associated RTstruct. All MEDscan classes hereby created are saved in `path_save` with a name
        varying with every scan.

        Returns:
            List[MEDscan]: List of MEDscan instances.
        """
        ray.init(local_mode=True, include_dashboard=True)

        print('--> Reading all DICOM objects to create MEDscan classes')
        self.__read_all_dicoms()

        print('--> Processing DICOMs and creating MEDscan objects')
        n_scans = len(self.__dicom.cell_path_images)
        if self.n_batch is None:
            n_batch = 1
        elif n_scans < self.n_batch:
            n_batch = n_scans
        else:
            n_batch = self.n_batch

        # Distribute the first tasks to all workers
        pds = [ProcessDICOM(
                        self.__dicom.cell_path_images[i], 
                        self.__dicom.cell_path_rs[i], 
                        self.paths._path_save,
                        self.save)
            for i in range(n_batch)]
        
        ids = [pd.process_files() for pd in pds]

        # Update the path to the created instances
        for name_save in ray.get(ids):
            if self.paths._path_save:
                self.path_to_objects.append(str(self.paths._path_save / name_save))
            # Update processing summary
            if name_save.split('_')[0].count('-') >= 2:
                scan_type = name_save[name_save.find('__')+2 : name_save.find('.')]
                if name_save.split('-')[0] not in self.__studies:
                    self.__studies.append(name_save.split('-')[0])  # add new study
                if name_save.split('-')[1] not in self.__institutions:
                    self.__institutions.append(name_save.split('-')[1])  # add new study
                if name_save.split('-')[0] not in self.summary:
                    self.summary[name_save.split('-')[0]] = {}
                if name_save.split('-')[1] not  in self.summary[name_save.split('-')[0]]:
                    self.summary[name_save.split('-')[0]][name_save.split('-')[1]] = {}  # add new institution
                if scan_type not in self.__scans:
                    self.__scans.append(scan_type)
                if scan_type not in self.summary[name_save.split('-')[0]][name_save.split('-')[1]]:
                    self.summary[name_save.split('-')[0]][name_save.split('-')[1]][scan_type] = []
                if name_save not in self.summary[name_save.split('-')[0]][name_save.split('-')[1]][scan_type]:
                    self.summary[name_save.split('-')[0]][name_save.split('-')[1]][scan_type].append(name_save)
            else:
                if self.save:
                    logging.warning(f"The patient ID of the following file: {name_save} does not respect the MEDimage "\
                        "naming convention 'study-institution-id' (Ex: Glioma-TCGA-001)")

        nb_job_left = n_scans - n_batch

        # Distribute the remaining tasks
        for _ in trange(n_scans):
            _, ids = ray.wait(ids, num_returns=1)
            if nb_job_left > 0:
                idx = n_scans - nb_job_left
                pd = ProcessDICOM(
                        self.__dicom.cell_path_images[idx], 
                        self.__dicom.cell_path_rs[idx], 
                        self.paths._path_save,
                        self.save)
                ids.extend([pd.process_files()])
                nb_job_left -= 1

            # Update the path to the created instances
            for name_save in ray.get(ids):
                if self.paths._path_save:
                    self.path_to_objects.extend(str(self.paths._path_save / name_save))
                # Update processing summary
                if name_save.split('_')[0].count('-') >= 2:
                    scan_type = name_save[name_save.find('__')+2 : name_save.find('.')]
                    if name_save.split('-')[0] not in self.__studies:
                        self.__studies.append(name_save.split('-')[0])  # add new study
                    if name_save.split('-')[1] not in self.__institutions:
                        self.__institutions.append(name_save.split('-')[1])  # add new study
                    if name_save.split('-')[0] not in self.summary:
                        self.summary[name_save.split('-')[0]] = {}
                    if name_save.split('-')[1] not  in self.summary[name_save.split('-')[0]]:
                        self.summary[name_save.split('-')[0]][name_save.split('-')[1]] = {}  # add new institution
                    if scan_type not in self.__scans:
                        self.__scans.append(scan_type)
                    if scan_type not in self.summary[name_save.split('-')[0]][name_save.split('-')[1]]:
                        self.summary[name_save.split('-')[0]][name_save.split('-')[1]][scan_type] = []
                    if name_save not in self.summary[name_save.split('-')[0]][name_save.split('-')[1]][scan_type]:
                        self.summary[name_save.split('-')[0]][name_save.split('-')[1]][scan_type].append(name_save)
                else:
                    if self.save:
                        logging.warning(f"The patient ID of the following file: {name_save} does not respect the MEDimage "\
                            "naming convention 'study-institution-id' (Ex: Glioma-TCGA-001)")
        print('DONE')

    def __read_all_niftis(self) -> None:
        """Reads all files in the initial path and organizes other path to images and roi
        in the class attributes.

        Returns:
            None.
        """
        print('\n--> Scanning all folders in initial directory')
        if not self.paths._path_to_niftis:
            raise ValueError("The path to the niftis is not defined")
        p = Path(self.paths._path_to_niftis)
        e_rglob1 = '*.nii'
        e_rglob2 = '*.nii.gz'

        # EXTRACT ALL FILES IN THE PATH TO DICOMS
        if p.is_dir():
            self.__nifti.stack_path_all = list(p.rglob(e_rglob1))
            self.__nifti.stack_path_all.extend(list(p.rglob(e_rglob2)))
        else:
            raise TypeError(f"{p} must be a path to a directory")

        all_niftis = list(self.__nifti.stack_path_all)
        for i in trange(0, len(all_niftis)):
            if 'ROI' in all_niftis[i].name.split("."):
                self.__nifti.stack_path_roi.append(all_niftis[i])
            else:
                self.__nifti.stack_path_images.append(all_niftis[i])
        print('DONE')

    def __associate_roi_to_image(
            self,
            image_file: Union[Path, str], 
            medscan: MEDscan, 
            nifti: nib.Nifti1Image,
            path_roi_data: Path = None
        ) -> MEDscan:
        """Extracts all ROI data from the given path for the given patient ID and updates all class attributes with
        the new extracted data.

        Args:
            image_file(Union[Path, str]): Path to the ROI data.
            medscan (MEDscan): MEDscan class instance that will hold the data. 

        Returns:
            MEDscan: Returns a MEDscan instance with updated roi attributes.
        """
        image_file = Path(image_file)
        roi_index = 0

        if not path_roi_data:
            if not self.paths._path_to_niftis:
                raise ValueError("The path to the niftis is not defined")
            else:
                path_roi_data = self.paths._path_to_niftis

        for file in path_roi_data.glob('*.nii.gz'):
            _id = image_file.name.split("(")[0] # id is PatientID__ImagingScanName
            # Load the patient's ROI nifti files:
            if file.name.startswith(_id) and 'ROI' in file.name.split("."):
                roi = nib.load(file)
                roi = image.resample_to_img(roi, nifti, interpolation='nearest')
                roi_data = roi.get_fdata()
                roi_name = file.name[file.name.find("(") + 1 : file.name.find(")")]
                name_set = file.name[file.name.find("_") + 2 : file.name.find("(")]
                medscan.data.ROI.update_indexes(key=roi_index, indexes=np.nonzero(roi_data.flatten()))
                medscan.data.ROI.update_name_set(key=roi_index, name_set=name_set)
                medscan.data.ROI.update_roi_name(key=roi_index, roi_name=roi_name)
                roi_index += 1
        return medscan

    def __associate_spatialRef(self, nifti_file: Union[Path, str], medscan: MEDscan) -> MEDscan:
        """Computes the imref3d spatialRef using a NIFTI file and updates the spatialRef attribute.

        Args:
            nifti_file(Union[Path, str]): Path to the nifti data.
            medscan (MEDscan): MEDscan class instance that will hold the data.

        Returns:
            MEDscan: Returns a MEDscan instance with updated spatialRef attribute.
        """
        # Loading the nifti file :
        nifti = nib.load(nifti_file)
        nifti_data = medscan.data.volume.array

        # spatialRef Creation
        pixel_x = abs(nifti.affine[0, 0])
        pixel_y = abs(nifti.affine[1, 1])
        slices = abs(nifti.affine[2, 2])
        min_grid = nifti.affine[:3, 3] * [-1.0, -1.0, 1.0] # x and y are flipped
        min_x_grid = min_grid[0]
        min_y_grid = min_grid[1]
        min_z_grid = min_grid[2]
        size_image = np.shape(nifti_data)
        spatialRef = imref3d(size_image, abs(pixel_x), abs(pixel_y), abs(slices))
        spatialRef.XWorldLimits = (np.array(spatialRef.XWorldLimits) -
                                (spatialRef.XWorldLimits[0] -
                                    (min_x_grid-pixel_x/2))
                                ).tolist()
        spatialRef.YWorldLimits = (np.array(spatialRef.YWorldLimits) -
                                (spatialRef.YWorldLimits[0] -
                                    (min_y_grid-pixel_y/2))
                                ).tolist()
        spatialRef.ZWorldLimits = (np.array(spatialRef.ZWorldLimits) -
                                (spatialRef.ZWorldLimits[0] -
                                    (min_z_grid-slices/2))
                                ).tolist()

        # Converting the results into lists
        spatialRef.ImageSize = spatialRef.ImageSize.tolist()
        spatialRef.XIntrinsicLimits = spatialRef.XIntrinsicLimits.tolist()
        spatialRef.YIntrinsicLimits = spatialRef.YIntrinsicLimits.tolist()
        spatialRef.ZIntrinsicLimits = spatialRef.ZIntrinsicLimits.tolist()

        # update spatialRef in the volume sub-class
        medscan.data.volume.update_spatialRef(spatialRef)

        return medscan

    def __process_one_nifti(self, nifti_file: Union[Path, str], path_data) -> MEDscan:
        """
        Processes one NIfTI file to create a MEDscan class instance.

        Args:
            nifti_file (Union[Path, str]): Path to the NIfTI file.
            path_data (Union[Path, str]): Path to the data.
        
        Returns:
            MEDscan: MEDscan class instance.
        """
        medscan = MEDscan()
        medscan.patientID = os.path.basename(nifti_file).split("_")[0]
        medscan.type = os.path.basename(nifti_file).split(".")[-3]
        medscan.series_description = nifti_file.name[nifti_file.name.find('__') + 2: nifti_file.name.find('(')]
        medscan.format = "nifti"
        medscan.data.set_orientation(orientation="Axial")
        medscan.data.set_patient_position(patient_position="HFS")
        medscan.data.volume.array = nib.load(nifti_file).get_fdata()
        medscan.data.volume.scan_rot = None
        
        # Update spatialRef
        self.__associate_spatialRef(nifti_file, medscan)
        
        # Assiocate ROI
        medscan = self.__associate_roi_to_image(nifti_file, medscan, nib.load(nifti_file), path_data)

        return medscan
    
    def process_all(self) -> None:
        """Processes both DICOM & NIfTI content to create MEDscan classes
        """
        self.process_all_dicoms()
        self.process_all_niftis()

    def process_all_niftis(self) -> List[MEDscan]:
        """This function reads the NIfTI content of all the sub-folder tree of a starting directory. 
        It then organizes the data in the MEDscan class including the region of  interest (ROI)
        defined by an associated mask file. All MEDscan classes hereby created are saved in a specific path
        with a name specific name varying with every scan.

        Args:
            None.

        Returns:
            List[MEDscan]: List of MEDscan instances.
        """
        self.__read_all_niftis()
        print('--> Reading all NIfTI objects (imaging volumes & masks) to create MEDscan classes')
        for file in tqdm(self.__nifti.stack_path_images):
            # INITIALIZE MEDscan INSTANCE AND UPDATE ATTRIBUTES
            medscan = MEDscan()
            medscan.patientID = os.path.basename(file).split("_")[0]
            medscan.type = os.path.basename(file).split(".")[-3]
            medscan.series_description = file.name[file.name.find('__') + 2: file.name.find('(')]
            medscan.format = "nifti"
            medscan.data.set_orientation(orientation="Axial")
            medscan.data.set_patient_position(patient_position="HFS")
            medscan.data.volume.array = nib.load(file).get_fdata()
            
            # RAS to LPS
            #medscan.data.volume.convert_to_LPS()
            medscan.data.volume.scan_rot = None
            
            # Update spatialRef
            medscan = self.__associate_spatialRef(file, medscan)
            
            # Get ROI
            medscan = self.__associate_roi_to_image(file, medscan, nib.load(file))

            # SAVE MEDscan INSTANCE
            if self.save and self.paths._path_save:
                save_MEDscan(medscan, self.paths._path_save)
            
            # Update the path to the created instances
            name_save = self.__get_MEDscan_name_save(medscan)

            # Clear memory
            del medscan

            # Update the path to the created instances
            if self.paths._path_save:
                self.path_to_objects.append(str(self.paths._path_save / name_save))
            
            # Update processing summary
            if name_save.split('_')[0].count('-') >= 2:
                scan_type = name_save[name_save.find('__')+2 : name_save.find('.')]
                if name_save.split('-')[0] not in self.__studies:
                    self.__studies.append(name_save.split('-')[0])  # add new study
                if name_save.split('-')[1] not in self.__institutions:
                    self.__institutions.append(name_save.split('-')[1])  # add new institution
                if name_save.split('-')[0] not in self.summary:
                    self.summary[name_save.split('-')[0]] = {}  # add new study to summary
                if name_save.split('-')[1] not  in self.summary[name_save.split('-')[0]]:
                    self.summary[name_save.split('-')[0]][name_save.split('-')[1]] = {}  # add new institution
                if scan_type not in self.__scans:
                    self.__scans.append(scan_type)
                if scan_type not in self.summary[name_save.split('-')[0]][name_save.split('-')[1]]:
                    self.summary[name_save.split('-')[0]][name_save.split('-')[1]][scan_type] = []
                if name_save not in self.summary[name_save.split('-')[0]][name_save.split('-')[1]][scan_type]:
                    self.summary[name_save.split('-')[0]][name_save.split('-')[1]][scan_type].append(name_save)
            else:
                if self.save:
                    logging.warning(f"The patient ID of the following file: {name_save} does not respect the MEDimage "\
                        "naming convention 'study-institution-id' (Ex: Glioma-TCGA-001)")
        print('DONE')

    def update_from_csv(self, path_csv: Union[str, Path] = None) -> None:
        """Updates the class from a given CSV and summarizes the processed scans again according to it.

        Args:
            path_csv(optional, Union[str, Path]): Path to a csv file, if not given, will check
                for csv info in the class attributes.
        
        Returns:
            None
        """
        if not (path_csv or self.paths._path_csv):
            print('No csv provided, no updates will be made')
        else:
            if path_csv:
                self.paths._path_csv = path_csv
            # Extract roi type label from csv file name
            name_csv = self.paths._path_csv.name
            roi_type_label = name_csv[name_csv.find('_')+1 : name_csv.find('.')]

            # Create a dictionary
            csv_data = {}
            csv_data[roi_type_label] = pd.read_csv(self.paths._path_csv)
            self.csv_data = csv_data
            self.summarize()

    def summarize(self):
        """Creates and shows a summary of processed scans organized by study, institution, scan type and roi type

        Args:
            None
        Returns:
            None
        """
        def count_scans(summary):
            count = 0
            if type(summary) == dict:
                for study in summary:
                    if type(summary[study]) == dict:
                        for institution in summary[study]:
                            if type(summary[study][institution]) == dict:
                                for scan in self.summary[study][institution]:
                                    count += len(summary[study][institution][scan])
                            else:
                                count += len(summary[study][institution])
                    else:
                        count += len(summary[study])
            elif type(summary) == list:
                count = len(summary)
            return count

        summary_df = pd.DataFrame(columns=['study', 'institution', 'scan_type', 'roi_type', 'count'])

        for study in self.summary:
            summary_df = summary_df.append({
                                        'study': study,
                                        'institution': "",
                                        'scan_type': "",
                                        'roi_type': "",
                                        'count' : count_scans(self.summary)
                                        }, ignore_index=True)
            for institution in self.summary[study]:
                summary_df = summary_df.append({
                                            'study': study,
                                            'institution': institution,
                                            'scan_type': "",
                                            'roi_type': "",
                                            'count' : count_scans(self.summary[study][institution])
                                            }, ignore_index=True)
                for scan in self.summary[study][institution]:
                    summary_df = summary_df.append({
                                                'study': study,
                                                'institution': institution,
                                                'scan_type': scan,
                                                'roi_type': "",
                                                'count' : count_scans(self.summary[study][institution][scan])
                                                }, ignore_index=True)
                    if self.csv_data:
                        roi_count = 0
                        for roi_type in self.csv_data:
                            csv_table = pd.DataFrame(self.csv_data[roi_type])
                            csv_table['under'] = '_'
                            csv_table['dot'] = '.'
                            csv_table['npy'] = '.npy'
                            name_patients = (pd.Series(
                                csv_table[['PatientID', 'under', 'under',
                                        'ImagingScanName',
                                        'dot',
                                        'ImagingModality',
                                        'npy']].fillna('').values.tolist()).str.join('')).tolist()
                            for patient_id in self.summary[study][institution][scan]:
                                if patient_id in name_patients:
                                    roi_count += 1
                            summary_df = summary_df.append({
                                                'study': study,
                                                'institution': institution,
                                                'scan_type': scan,
                                                'roi_type': roi_type,
                                                'count' : roi_count
                                                }, ignore_index=True)
        print(summary_df.to_markdown(index=False))

    def __pre_radiomics_checks_dimensions(
        self,
        path_data: Union[Path, str] = None,
        wildcards_dimensions: List[str] = [],
        min_percentile: float = 0.05,
        max_percentile: float = 0.95,
        save: bool = False
        ) -> None:
        """Finds proper voxels dimension options for radiomics analyses for a group of scans

        Args:
            path_data (Path, optional): Path to the MEDscan objects, if not specified will use ``path_save`` from the 
                inner-class ``Paths`` in the current instance.
            wildcards_dimensions(List[str], optional): List of wildcards that determines the scans 
                that will be analyzed. You can learn more about wildcards in
                :ref:`this link <https://www.linuxtechtips.com/2013/11/how-wildcards-work-in-linux-and-unix.html>`.
            min_percentile (float, optional): Minimum percentile to use for the histograms. Defaults to 0.05.
            max_percentile (float, optional): Maximum percentile to use for the histograms. Defaults to 0.95.
            save (bool, optional): If True, will save the results in a json file. Defaults to False.
        
        Returns:
            None.
        """
        xy_dim = {
            "data": [],
            "mean": [],
            "median": [],
            "std": [],
            "min": [],
            "max": [],
            f"p{min_percentile}": [],
            f"p{max_percentile}": []
        }
        z_dim = {
            "data": [],
            "mean": [],
            "median": [],
            "std": [],
            "min": [],
            "max": [],
            f"p{min_percentile}": [],
            f"p{max_percentile}": []
        }
        if type(wildcards_dimensions) is str:
            wildcards_dimensions = [wildcards_dimensions]

        if len(wildcards_dimensions) == 0:
            print("Wildcard is empty, the pre-checks will be aborted")
            return

        # TODO: seperate by studies and scan type (MRscan, CTscan...)
        # TODO: Two summaries (df, list of names saves) -> 
        # name_save = name_save(ROI) : Glioma-Huashan-001__T1.MRscan.npy({GTV})
        file_paths = list()
        for w in range(len(wildcards_dimensions)):
            wildcard = wildcards_dimensions[w]
            if path_data:
                file_paths = get_file_paths(path_data, wildcard)
            elif self.paths._path_save:
                file_paths = get_file_paths(self.paths._path_save, wildcard)
            else:
                raise ValueError("Path data is invalid.")
            n_files = len(file_paths)
            xy_dim["data"] = np.zeros((n_files, 1))
            xy_dim["data"] = np.multiply(xy_dim["data"], np.nan)
            z_dim["data"] = np.zeros((n_files, 1))
            z_dim["data"] = np.multiply(z_dim["data"], np.nan)
            for f in tqdm(range(len(file_paths))):
                try:
                    if file_paths[f].name.endswith("nii.gz") or file_paths[f].name.endswith("nii"):
                        medscan = nib.load(file_paths[f])
                        xy_dim["data"][f] = medscan.header.get_zooms()[0]
                        z_dim["data"][f]  = medscan.header.get_zooms()[2]
                    else:
                        medscan = np.load(file_paths[f], allow_pickle=True)
                        xy_dim["data"][f] = medscan.data.volume.spatialRef.PixelExtentInWorldX
                        z_dim["data"][f]  = medscan.data.volume.spatialRef.PixelExtentInWorldZ
                except Exception as e:
                    print(e)

            # Running analysis
            xy_dim["data"] = np.concatenate(xy_dim["data"])
            xy_dim["mean"] = np.mean(xy_dim["data"][~np.isnan(xy_dim["data"])])
            xy_dim["median"] = np.median(xy_dim["data"][~np.isnan(xy_dim["data"])])
            xy_dim["std"] = np.std(xy_dim["data"][~np.isnan(xy_dim["data"])])
            xy_dim["min"] = np.min(xy_dim["data"][~np.isnan(xy_dim["data"])])
            xy_dim["max"] = np.max(xy_dim["data"][~np.isnan(xy_dim["data"])])
            xy_dim[f"p{min_percentile}"] = np.percentile(xy_dim["data"][~np.isnan(xy_dim["data"])], 
                                                        min_percentile)
            xy_dim[f"p{max_percentile}"] = np.percentile(xy_dim["data"][~np.isnan(xy_dim["data"])], 
                                                        max_percentile)
            z_dim["mean"] = np.mean(z_dim["data"][~np.isnan(z_dim["data"])])
            z_dim["median"] = np.median(z_dim["data"][~np.isnan(z_dim["data"])])
            z_dim["std"] = np.std(z_dim["data"][~np.isnan(z_dim["data"])])
            z_dim["min"] = np.min(z_dim["data"][~np.isnan(z_dim["data"])])
            z_dim["max"] = np.max(z_dim["data"][~np.isnan(z_dim["data"])])
            z_dim[f"p{min_percentile}"] = np.percentile(z_dim["data"][~np.isnan(z_dim["data"])], 
                                                        min_percentile)
            z_dim[f"p{max_percentile}"] = np.percentile(z_dim["data"][~np.isnan(z_dim["data"])], max_percentile)
            xy_dim["data"] = xy_dim["data"].tolist()
            z_dim["data"] = z_dim["data"].tolist()
            
            # Plotting xy-spacing data histogram
            df_xy = pd.DataFrame(xy_dim["data"], columns=['data'])
            del xy_dim["data"]  # no interest in keeping data (we only need statistics)
            ax = df_xy.hist(column='data')
            min_quant, max_quant, median = df_xy.quantile(min_percentile), df_xy.quantile(max_percentile), df_xy.median()
            for x in ax[0]:
                x.axvline(min_quant.data, linestyle=':', color='r', label=f"Min Percentile: {float(min_quant):.3f}")
                x.axvline(max_quant.data, linestyle=':', color='g', label=f"Max Percentile: {float(max_quant):.3f}")
                x.axvline(median.data, linestyle='solid', color='gold', label=f"Median: {float(median.data):.3f}")
                x.grid(False)
                plt.title(f"Voxels xy-spacing checks for {wildcard}")
                plt.legend()
                plt.show()
            
            # Plotting z-spacing data histogram
            df_z = pd.DataFrame(z_dim["data"], columns=['data'])
            del z_dim["data"]  # no interest in keeping data (we only need statistics)
            ax = df_z.hist(column='data')
            min_quant, max_quant, median = df_z.quantile(min_percentile), df_z.quantile(max_percentile), df_z.median()
            for x in ax[0]:
                x.axvline(min_quant.data, linestyle=':', color='r', label=f"Min Percentile: {float(min_quant):.3f}")
                x.axvline(max_quant.data, linestyle=':', color='g', label=f"Max Percentile: {float(max_quant):.3f}")
                x.axvline(median.data, linestyle='solid', color='gold', label=f"Median: {float(median.data):.3f}")
                x.grid(False)
                plt.title(f"Voxels z-spacing checks for {wildcard}")
                plt.legend()
                plt.show()
                
            # Saving files using wildcard for name
            if save:
                wildcard = str(wildcard).replace('*', '').replace('.npy', '.json')
                save_json(self.paths._path_save_checks / ('xyDim_' + wildcard), xy_dim, cls=NumpyEncoder)
                save_json(self.paths._path_save_checks / ('zDim_' + wildcard), z_dim, cls=NumpyEncoder)

    def __pre_radiomics_checks_window(
        self,
        path_data: Union[str, Path] = None,
        wildcards_window: List = [], 
        path_csv: Union[str, Path] = None,
        min_percentile: float = 0.05,
        max_percentile: float = 0.95,
        bin_width: int = 0,
        hist_range: list = [],
        nifti: bool = True,
        save: bool = False
        ) -> None:
        """Finds proper re-segmentation ranges options for radiomics analyses for a group of scans

        Args:
            path_data (Path, optional): Path to the MEDscan objects, if not specified will use ``path_save`` from the 
                inner-class ``Paths`` in the current instance.
            wildcards_window(List[str], optional): List of wildcards that determines the scans 
                that will be analyzed. You can learn more about wildcards in
                :ref:`this link <https://www.linuxtechtips.com/2013/11/how-wildcards-work-in-linux-and-unix.html>`.
            path_csv(Union[str, Path], optional): Path to a csv file containing a list of the scans that will be
                analyzed (a CSV file for a single ROI type).
            min_percentile (float, optional): Minimum percentile to use for the histograms. Defaults to 0.05.
            max_percentile (float, optional): Maximum percentile to use for the histograms. Defaults to 0.95.
            bin_width(int, optional): Width of the bins for the histograms. If not provided, will use the 
                default number of bins in the method 
                :ref:`pandas.DataFrame.hist <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.hist.html>`: 10 bins.
            hist_range(list, optional): Range of the histograms. If empty, will use the minimum and maximum values.
            nifti(bool, optional): If True, will use the NIfTI files, otherwise will use the numpy files.
            save (bool, optional): If True, will save the results in a json file. Defaults to False.
        
        Returns:
            None.
        """
        if type(wildcards_window) is str:
            wildcards_window = [wildcards_window]

        if len(wildcards_window) == 0:
            print("Wilcards is empty")
            return
        if path_csv:
            self.paths._path_csv = Path(path_csv)
        roi_table = pd.read_csv(self.paths._path_csv)
        if nifti:
            roi_table['under'] = '_'
            roi_table['dot'] = '.'
            roi_table['roi_label'] = 'GTV'
            roi_table['oparenthesis'] = '('
            roi_table['cparenthesis'] = ')'
            roi_table['ext'] = '.nii.gz'
            patient_names = (pd.Series(
                roi_table[['PatientID', 'under', 'under',
                        'ImagingScanName',
                        'oparenthesis',
                        'roi_label',
                        'cparenthesis',
                        'dot',
                        'ImagingModality',
                        'ext']].fillna('').values.tolist()).str.join('')).tolist()
        else:
            roi_names = [[], [], []]
            roi_names[0] = roi_table['PatientID']
            roi_names[1] = roi_table['ImagingScanName']
            roi_names[2] = roi_table['ImagingModality']
            patient_names = get_patient_names(roi_names)
        for w in range(len(wildcards_window)):
            temp_val = []
            temp = []
            file_paths = []
            roi_data= {
                "data": [],
                "mean": [],
                "median": [],
                "std": [],
                "min": [],
                "max": [],
                f"p{min_percentile}": [],
                f"p{max_percentile}": []
            }
            wildcard = wildcards_window[w]
            if path_data:
                file_paths = get_file_paths(path_data, wildcard)
            elif self.paths._path_save:
                path_data = self.paths._path_save
                file_paths = get_file_paths(self.paths._path_save, wildcard)
            else:
                raise ValueError("Path data is invalid.")
            n_files = len(file_paths)
            i = 0
            for f in tqdm(range(n_files)):
                file = file_paths[f]
                _, filename = os.path.split(file)
                filename, ext = os.path.splitext(filename)
                patient_name = filename + ext
                try:
                    if file.name.endswith('nii.gz') or file.name.endswith('nii'):
                        medscan = self.__process_one_nifti(file, path_data)
                    else:
                        medscan = np.load(file, allow_pickle=True)
                        if re.search('PTscan', wildcard) and medscan.format != 'nifti':
                            medscan.data.volume.array = compute_suv_map(
                                                        np.double(medscan.data.volume.array), 
                                                        medscan.dicomH[2])
                    patient_names = pd.Index(patient_names)
                    ind_roi = patient_names.get_loc(patient_name)
                    name_roi = roi_table.loc[ind_roi][3]
                    vol_obj_init, roi_obj_init = get_roi_from_indexes(medscan, name_roi, 'box')
                    temp = vol_obj_init.data[roi_obj_init.data == 1]
                    temp_val.append(len(temp))
                    roi_data["data"].append(np.zeros(shape=(n_files, temp_val[i])))
                    roi_data["data"][i] = temp
                    i+=1
                    del medscan
                    del vol_obj_init
                    del roi_obj_init
                except Exception as e:
                    print(f"Problem with patient {patient_name}, error: {e}")
            
            roi_data["data"] = np.concatenate(roi_data["data"])
            roi_data["mean"] = np.mean(roi_data["data"][~np.isnan(roi_data["data"])])
            roi_data["median"] = np.median(roi_data["data"][~np.isnan(roi_data["data"])])
            roi_data["std"] = np.std(roi_data["data"][~np.isnan(roi_data["data"])])
            roi_data["min"] = np.min(roi_data["data"][~np.isnan(roi_data["data"])])
            roi_data["max"] = np.max(roi_data["data"][~np.isnan(roi_data["data"])])
            roi_data[f"p{min_percentile}"] = np.percentile(roi_data["data"][~np.isnan(roi_data["data"])], 
                                                        min_percentile)
            roi_data[f"p{max_percentile}"] = np.percentile(roi_data["data"][~np.isnan(roi_data["data"])], 
                                                        max_percentile)
            
            # Set bin width if not provided
            if bin_width != 0:
                if hist_range:
                    nb_bins = (round(hist_range[1]) - round(hist_range[0])) // bin_width
                else:
                    nb_bins = (round(roi_data["max"]) - round(roi_data["min"])) // bin_width
            else:
                nb_bins = 10
                if hist_range:
                    bin_width = int((round(hist_range[1]) - round(hist_range[0])) // nb_bins)
                else:
                    bin_width = int((round(roi_data["max"]) - round(roi_data["min"])) // nb_bins)
            nb_bins = int(nb_bins)

            # Set histogram range if not provided
            if not hist_range:
                hist_range = (roi_data["min"], roi_data["max"])

           # re-segment data according to histogram range
            roi_data["data"] = roi_data["data"][(roi_data["data"] > hist_range[0]) & (roi_data["data"] < hist_range[1])]
            df_data = pd.DataFrame(roi_data["data"], columns=['data'])
            del roi_data["data"]  # no interest in keeping data (we only need statistics)

            # Plot histogram
            ax = df_data.hist(column='data', bins=nb_bins, range=(hist_range[0], hist_range[1]), edgecolor='black')
            min_quant, max_quant= df_data.quantile(min_percentile), df_data.quantile(max_percentile)
            for x in ax[0]:
                x.axvline(min_quant.data, linestyle=':', color='r', label=f"{min_percentile*100}% Percentile: {float(min_quant):.3f}")
                x.axvline(max_quant.data, linestyle=':', color='g', label=f"{max_percentile*100}% Percentile: {float(max_quant):.3f}")
                x.grid(False)
                x.xaxis.set_ticks(np.arange(hist_range[0], hist_range[1], bin_width, dtype=int))
                x.set_xticklabels(x.get_xticks(), rotation=45)
                x.xaxis.set_tick_params(pad=15)
                plt.title(f"Intensity range checks for {wildcard}, bw={bin_width}")
                plt.legend()
                plt.show()
            
            # save final checks
            if save:
                wildcard = str(wildcard).replace('*', '').replace('.npy', '.json')
                save_json(self.paths._path_save_checks / ('roi_data_' + wildcard), roi_data, cls=NumpyEncoder)

    def pre_radiomics_checks(self,
                            path_data: Union[str, Path] = None,
                            wildcards_dimensions: List = [],
                            wildcards_window: List = [],
                            path_csv: Union[str, Path] = None,
                            min_percentile: float = 0.05,
                            max_percentile: float = 0.95,
                            bin_width: int = 0,
                            hist_range: list = [],
                            nifti: bool = False,
                            save: bool = False) -> None:
        """Finds proper dimension and re-segmentation ranges options for radiomics analyses. 

        The resulting files from this method can then be analyzed and used to set up radiomics 
        parameters options in computation methods.

        Args:
            path_data (Path, optional): Path to the MEDscan objects, if not specified will use ``path_save`` from the 
                inner-class ``Paths`` in the current instance.
            wildcards_dimensions(List[str], optional): List of wildcards that determines the scans 
                that will be analyzed. You can learn more about wildcards in
                `this link <https://www.linuxtechtips.com/2013/11/how-wildcards-work-in-linux-and-unix.html>`_.
            wildcards_window(List[str], optional): List of wildcards that determines the scans 
                that will be analyzed. You can learn more about wildcards in
                `this link <https://www.linuxtechtips.com/2013/11/how-wildcards-work-in-linux-and-unix.html>`_.
            path_csv(Union[str, Path], optional): Path to a csv file containing a list of the scans that will be
                analyzed (a CSV file for a single ROI type).
            min_percentile (float, optional): Minimum percentile to use for the histograms. Defaults to 0.05.
            max_percentile (float, optional): Maximum percentile to use for the histograms. Defaults to 0.95.
            bin_width(int, optional): Width of the bins for the histograms. If not provided, will use the 
                default number of bins in the method 
                :ref:`pandas.DataFrame.hist <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.hist.html>`: 10 bins.
            hist_range(list, optional): Range of the histograms. If empty, will use the minimum and maximum values.
            nifti (bool, optional): Set to True if the scans are nifti files. Defaults to False.
            save (bool, optional): If True, will save the results in a json file. Defaults to False.

        Returns:
            None
        """
        # Initialization
        path_study = Path.cwd()

        # Load params
        if not self.paths._path_pre_checks_settings:
            if not wildcards_dimensions or not wildcards_window:
                raise ValueError("path to pre-checks settings is None.\
                    wildcards_dimensions and wildcards_window need to be specified")
        else:
            settings = self.paths._path_pre_checks_settings
            settings = load_json(settings)
            settings = settings['pre_radiomics_checks']  

            # Setting up paths
            if 'path_save_checks' in settings and settings['path_save_checks']:
                self.paths._path_save_checks = Path(settings['path_save_checks']) 
            if 'path_csv' in settings and settings['path_csv']:
                self.paths._path_csv = Path(settings['path_csv']) 

            # Wildcards of groups of files to analyze for dimensions in path_data.
            # See for example: https://www.linuxtechtips.com/2013/11/how-wildcards-work-in-linux-and-unix.html
            # Keep the cell empty if no dimension checks are to be performed.
            if not wildcards_dimensions:
                wildcards_dimensions = []
                for i in range(len(settings['wildcards_dimensions'])):
                    wildcards_dimensions.append(settings['wildcards_dimensions'][i])

            # ROI intensity window checks params
            if not wildcards_window:
                wildcards_window = []
                for i in range(len(settings['wildcards_window'])):
                    wildcards_window.append(settings['wildcards_window'][i])

        # PRE-RADIOMICS CHECKS
        if not self.paths._path_save_checks:
            if (path_study / 'checks').exists():
                self.paths._path_save_checks = Path(path_study / 'checks')
            else:
                os.mkdir(path_study / 'checks')
                self.paths._path_save_checks = Path(path_study / 'checks')
        else:
            if self.paths._path_save_checks.name != 'checks':
                if (self.paths._path_save_checks / 'checks').exists():
                    self.paths._path_save_checks /= 'checks'
                else:
                    os.mkdir(self.paths._path_save_checks / 'checks')
                    self.paths._path_save_checks = Path(self.paths._path_save_checks / 'checks')

        # Initializing plotting params
        plt.rcParams["figure.figsize"] = (20,20)
        plt.rcParams.update({'font.size': 22})
        
        start = time()
        print('\n\n************************* PRE-RADIOMICS CHECKS *************************', end='')

        # 1. PRE-RADIOMICS CHECKS -- DIMENSIONS
        start1 = time()
        print('\n--> PRE-RADIOMICS CHECKS -- DIMENSIONS ... ', end='')
        self.__pre_radiomics_checks_dimensions(
                                        path_data, 
                                        wildcards_dimensions, 
                                        min_percentile, 
                                        max_percentile,
                                        save)
        print('DONE', end='')
        time1 = f"{time() - start1:.2f}"
        print(f'\nElapsed time: {time1} sec', end='')

        # 2. PRE-RADIOMICS CHECKS - WINDOW
        start2 = time()
        print('\n\n--> PRE-RADIOMICS CHECKS -- WINDOW ... \n', end='')
        self.__pre_radiomics_checks_window(
                                        path_data, 
                                        wildcards_window, 
                                        path_csv,
                                        min_percentile, 
                                        max_percentile,
                                        bin_width,
                                        hist_range,
                                        nifti,
                                        save)
        print('DONE', end='')
        time2 = f"{time() - start2:.2f}"
        print(f'\nElapsed time: {time2} sec', end='')

        time_elapsed = f"{time() - start:.2f}"
        print(f'\n\n--> TOTAL TIME FOR PRE-RADIOMICS CHECKS: {time_elapsed} seconds')
        print('-------------------------------------------------------------------------------------')

    def perform_mr_imaging_summary(self, 
                                wildcards_scans: List[str],
                                path_data: Path = None,
                                path_save_checks: Path = None,
                                min_percentile: float = 0.05,
                                max_percentile: float = 0.95
                                ) -> None:
        """
        Summarizes MRI imaging acquisition parameters. Plots summary histograms
        for different dimensions and saves all acquisition parameters locally in JSON files.

        Args:
            wildcards_scans (List[str]): List of wildcards that determines the scans 
                that will be analyzed (Only MRI scans will be analyzed). You can learn more about wildcards in
                `this link <https://www.linuxtechtips.com/2013/11/how-wildcards-work-in-linux-and-unix.html>`_.
                For example: ``[\"STS*.MRscan.npy\"]``.
            path_data (Path, optional): Path to the MEDscan objects, if not specified will use ``path_save`` from the 
                inner-class ``Paths`` in the current instance.
            path_save_checks (Path, optional): Path where to save the checks, if not specified will use the one 
                in the current instance.
            min_percentile (float, optional): Minimum percentile to use for the histograms. Defaults to 0.05.
            max_percentile (float, optional): Maximum percentile to use for the histograms. Defaults to 0.95.
        
        Returns:
            None.
        """
        # Initializing data structures
        class param:
            dates = []
            manufacturer = []
            scanning_sequence = []
            class years:
                data = []

            class fieldStrength:
                data = []

            class repetitionTime:
                data = []

            class echoTime:
                data = []

            class inversionTime:
                data = []

            class echoTrainLength:
                data = []

            class flipAngle:
                data = []

            class numberAverages:
                data = []

            class xyDim:
                data = []

            class zDim:
                data = []
        
        if len(wildcards_scans) == 0:
            print('wildcards_scans is empty')

        # wildcards checks:
        no_mr_scan = True
        for wildcard in wildcards_scans:
            if 'MRscan' in wildcard:
                no_mr_scan = False
        if no_mr_scan:
            raise ValueError(f"wildcards: {wildcards_scans} does not include MR scans. (Only MR scans are supported)")
        
        # Initialization
        if path_data is None:
            if self.paths._path_save:
                path_data = Path(self.paths._path_save)
            else:
                print("No path to data was given and path save is None.")
                return 0
        
        if not path_save_checks:
            if self.paths._path_save_checks:
                path_save_checks = Path(self.paths._path_save_checks)
            else:
                if (Path(os.getcwd()) / "checks").exists():
                    path_save_checks = Path(os.getcwd()) / "checks"
                else:
                    path_save_checks = (Path(os.getcwd()) / "checks").mkdir()
        # Looping through all the different wildcards
        for i in tqdm(range(len(wildcards_scans))):
            wildcard = wildcards_scans[i]
            file_paths = get_file_paths(path_data, wildcard)
            n_files = len(file_paths)
            param.dates = np.zeros(n_files)
            param.years.data = np.zeros((n_files, 1))
            param.years.data = np.multiply(param.years.data, np.NaN)
            param.manufacturer = [None] * n_files
            param.scanning_sequence = [None] * n_files
            param.fieldStrength.data = np.zeros((n_files, 1))
            param.fieldStrength.data = np.multiply(param.fieldStrength.data, np.NaN)
            param.repetitionTime.data = np.zeros((n_files, 1))
            param.repetitionTime.data = np.multiply(param.repetitionTime.data, np.NaN)
            param.echoTime.data = np.zeros((n_files, 1))
            param.echoTime.data = np.multiply(param.echoTime.data, np.NaN)
            param.inversionTime.data = np.zeros((n_files, 1))
            param.inversionTime.data = np.multiply(param.inversionTime.data, np.NaN)
            param.echoTrainLength.data = np.zeros((n_files, 1))
            param.echoTrainLength.data = np.multiply(param.echoTrainLength.data, np.NaN)
            param.flipAngle.data = np.zeros((n_files, 1))
            param.flipAngle.data = np.multiply(param.flipAngle.data, np.NaN)
            param.numberAverages.data = np.zeros((n_files, 1))
            param.numberAverages.data = np.multiply(param.numberAverages.data, np.NaN)
            param.xyDim.data = np.zeros((n_files, 1))
            param.xyDim.data = np.multiply(param.xyDim.data, np.NaN)
            param.zDim.data = np.zeros((n_files, 1))
            param.zDim.data = np.multiply(param.zDim.data, np.NaN)
            
            # Loading and recording data
            for f in tqdm(range(n_files)):
                file = file_paths[f]

                #Open file for warning
                try:
                    warn_file = open(path_save_checks / 'imaging_summary_mr_warnings.txt', 'a')
                except IOError:
                    print("Could not open warning file")

                # Loading Data
                try:
                    print(f'\nCurrently working on: {file}', file = warn_file)
                    with open(path_data / file, 'rb') as fe: medscan = pickle.load(fe)

                    # Example of DICOM header
                    info = medscan.dicomH[1]
                    # Recording dates (info.AcquistionDates)
                    try:
                        param.dates[f] = info.AcquisitionDate
                    except AttributeError:
                        param.dates[f] = info.StudyDate
                    # Recording years
                    try:
                        y = str(param.dates[f])  # Only the first four characters represent the years
                        param.years.data[f] = y[0:4]
                    except Exception as e:
                        print(f'Cannot read years of: {file}. Error: {e}', file = warn_file)
                    # Recording manufacturers
                    try:
                        param.manufacturer[f] = info.Manufacturer
                    except Exception as e:
                        print(f'Cannot read manufacturer of: {file}. Error: {e}', file = warn_file)
                    # Recording scanning sequence
                    try:
                        param.scanning_sequence[f] = info.scanning_sequence
                    except Exception as e:
                        print(f'Cannot read scanning sequence of: {file}. Error: {e}', file = warn_file)
                    # Recording field strength
                    try:
                        param.fieldStrength.data[f] = info.MagneticFieldStrength
                    except Exception as e:
                        print(f'Cannot read field strength of: {file}. Error: {e}', file = warn_file)
                    # Recording repetition time
                    try:
                        param.repetitionTime.data[f] = info.RepetitionTime
                    except Exception as e:
                        print(f'Cannot read repetition time of: {file}. Error: {e}', file = warn_file)
                    # Recording echo time
                    try:
                        param.echoTime.data[f] = info.EchoTime
                    except Exception as e:
                        print(f'Cannot read echo time of: {file}. Error: {e}', file = warn_file)
                    # Recording inversion time
                    try:
                        param.inversionTime.data[f] = info.InversionTime
                    except Exception as e:
                        print(f'Cannot read inversion time of: {file}. Error: {e}', file = warn_file)
                    # Recording echo train length
                    try:
                        param.echoTrainLength.data[f] = info.EchoTrainLength
                    except Exception as e:
                        print(f'Cannot read echo train length of: {file}. Error: {e}', file = warn_file)
                    # Recording flip angle
                    try:
                        param.flipAngle.data[f] = info.FlipAngle
                    except Exception as e:
                        print(f'Cannot read flip angle of: {file}. Error: {e}', file = warn_file)
                    # Recording number of averages
                    try:
                        param.numberAverages.data[f] = info.NumberOfAverages
                    except Exception as e:
                        print(f'Cannot read number averages of: {file}. Error: {e}', file = warn_file)
                    # Recording xy spacing
                    try:
                        param.xyDim.data[f] = medscan.data.volume.spatialRef.PixelExtentInWorldX
                    except Exception as e:
                        print(f'Cannot read x spacing of: {file}. Error: {e}', file = warn_file)
                    # Recording z spacing
                    try:
                        param.zDim.data[f] = medscan.data.volume.spatialRef.PixelExtentInWorldZ
                    except Exception as e:
                        print(f'Cannot read z spacing of: {file}', file = warn_file)
                except Exception as e:
                    print(f'Cannot read file: {file}. Error: {e}', file = warn_file)

                warn_file.close()

            # Summarize data
                # Summarizing years
            df_years = pd.DataFrame(param.years.data, 
                                    columns=['years']).describe(percentiles=[min_percentile, max_percentile], 
                                    include='all')
                # Summarizing field strength
            df_fs = pd.DataFrame(param.fieldStrength.data, 
                                columns=['fieldStrength']).describe(percentiles=[min_percentile, max_percentile], 
                                include='all')
                # Summarizing  repetition time
            df_rt = pd.DataFrame(param.repetitionTime.data, 
                                columns=['repetitionTime']).describe(percentiles=[min_percentile, max_percentile], 
                                include='all')
                # Summarizing echo time
            df_et = pd.DataFrame(param.echoTime.data, 
                                columns=['echoTime']).describe(percentiles=[min_percentile, max_percentile], 
                                include='all')
                # Summarizing inversion time
            df_it = pd.DataFrame(param.inversionTime.data, 
                                columns=['inversionTime']).describe(percentiles=[min_percentile, max_percentile], 
                                include='all')
                # Summarizing echo train length
            df_etl = pd.DataFrame(param.echoTrainLength.data, 
                                columns=['echoTrainLength']).describe(percentiles=[min_percentile, max_percentile], 
                                include='all')
                # Summarizing flip  angle
            df_fa = pd.DataFrame(param.flipAngle.data, 
                                columns=['flipAngle']).describe(percentiles=[min_percentile, max_percentile], 
                                include='all')
                # Summarizing number of  averages
            df_na = pd.DataFrame(param.numberAverages.data, 
                                columns=['numberAverages']).describe(percentiles=[min_percentile, max_percentile], 
                                include='all')
                # Summarizing xy-spacing
            df_xy = pd.DataFrame(param.xyDim.data, 
                                columns=['xyDim'])
                # Summarizing z-spacing
            df_z = pd.DataFrame(param.zDim.data, 
                                columns=['zDim'])

            # Plotting xy-spacing histogram
            ax = df_xy.hist(column='xyDim')
            min_quant, max_quant, average = df_xy.quantile(min_percentile), df_xy.quantile(max_percentile), param.xyDim.data.mean()
            for x in ax[0]:
                x.axvline(min_quant.xyDim, linestyle=':', color='r', label=f"Min Percentile: {float(min_quant):.3f}")
                x.axvline(max_quant.xyDim, linestyle=':', color='g', label=f"Max Percentile: {float(max_quant):.3f}")
                x.axvline(average, linestyle='solid', color='gold', label=f"Average: {float(average):.3f}")
                x.grid(False)
                plt.title(f"MR xy-spacing imaging summary for {wildcard}")
                plt.legend()
                plt.show()
            
            # Plotting z-spacing histogram
            ax = df_z.hist(column='zDim')
            min_quant, max_quant, average = df_z.quantile(min_percentile), df_z.quantile(max_percentile), param.zDim.data.mean()
            for x in ax[0]:
                x.axvline(min_quant.zDim, linestyle=':', color='r', label=f"Min Percentile: {float(min_quant):.3f}")
                x.axvline(max_quant.zDim, linestyle=':', color='g', label=f"Max Percentile: {float(max_quant):.3f}")
                x.axvline(average, linestyle='solid', color='gold', label=f"Average: {float(average):.3f}")
                x.grid(False)
                plt.title(f"MR z-spacing imaging summary for {wildcard}")
                plt.legend()
                plt.show()
            
            # Summarizing xy-spacing
            df_xy = df_xy.describe(percentiles=[min_percentile, max_percentile], include='all')
            # Summarizing  z-spacing
            df_z = df_z.describe(percentiles=[min_percentile, max_percentile], include='all')
            
            # Saving data
            name_save = wildcard.replace('*', '').replace('.npy', '')
            save_name = 'imagingSummary__' + name_save + ".json"
            df_all = [df_years, df_fs, df_rt, df_et, df_it, df_etl, df_fa, df_na, df_xy, df_z]
            df_all = df_all[0].join(df_all[1:])
            df_all.to_json(path_save_checks / save_name, orient='columns', indent=4)

    def perform_ct_imaging_summary(self, 
                                wildcards_scans: List[str],
                                path_data: Path = None,
                                path_save_checks: Path = None,
                                min_percentile: float = 0.05,
                                max_percentile: float = 0.95
                                ) -> None:
        """
        Summarizes CT imaging acquisition parameters. Plots summary histograms
        for different dimensions and saves all acquisition parameters locally in JSON files.

        Args:
            wildcards_scans (List[str]): List of wildcards that determines the scans 
                that will be analyzed (Only MRI scans will be analyzed). You can learn more about wildcards in
                `this link <https://www.linuxtechtips.com/2013/11/how-wildcards-work-in-linux-and-unix.html>`_.
                For example: ``[\"STS*.CTscan.npy\"]``.
            path_data (Path, optional): Path to the MEDscan objects, if not specified will use ``path_save`` from the 
                inner-class ``Paths`` in the current instance.
            path_save_checks (Path, optional): Path where to save the checks, if not specified will use the one 
                in the current instance.
            min_percentile (float, optional): Minimum percentile to use for the histograms. Defaults to 0.05.
            max_percentile (float, optional): Maximum percentile to use for the histograms. Defaults to 0.95.
        
        Returns:
            None.
        """

        class param:
            manufacturer = []
            dates = []
            kernel = []

            class years:
                data = []
            class voltage:
                data = []
            class exposure:
                data = []
            class xyDim:
                data = []
            class zDim:
                data = []

        if len(wildcards_scans) == 0:
            print('wildcards_scans is empty')

        # wildcards checks:
        no_mr_scan = True
        for wildcard in wildcards_scans:
            if 'CTscan' in wildcard:
                no_mr_scan = False
        if no_mr_scan:
            raise ValueError(f"wildcards: {wildcards_scans} does not include CT scans. (Only CT scans are supported)")

        # Initialization
        if path_data is None:
            if self.paths._path_save:
                path_data = Path(self.paths._path_save)
            else:
                print("No path to data was given and path save is None.")
                return 0
        
        if not path_save_checks:
            if self.paths._path_save_checks:
                path_save_checks = Path(self.paths._path_save_checks)
            else:
                if (Path(os.getcwd()) / "checks").exists():
                    path_save_checks = Path(os.getcwd()) / "checks"
                else:
                    path_save_checks = (Path(os.getcwd()) / "checks").mkdir()

        # Looping through all the different wildcards
        for i in tqdm(range(len(wildcards_scans))):
            wildcard = wildcards_scans[i]
            file_paths = get_file_paths(path_data, wildcard)
            n_files = len(file_paths)
            param.dates = np.zeros(n_files)
            param.years.data = np.zeros(n_files)
            param.years.data = np.multiply(param.years.data, np.NaN)
            param.manufacturer = [None] * n_files
            param.voltage.data = np.zeros(n_files)
            param.voltage.data = np.multiply(param.voltage.data, np.NaN)
            param.exposure.data = np.zeros(n_files)
            param.exposure.data = np.multiply(param.exposure.data, np.NaN)
            param.kernel = [None] * n_files
            param.xyDim.data = np.zeros(n_files)
            param.xyDim.data = np.multiply(param.xyDim.data, np.NaN)
            param.zDim.data = np.zeros(n_files)
            param.zDim.data = np.multiply(param.zDim.data, np.NaN)
        
            # Loading and recording data
            for f in tqdm(range(n_files)):
                file = file_paths[f]

                # Open file for warning
                try:
                    warn_file = open(path_save_checks / 'imaging_summary_ct_warnings.txt', 'a')
                except IOError:
                    print("Could not open file")

                # Loading Data
                try:
                    with open(path_data / file, 'rb') as fe: medscan = pickle.load(fe)
                    print(f'Currently working on: {file}', file=warn_file)
                    
                    # DICOM header
                    info = medscan.dicomH[1]

                    # Recording dates
                    try:
                        param.dates[f] = info.AcquisitionDate
                    except AttributeError:
                        param.dates[f] = info.StudyDate
                        # Recording years
                    try:
                        years = str(param.dates[f])  # Only the first four characters represent the years
                        param.years.data[f] = years[0:4]
                    except Exception as e:
                        print(f'Cannot read dates of : {file}. Error: {e}', file=warn_file)
                        # Recording manufacturers
                    try:
                        param.manufacturer[f] = info.Manufacturer
                    except Exception as e:
                        print(f'Cannot read Manufacturer of: {file}. Error: {e}', file=warn_file)
                        # Recording voltage
                    try:
                        param.voltage.data[f] = info.KVP
                    except Exception as e:
                        print(f'Cannot read voltage of: {file}. Error: {e}', file=warn_file)
                        # Recording exposure
                    try:
                        param.exposure.data[f] = info.Exposure
                    except Exception as e:
                        print(f'Cannot read exposure of: {file}. Error: {e}', file=warn_file)
                        # Recording reconstruction kernel
                    try:
                        param.kernel[f] = info.ConvolutionKernel
                    except Exception as e:
                        print(f'Cannot read Kernel of: {file}. Error: {e}', file=warn_file)
                        # Recording xy spacing
                    try:
                        param.xyDim.data[f] = medscan.data.volume.spatialRef.PixelExtentInWorldX
                    except Exception as e:
                        print(f'Cannot read x spacing of: {file}. Error: {e}', file=warn_file)
                        # Recording z spacing
                    try:
                        param.zDim.data[f] = medscan.data.volume.spatialRef.PixelExtentInWorldZ
                    except Exception as e:
                        print(f'Cannot read z spacing of: {file}. Error: {e}', file=warn_file)
                except Exception as e:
                    print(f'Cannot load file: {file}', file=warn_file)

                warn_file.close()

            # Summarize data
                # Summarizing years
            df_years = pd.DataFrame(param.years.data, columns=['years']).describe(percentiles=[min_percentile, max_percentile], include='all')
                # Summarizing voltage
            df_voltage = pd.DataFrame(param.voltage.data, columns=['voltage']).describe(percentiles=[min_percentile, max_percentile], include='all')
                # Summarizing exposure
            df_exposure = pd.DataFrame(param.exposure.data, columns=['exposure']).describe(percentiles=[min_percentile, max_percentile], include='all')
                # Summarizing kernel
            df_kernel = pd.DataFrame(param.kernel, columns=['kernel']).describe(percentiles=[min_percentile, max_percentile], include='all')
                # Summarize xy spacing
            df_xy = pd.DataFrame(param.xyDim.data, columns=['xyDim']).describe(percentiles=[min_percentile, max_percentile], include='all')
                # Summarize z spacing
            df_z = pd.DataFrame(param.zDim.data, columns=['zDim']).describe(percentiles=[min_percentile, max_percentile], include='all')
                # Summarizing xy-spacing
            df_xy = pd.DataFrame(param.xyDim.data, columns=['xyDim'])
                # Summarizing z-spacing
            df_z = pd.DataFrame(param.zDim.data, columns=['zDim'])

            # Plotting xy-spacing histogram
            ax = df_xy.hist(column='xyDim')
            min_quant, max_quant, average = df_xy.quantile(min_percentile), df_xy.quantile(max_percentile), param.xyDim.data.mean()
            for x in ax[0]:
                x.axvline(min_quant.xyDim, linestyle=':', color='r', label=f"Min Percentile: {float(min_quant):.3f}")
                x.axvline(max_quant.xyDim, linestyle=':', color='g', label=f"Max Percentile: {float(max_quant):.3f}")
                x.axvline(average, linestyle='solid', color='gold', label=f"Average: {float(average):.3f}")
                x.grid(False)
                plt.title(f"CT xy-spacing imaging summary for {wildcard}")
                plt.legend()
                plt.show()
            
            # Plotting z-spacing histogram
            ax = df_z.hist(column='zDim')
            min_quant, max_quant, average = df_z.quantile(min_percentile), df_z.quantile(max_percentile), param.zDim.data.mean()
            for x in ax[0]:
                x.axvline(min_quant.zDim, linestyle=':', color='r', label=f"Min Percentile: {float(min_quant):.3f}")
                x.axvline(max_quant.zDim, linestyle=':', color='g', label=f"Max Percentile: {float(max_quant):.3f}")
                x.axvline(average, linestyle='solid', color='gold', label=f"Average: {float(average):.3f}")
                x.grid(False)
                plt.title(f"CT z-spacing imaging summary for {wildcard}")
                plt.legend()
                plt.show()
            
            # Summarizing xy-spacing
            df_xy = df_xy.describe(percentiles=[min_percentile, max_percentile], include='all')
            # Summarizing  z-spacing
            df_z = df_z.describe(percentiles=[min_percentile, max_percentile], include='all')

            # Saving data
            name_save = wildcard.replace('*', '').replace('.npy', '')
            save_name = 'imagingSummary__' + name_save + ".json"
            df_all = [df_years, df_voltage, df_exposure, df_kernel, df_xy, df_z]
            df_all = df_all[0].join(df_all[1:])
            df_all.to_json(path_save_checks / save_name, orient='columns', indent=4)

    def perform_imaging_summary(self, 
                                wildcards_scans: List[str],
                                path_data: Path = None,
                                path_save_checks: Path = None,
                                min_percentile: float = 0.05,
                                max_percentile: float = 0.95
                                ) -> None:
        """
        Summarizes CT and MR imaging acquisition parameters. Plots summary histograms
        for different dimensions and saves all acquisition parameters locally in JSON files.

        Args:
            wildcards_scans (List[str]): List of wildcards that determines the scans 
                that will be analyzed (CT and MRI scans will be analyzed). You can learn more about wildcards in
                `this link <https://www.linuxtechtips.com/2013/11/how-wildcards-work-in-linux-and-unix.html>`_.
                For example: ``[\"STS*.CTscan.npy\", \"STS*.MRscan.npy\"]``.
            path_data (Path, optional): Path to the MEDscan objects, if not specified will use ``path_save`` from the 
                inner-class ``Paths`` in the current instance.
            path_save_checks (Path, optional): Path where to save the checks, if not specified will use the one 
                in the current instance.
            min_percentile (float, optional): Minimum percentile to use for the histograms. Defaults to 0.05.
            max_percentile (float, optional): Maximum percentile to use for the histograms. Defaults to 0.95.
        
        Returns:
            None.
        """
        # MR imaging summary
        wildcards_scans_mr = [wildcard for wildcard in wildcards_scans if 'MRscan' in wildcard]
        if len(wildcards_scans_mr) == 0:
            print("Cannot perform imaging summary for MR, no MR scan wildcard was given! ")
        else:
            self.perform_mr_imaging_summary(
                                wildcards_scans_mr, 
                                path_data, 
                                path_save_checks, 
                                min_percentile, 
                                max_percentile)
        # CT imaging summary
        wildcards_scans_ct = [wildcard for wildcard in wildcards_scans if 'CTscan' in wildcard]
        if len(wildcards_scans_ct) == 0:
            print("Cannot perform imaging summary for CT, no CT scan wildcard was given! ")
        else:
            self.perform_ct_imaging_summary(
                                wildcards_scans_ct, 
                                path_data, 
                                path_save_checks, 
                                min_percentile, 
                                max_percentile)
