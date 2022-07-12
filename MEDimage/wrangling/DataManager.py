import json
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import pydicom.errors
import pydicom.misc
import ray
from MEDimage.MEDimage import MEDimage
from MEDimage.utils.imref import imref3d
from tqdm import tqdm, trange

from ..utils.save_MEDimage import save_MEDimage
from .process_dicom_scan_files import process_dicom_scan_files as pdsf


class DataManager(object):
    """
    Manages all the raw data (DICOM, NIfTI) and creates MEDimage class instances from it.
    """


    @dataclass
    class DICOM(object):
        """
        DICOM data management class that will organize data during the conversion to MEDimage class process
        """
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
        """
        NIfTI data management class that will organize data during the conversion to MEDimage class process
        """
        stack_path_images: List
        stack_path_roi: List
        stack_path_all: List

    def __init__(
            self, 
            path_to_dicoms: List = [],
            path_to_niftis: List = [],
            path_save: Path = None,
            save: bool = False,
            keep_instances: bool = True,
            n_batch: int = 2
    ) -> None:
        """
        Constructor of the class DataManager.

        Args:
            path_to_dicoms (Path, optional): Path specifying the full path to the starting directory
                where the DICOM data is located.
            path_to_niftis (Path, optional): Path specifying the full path to the starting directory
                where the NIfTI is located.
            path_save (Path, optional): Full path to the directory where to save all the MEDimage classes.
            save (bool, optional): True to save the MEDimage classes, False to return them.
            keep_instances(bool, optional): If True, will keep the created MEDimage instances in
                the `instances` attribute.
            n_batch (int, optional): Numerical value specifying the number of batch to use in the
                parallel computations (use 0 for serial).

        Returns:
            None
        """
        self._path_to_dicoms = path_to_dicoms
        self._path_to_niftis = path_to_niftis
        self._path_save = path_save
        self.save = save
        self.keep_instances = keep_instances
        self.n_batch = n_batch
        self.dicom = self.DICOM(
                stack_series_rs=[],
                stack_path_rs=[],
                stack_frame_rs=[],
                cell_series_id=[],
                cell_path_rs=[],
                cell_path_images=[],
                cell_frame_rs=[],
                cell_frame_id=[]
        )
        self.nifti = self.NIfTI(
                stack_path_images=[],
                stack_path_roi=[],
                stack_path_all=[]
        )
        self.instances = []
        self.path_to_objects = []
        self.summary = []
        self.__warned = False

    def __find_uid_cell_index(self, uid, cell) -> List: 
        """
        Finds the cell with the same `uid`. If not is present in `cell`, creates a new position 
        in the `cell` for the new `uid`.

        Args:
            uid (str):  Unique identifier of the Series to find.
            cell (List[str]): List of Unique identifiers of the Series.

        Returns:
            Union[List[str], str]: List or string of the uid  
        """
        return [len(cell)] if uid not in cell else[i for i, e in enumerate(cell) if e == uid]

    def __get_list_of_files(self, dir_name) -> List:
        """
        Gets all files in the given directory

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

    def __get_MEDimage_name_save(self, MEDimg: MEDimage) -> str:
        """
        Returns the name that will be used to save the MEDimage instance, based on the values of the attributes.

        Args:
            MEDimg(MEDimage): A MEDimage class instance.
        
        Returns:
            str: String of the name save.
        """
        if MEDimg.format == 'nifti':
            series_description = MEDimg.type.split('scan')[0]
        elif MEDimg.format == 'dicom':
            try:
                series_description = MEDimg.dicomH[0].SeriesDescription.translate({ord(ch): '-' for ch in '/\\ ()&:*'})
            except:
                series_description = MEDimg.dicomH[0].Modality.translate({ord(ch): '-' for ch in '/\\ ()&:*'})
        else:
            raise ValueError("Invalid format in the given MEDimage instance, must be 'npy' or 'nifti'")
        name_id = MEDimg.patientID.translate({ord(ch): '-' for ch in '/\\ ()&:*'})
        # final saving name
        name_complete = name_id + '__' + series_description + '.' + MEDimg.type + '.npy'
        return name_complete

    def __associate_rt_stuct(self) -> None:
        """
        Associates the imaging volumes to their mask using UIDs

        Returns:
            None
        """
        print('--> Associating all RT objects to imaging volumes')
        nRS = len(self.dicom.stack_path_rs)
        if nRS:
            for i in trange(0, nRS):
                try: # PUT ALL THE DICOM PATHS WITH THE SAME UID IN THE SAME PATH LIST
                    ind_series_id = self.__find_uid_cell_index(
                                                        self.dicom.stack_series_rs[i], 
                                                        self.dicom.cell_series_id)
                    for n in range(len(ind_series_id)):
                        self.dicom.cell_path_rs[ind_series_id[n]] += [self.dicom.stack_path_rs[i]]
                except:
                    ind_series_id = self.__find_uid_cell_index(
                                                        self.dicom.stack_frame_rs[i], 
                                                        self.dicom.cell_frame_id)
                    for n in range(len(ind_series_id)):
                        self.dicom.cell_path_rs[ind_series_id[n]] += [self.dicom.stack_path_rs[i]]
        print('DONE')

    def __read_all_dicoms(self) -> None:
        """
        Reads all the dicom files in the all the paths of the attribute `_path_to_dicoms`

        Returns:
            None
        """
        # SCANNING ALL FOLDERS IN INITIAL DIRECTORY
        print('\n--> Scanning all folders in initial directory...', end='')
        p = Path(self._path_to_dicoms)
        e_rglob = '*.[!xlsx,!xls,!py,!.DS_Store,!csv,!.,!txt,!..,!TXT,!npy,!m,!CT.npy]*'

        # EXTRACT ALL FILES IN THE PATH TO DICOMS
        if self._path_to_dicoms.is_dir():
            stack_folder_temp = list(p.rglob(e_rglob))
            stack_folder = [x for x in stack_folder_temp if not x.is_dir()]
        elif str(self._path_to_dicoms).find('json') != -1:
            with open(self._path_to_dicoms) as f:
                data = json.load(f)
                for value in data.values():
                    stack_folder_temp = value
            directory_name = str(stack_folder_temp).replace("'", '').replace('[', '').replace(']', '')
            stack_folder = self.__get_list_of_files(directory_name)

        # READ ALL DICOM FILES AND UPDATE ATTRBIUTES FOR FURTHER PROCESSING
        for file in tqdm(stack_folder):
            if pydicom.misc.is_dicom(file):
                try:
                    info = pydicom.dcmread(str(file))
                    if info.Modality in ['MR', 'PT', 'CT']:
                        ind_series_id = self.__find_uid_cell_index(
                                                        info.SeriesInstanceUID, 
                                                        self.dicom.cell_series_id)[0]
                        if ind_series_id == len(self.dicom.cell_series_id):  # New volume
                            self.dicom.cell_series_id = self.dicom.cell_series_id + [info.SeriesInstanceUID]
                            self.dicom.cell_frame_id += [info.FrameOfReferenceUID]
                            self.dicom.cell_path_images += [[]]
                            self.dicom.cell_path_rs = self.dicom.cell_path_rs + [[]]
                        self.dicom.cell_path_images[ind_series_id] += [file]
                    elif info.Modality == 'RTSTRUCT':
                        self.dicom.stack_path_rs += [file]
                        try:
                            series_uid = info.ReferencedFrameOfReferenceSequence[
                                        0].RTReferencedStudySequence[
                                        0].RTReferencedSeriesSequence[
                                        0].SeriesInstanceUID
                        except:
                            series_uid = 'NotFound'
                        self.dicom.stack_series_rs += [series_uid]
                        try:
                            frame_uid = info.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
                        except:
                            frame_uid = info.FrameOfReferenceUID
                        self.dicom.stack_frame_rs += [frame_uid]

                except Exception as e:
                    print(f'Error while reading: {file}, error: {e}\n')
                    continue
        print('DONE')

        # ASSIOCIATE ALL VOLUMES TO THEIR MASK
        self.__associate_rt_stuct()

    def process_all_dicoms(self) -> List[MEDimage]:
        """
        This function reads the DICOM content of all the sub-folder tree of a starting directory defined by 
        `path_to_dicoms`. It then organizes the data (files throughout the starting directory are associated by
        'SeriesInstanceUID') in the MEDimage class including the region of  interest (ROI) defined by an 
        associated RTstruct. All MEDimage classes hereby created are saved in `path_save` with a name
        varying with every scan.

        Returns:
            List[MEDimage]: List of MEDimage instances.
        """
        ray.init(local_mode=True, include_dashboard=True)

        print('--> Reading all DICOM objects to create MEDimage classes')
        self.__read_all_dicoms()

        n_scans = len(self.dicom.cell_path_images)
        if self.n_batch is None:
            n_batch = 1
        elif n_scans < self.n_batch:
            n_batch = n_scans
        else:
            n_batch = self.n_batch

        # Distribute the first tasks to all workers
        ids = [pdsf.remote(
                        self.dicom.cell_path_images[i], 
                        self.dicom.cell_path_rs[i], 
                        self._path_save,
                        self.save)
            for i in range(n_batch)]
        # Update the path to the created instances
        if self._path_save:
            for instance in ray.get(ids):
                name_save = self.__get_MEDimage_name_save(instance)
                self.path_to_objects.append(str(self._path_save / name_save))
                # Update processing summary:
                roi_names = instance.scan.ROI.roi_names
                name_save += '+' + '+'.join(roi_names.values())
                self.summary.append(name_save)

        nb_job_left = n_scans - n_batch

        # Get MEDimage instances
        if len(self.instances)>10 and not self.__warned:
            # User cannot save over 10 instances in the class
            warnings.warn("You have more than 10 MEDimage objects saved in the current DataManager instance, \
                the rest of the instances will/can be saved locally only.")
            self.__warned = True
        elif self.keep_instances:
            self.instances.extend(ray.get(ids))

        # Distribute the remaining tasks
        for _ in trange(n_scans):
            _, not_ready = ray.wait(ids, num_returns=1)
            ids = not_ready
            if nb_job_left > 0:
                idx = n_scans - nb_job_left
                ids.extend([pdsf.remote(self.dicom.cell_path_images[idx], 
                                        self.dicom.cell_path_rs[idx], 
                                        self._path_save,
                                        self.save)])
                nb_job_left -= 1

        # Update the path to the created instances
        if self._path_save:
            for instance in ray.get(ids):
                name_save = self.__get_MEDimage_name_save(instance)
                self.path_to_objects.extend(str(self._path_save / name_save))
                # Update processing summary:
                roi_names = instance.scan.ROI.roi_names
                name_save += '+' + '+'.join(roi_names.values())
                self.summary.append(name_save)

        # Get MEDimage instances
        if len(self.instances)>10 and not self.__warned:
            warnings.warn("You have more than 10 MEDimage objects saved in the current DataManager instance, \
                the rest of the instances will/can be saved locally only.")
            self.__warned = True
        else:
            self.instances.extend(ray.get(ids))
        print('DONE')

        return self.instances

    def __read_all_niftis(self) -> None:
        """
        Reads all files in the initial path and organizes other path to images and roi
        in the class attrbiutes.

        Note:
            The `nifti_image_path` filename must respect the following naming convention:
            PatientID__ImagingScanName(tumorAuto).ImagingModality.nii.gz

        Returns:
            None.
        """
        print('\n--> Scanning all folders in initial directory')
        p = Path(self._path_to_niftis)
        e_rglob1 = '*.nii'
        e_rglob2 = '*.nii.gz'

        # EXTRACT ALL FILES IN THE PATH TO DICOMS
        if p.is_dir():
            self.nifti.stack_path_all = list(p.rglob(e_rglob1))
            self.nifti.stack_path_all.extend(list(p.rglob(e_rglob2)))
        else:
            raise TypeError("`_path_to_niftis` must be a path to a directory")

        all_niftis = list(self.nifti.stack_path_all)
        for i in trange(0, len(all_niftis)):
            if 'ROI' in all_niftis[i].name.split("."):
                self.nifti.stack_path_roi.append(all_niftis[i])
            else:
                self.nifti.stack_path_images.append(all_niftis[i])
        print('DONE')

    def __associate_roi_to_image(self, image_file: Union[Path, str], MEDimg: MEDimage) -> MEDimage:
        """
        Extracts all ROI data from the given path for the given patient ID and updates all class attributes with 
        the new extracted data.

        Args:
            image_file(Union[Path, str]): Path to the ROI data.
            MEDimg (MEDimage): MEDimage class instance that will hold the data. 

        Returns:
            MEDimage: Returns a MEDimage instance with updated roi attributes.
        """
        image_file = Path(image_file)
        roi_index = 0

        for file in self.nifti.stack_path_roi:
            _id = image_file.name.split("(")[0] # id is PatientID__ImagingScanName
            # Load the patient's ROI nifti files:
            if file.name.startswith(_id) and 'ROI' in file.name.split("."):
                roi = nib.load(file)
                roi_data = MEDimg.scan.ROI.convert_to_LPS(data=roi.get_fdata())
                roi_name = file.name[file.name.find("(") + 1 : file.name.find(")")]
                name_set = file.name[file.name.find("_") + 2 : file.name.find("(")]
                MEDimg.scan.ROI.update_indexes(key=roi_index, indexes=np.nonzero(roi_data.flatten()))
                MEDimg.scan.ROI.update_nameSet(key=roi_index, nameSet=name_set)
                MEDimg.scan.ROI.update_ROIname(key=roi_index, ROIname=roi_name)
                roi_index += 1
        return MEDimg

    def __associate_spatialRef(self, nifti_file: Union[Path, str], MEDimg: MEDimage) -> MEDimage:
        """
        Computes the imref3d spatialRef using a NIFTI file and updates the spatialRef attribute.

        Args:
            nifti_file(Union[Path, str]): Path to the nifti data.
            MEDimg (MEDimage): MEDimage class instance that will hold the data. 

        Returns:
            MEDimage: Returns a MEDimage instance with updated spatialRef attribute.
        """
        # Loading the nifti file :
        nifti = nib.load(nifti_file)
        nifti_data = MEDimg.scan.volume.data

        # spatialRef Creation
        pixelX = abs(nifti.affine[0, 0])
        pixelY = abs(nifti.affine[1, 1])
        sliceS = abs(nifti.affine[2, 2])
        min_grid = nifti.affine[:3, 3] * [-1.0, -1.0, 1.0] # x and y are flipped
        min_Xgrid = min_grid[0]
        min_Ygrid = min_grid[1]
        min_Zgrid = min_grid[2]
        size_image = np.shape(nifti_data)
        spatialRef = imref3d(size_image, abs(pixelX), abs(pixelY), abs(sliceS))
        spatialRef.XWorldLimits = (np.array(spatialRef.XWorldLimits) -
                                (spatialRef.XWorldLimits[0] -
                                    (min_Xgrid-pixelX/2))
                                ).tolist()
        spatialRef.YWorldLimits = (np.array(spatialRef.YWorldLimits) -
                                (spatialRef.YWorldLimits[0] -
                                    (min_Ygrid-pixelY/2))
                                ).tolist()
        spatialRef.ZWorldLimits = (np.array(spatialRef.ZWorldLimits) -
                                (spatialRef.ZWorldLimits[0] -
                                    (min_Zgrid-sliceS/2))
                                ).tolist()

        # Converting the results into lists
        spatialRef.ImageSize = spatialRef.ImageSize.tolist()
        spatialRef.XIntrinsicLimits = spatialRef.XIntrinsicLimits.tolist()
        spatialRef.YIntrinsicLimits = spatialRef.YIntrinsicLimits.tolist()
        spatialRef.ZIntrinsicLimits = spatialRef.ZIntrinsicLimits.tolist()

        # update spatialRef in the volume sub-class
        MEDimg.scan.volume.update_spatialRef(spatialRef)

        return MEDimg

    def process_all_niftis(self) -> List[MEDimage]:
        """
        This function reads the NIfTI content of all the sub-folder tree of a starting directory defined by 
        `path_to_niftis`. It then organizes the data in the MEDimage class including the region of  interest (ROI)
        defined by an associated mask file. All MEDimage classes hereby created are saved in `path_save` with a name
        varying with every scan.

        Returns:
            List[MEDimage]: List of MEDimage instances.
        """
        self.__read_all_niftis()
        print('--> Reading all NIfTI objects (imaging volumes & masks) to create MEDimage classes')
        for file in tqdm(self.nifti.stack_path_images):
            # User cannot save over 10 instances in the class
            if len(self.instances)>10 and not self.__warned:
                warnings.warn("You have more than 10 MEDimage objects saved in the current DataManager instance, \
                    the rest of the instances will/can be saved locally only.")
                self.__warned = True
            # INITIALIZE MEDimage INSTANCE AND UPDATE ATTRIBUTES
            MEDimg = MEDimage()
            MEDimg.patientID = os.path.basename(file).split("_")[0]
            MEDimg.type = os.path.basename(file).split(".")[-3]
            MEDimg.format = "nifti"
            MEDimg.scan.set_orientation(orientation="Axial")
            MEDimg.scan.set_patientPosition(patientPosition="HFS")
            MEDimg.scan.volume.data = nib.load(file).get_fdata()
            # RAS to LPS
            MEDimg.scan.volume.convert_to_LPS()
            MEDimg.scan.volume.scan_rot = None
            # UPDATE spatialRef
            self.__associate_spatialRef(file, MEDimg)
            # GET ROI
            MEDimage_instance = self.__associate_roi_to_image(file, MEDimg)
            if self.keep_instances:
                self.instances.append(MEDimage_instance)
            if self.save and self._path_save:
                save_MEDimage(MEDimg, MEDimg.type.split('scan')[0], self._path_save)
            # Update the path to the created instances
            name_save = self.__get_MEDimage_name_save(MEDimage_instance)
            if self._path_save:
                self.path_to_objects.append(str(self._path_save / name_save))
            # Update processing summary:
            roi_names = MEDimage_instance.scan.ROI.roi_names
            name_save += '+' + '+'.join(roi_names.values())
            self.summary.append(name_save)
        print('DONE')
        return self.instances

    def summarize(self):
        """
        Creates and shows a summary of processed scans by study, institution, scan type and roi type

        Args:
            None
        Returns:
            None
        """
        summary = pd.DataFrame(columns=['study', 'institution', 'scan_type', 'roi_type', 'count'])
        # Process each scan in summary
        for scan in self.summary:
            if '-' not in scan.split('_')[0]:
                logging.warning(f"The patient ID of the following file: {scan} does not respect the MEDimage "\
                    "naming convention 'study-institution-id' (Ex: Glioma-TCGA-001)")
                continue
            # Initialization
            study = scan.split('-')[0]
            institution = scan.split('-')[1]
            scan_type = scan[scan.find('__')+2 : scan.find('.')]
            roi_names = scan.split('+')[1:]
            roi_names = [roi_names] if roi_names is str else roi_names
            # summarize by study
            if study in summary['study'].values:
                summary.loc[(summary['study'].dropna() == study).argmax(), 'count'] += 1
                # update institutions data
                if institution in summary['institution'][(summary['study'].dropna() == study)].values:
                    summary.loc[((summary['study'].dropna() == study) & 
                                (summary['institution'].dropna() == institution)).argmax(), 'count'] += 1
                    # update scans type data
                    if scan_type in summary['scan_type'][(summary['study'].dropna() == study)
                                                        & (summary['institution'].dropna() == institution)].values:
                        summary.loc[((summary['study'].dropna() == study) & 
                                    (summary['institution'].dropna() == institution) &
                                    (summary['scan_type'].dropna() == scan_type)).argmax(), 'count'] += 1
                        # update rois type data
                        for roi_name in roi_names:
                            if roi_name in summary['roi_type'][(summary['study'].dropna() == study)
                                                        & (summary['institution'].dropna() == institution)
                                                        & (summary['roi_type'].dropna() == roi_name)].values:
                                summary.loc[((summary['study'].dropna() == study) & 
                                            (summary['institution'].dropna() == institution) &
                                            (summary['scan_type'].dropna() == scan_type) &
                                            (summary['roi_type'].dropna() == roi_name)).argmax(), 'count'] += 1
                            else:
                                summary = summary.append({
                                                        'study': study, 
                                                        'institution': institution, 
                                                        'scan_type': scan_type, 
                                                        'roi_type': roi_name, 
                                                        'count' : 1
                                                        }, ignore_index=True)
                    else:
                        summary = summary.append({
                                                'study': study, 
                                                'institution': institution, 
                                                'scan_type': scan_type,
                                                'roi_type': "",
                                                'count' : 1
                                                }, ignore_index=True)
                        for roi_name in roi_names:
                            summary = summary.append({
                                                    'study': study, 
                                                    'institution': institution, 
                                                    'scan_type': scan_type, 
                                                    'roi_type': roi_name, 
                                                    'count' : 1
                                                    }, ignore_index=True)
                else:
                    summary = summary.append({
                                            'study': study, 
                                            'institution': institution, 
                                            'scan_type': "", 
                                            'roi_type': "", 
                                            'count' : 1
                                            }, ignore_index=True)
                    summary = summary.append({
                                            'study': study, 
                                            'institution': institution, 
                                            'scan_type': scan_type, 
                                            'roi_type': "", 
                                            'count' : 1
                                            }, ignore_index=True)
                    for roi_name in roi_names:
                        summary = summary.append({
                                                'study': study, 
                                                'institution': institution, 
                                                'scan_type': scan_type, 
                                                'roi_type': roi_name, 
                                                'count' : 1
                                                }, ignore_index=True)
            # Add new study
            else:
                summary = summary.append({
                                        'study': study, 
                                        'institution': "", 
                                        'scan_type': "", 
                                        'roi_type': "", 
                                        'count' : 1
                                        }, ignore_index=True)
                summary = summary.append({
                                        'study': study, 
                                        'institution': institution, 
                                        'scan_type': "", 
                                        'roi_type': "", 
                                        'count' : 1
                                        }, ignore_index=True)
                summary = summary.append({
                                        'study': study, 
                                        'institution': institution, 
                                        'scan_type': scan_type, 
                                        'roi_type': "", 
                                        'count' : 1
                                        }, ignore_index=True)
                for roi_name in roi_names:
                    summary = summary.append({
                                            'study': study, 
                                            'institution': institution, 
                                            'scan_type': scan_type, 
                                            'roi_type': roi_name, 
                                            'count' : 1
                                            }, ignore_index=True)
        self.summary = summary
        print(summary.to_string(index=False))
