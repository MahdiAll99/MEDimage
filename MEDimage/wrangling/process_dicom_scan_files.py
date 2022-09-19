#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Union
import warnings

import numpy as np
import pydicom
import ray

from ..utils.imref import imref3d

warnings.simplefilter("ignore")

from pathlib import Path

from MEDimage.MEDimage import MEDimage

from ..processing.segmentation import get_roi
from ..utils.save_MEDimage import save_MEDimage


class ProcessDICOM():
    """
    Class to process dicom files and extract 3D masks from RT struct ROIs
    in order to oganize all this imaging data in a MEDimage class object.
    """

    def __init__(
        self,
        path_images: List[Path],
        path_rs: List[Path],
        path_save: Union[str, Path],
        save: bool) -> None:
        """
        Args:
            path_images (List[Path]): List of paths to the dicom files of a single scan.
            path_rs (List[Path]): List of paths to the RT struct dicom files for the same scan.
            path_save (Union[str, Path]): Path to the folder where the MEDimage object will be saved.
            save (bool): Whether to save the MEDimage object or not.
        
        Returns:
            None.
        """
        self.path_images = path_images
        self.path_rs = path_rs
        self.path_save = Path(path_save) if path_save is str else path_save
        self.save = save
    
    def __get_dicom_scan_orientation(self, dicom_header: List[pydicom.dataset.FileDataset]) -> str:
        """
        Get the orientation of the scan.
        
        Args:
            dicom_header (List[pydicom.dataset.FileDataset]): List of dicom headers.
        
        Returns:
            str: Orientation of the scan.
        """
        n_slices = len(dicom_header)
        image_patient_positions_x = [dicom_header[i].ImagePositionPatient[0] for i in range(n_slices)]
        image_patient_positions_y = [dicom_header[i].ImagePositionPatient[1] for i in range(n_slices)]
        image_patient_positions_z = [dicom_header[i].ImagePositionPatient[2] for i in range(n_slices)]
        dist = [
            max(np.abs(np.diff(image_patient_positions_x))), 
            max(np.abs(np.diff(image_patient_positions_y))), 
            max(np.abs(np.diff(image_patient_positions_z)))
        ]
        index = dist.index(max(dist))
        if index == 0:
            orientation = 'Sagittal'
        elif index == 1:
            orientation = 'Coronal'
        else:
            orientation = 'Axial'
        
        return orientation

    @ray.remote
    def process_files(self) -> MEDimage:
        """
        Reads DICOM files (imaging volume + ROIs) in the instance data path 
        and then organizes it in the MEDimage class.
        
        Args:
            None.
        
        Returns:
            MEDimg (MEDimage): Instance of a MEDimage class.
        """
        # Since we created a worker, we need to add code path to the system
        from .combine_slices import combine_slices

        # PARTIAL PARSING OF ARGUMENTS
        if self.path_images is None:
            raise ValueError('At least two arguments must be provided')

        # INITIALIZATION
        MEDimg = MEDimage()

        # IMAGING DATA AND ROI DEFINITION (if applicable)
        # Reading DICOM images and headers
        dicom_hi = [pydicom.dcmread(str(dicom_file), force=True)
                for dicom_file in self.path_images]

        try:
            # Determination of the scan orientation
            MEDimg.scan.orientation = self.__get_dicom_scan_orientation(dicom_hi)

            # IMPORTANT NOTE: extract_voxel_data using combine_slices from dicom_numpy
            # missing slices and oblique restrictions apply see the reference:
            # https://dicom-numpy.readthedocs.io/en/latest/index.html#dicom_numpy.combine_slices
            try:
                voxel_ndarray, ijk_to_xyz, rotation_m, scaling_m = combine_slices(dicom_hi)
            except ValueError as e:
                raise ValueError(f'Invalid DICOM data for combine_slices(). Error: {e}')

            # Alignment of scan coordinates for MR scans
            # (inverse of ImageOrientationPatient rotation matrix)
            if not np.allclose(rotation_m, np.eye(rotation_m.shape[0])):
                MEDimg.scan.volume.scan_rot = rotation_m

            MEDimg.scan.volume.data = voxel_ndarray
            MEDimg.type = dicom_hi[0].Modality + 'scan'

            # 7. Creation of imref3d object
            pixel_x = scaling_m[0, 0]
            pixel_y = scaling_m[1, 1]
            slice_s = scaling_m[2, 2]
            min_grid = rotation_m@ijk_to_xyz[:3, 3]
            min_x_grid = min_grid[0]
            min_y_grid = min_grid[1]
            min_z_grid = min_grid[2]
            size_image = np.shape(voxel_ndarray)
            spatial_ref = imref3d(size_image, pixel_x, pixel_y, slice_s)
            spatial_ref.XWorldLimits = (np.array(spatial_ref.XWorldLimits) -
                                    (spatial_ref.XWorldLimits[0] -
                                        (min_x_grid-pixel_x/2))).tolist()
            spatial_ref.YWorldLimits = (np.array(spatial_ref.YWorldLimits) -
                                    (spatial_ref.YWorldLimits[0] -
                                        (min_y_grid-pixel_y/2))).tolist()
            spatial_ref.ZWorldLimits = (np.array(spatial_ref.ZWorldLimits) -
                                    (spatial_ref.ZWorldLimits[0] -
                                        (min_z_grid-slice_s/2))).tolist()

            # Converting the results into lists
            spatial_ref.ImageSize = spatial_ref.ImageSize.tolist()
            spatial_ref.XIntrinsicLimits = spatial_ref.XIntrinsicLimits.tolist()
            spatial_ref.YIntrinsicLimits = spatial_ref.YIntrinsicLimits.tolist()
            spatial_ref.ZIntrinsicLimits = spatial_ref.ZIntrinsicLimits.tolist()

            # Update the spatial reference in the MEDimage class
            MEDimg.scan.volume.spatialRef = spatial_ref
            
            # DICOM HEADERS OF IMAGING DATA
            dicom_h = [
                pydicom.dcmread(str(dicom_file),stop_before_pixels=True,force=True) for dicom_file in self.path_images
                ]
            for i in range(0, len(dicom_h)):
                dicom_h[i].remove_private_tags()
            MEDimg.dicomH = dicom_h

            # DICOM RTstruct (if applicable)
            if self.path_rs is not None and len(self.path_rs) > 0:
                dicom_rs_full = [
                    pydicom.dcmread(str(dicom_file),
                                    stop_before_pixels=True,
                                    force=True)
                    for dicom_file in self.path_rs
                ]
                for i in range(0, len(dicom_rs_full)):
                    dicom_rs_full[i].remove_private_tags()

            # GATHER XYZ POINTS OF ROIs USING RTstruct
            n_rs = len(dicom_rs_full) if type(dicom_rs_full) is list else dicom_rs_full
            contour_num = 0
            for rs in range(n_rs):
                n_roi = len(dicom_rs_full[rs].StructureSetROISequence)
                for roi in range(n_roi):
                    if roi!=0:
                        if dicom_rs_full[rs].StructureSetROISequence[roi].ROIName == \
                                dicom_rs_full[rs].StructureSetROISequence[roi-1].ROIName:
                            continue
                    points = []
                    name_set_strings = ['StructureSetName', 'StructureSetDescription',
                                    'series_description', 'SeriesInstanceUID']
                    for name_field in name_set_strings:
                        if name_field in dicom_rs_full[rs]:
                            name_set = getattr(dicom_rs_full[rs], name_field)
                            name_set_info = name_field
                            break

                    MEDimg.scan.ROI.update_roi_name(key=contour_num,
                                                    roi_name=dicom_rs_full[rs].StructureSetROISequence[roi].ROIName)
                    MEDimg.scan.ROI.update_indexes(key=contour_num,
                                                    indexes=None)
                    MEDimg.scan.ROI.update_name_set(key=contour_num,
                                                    name_set=name_set)
                    MEDimg.scan.ROI.update_name_set_info(key=contour_num,
                                                    nameSetInfo=name_set_info)
                    
                    try:
                        n_closed_contour = len(dicom_rs_full[rs].ROIContourSequence[roi].ContourSequence)
                        ind_closed_contour = []
                        for s in range(0, n_closed_contour):
                            # points stored in the RTstruct file for a given closed
                            # contour (beware: there can be multiple closed contours
                            # on a given slice).
                            pts_temp = dicom_rs_full[rs].ROIContourSequence[roi].ContourSequence[s].ContourData
                            n_points = int(len(pts_temp) / 3)
                            if len(pts_temp) > 0:
                                ind_closed_contour = ind_closed_contour + np.tile(s, n_points).tolist()
                                if type(points) == list:
                                    points = np.reshape(np.transpose(pts_temp),(n_points, 3))
                                else:
                                    points = np.concatenate(
                                            (points, np.reshape(np.transpose(pts_temp), (n_points, 3))),
                                            axis=0
                                            )
                        # Save the XYZ points in the MEDimage class
                        MEDimg.scan.ROI.update_indexes(
                                                    key=contour_num, 
                                                    indexes=np.concatenate(
                                                            (points, 
                                                            np.reshape(ind_closed_contour, (len(ind_closed_contour), 1))),
                                                    axis=1)
                                                    )
                        # Compute the ROI box
                        _, roi_obj = get_roi(
                                        MEDimg,
                                        name_roi='{' + dicom_rs_full[rs].StructureSetROISequence[roi].ROIName + '}',
                                        box_string='full'
                                        )

                        # Save the ROI box non-zero indexes in the MEDimage class
                        MEDimg.scan.ROI.update_indexes(key=contour_num, indexes=np.nonzero(roi_obj.data.flatten()))

                    except Exception as e:
                        print('patientID: ' + dicom_hi[0].PatientID + ' error: ' + str(e) + ' n_roi: ' + str(roi) + ' n_rs:' + str(rs))
                        MEDimg.scan.ROI.update_indexes(key=contour_num, indexes=np.NaN)
                    contour_num += 1

            # Save additional scan information in the MEDimage class
            MEDimg.scan.patientPosition = dicom_h[0].PatientPosition
            MEDimg.patientID = str(dicom_h[0].PatientID)
            MEDimg.format = "dicom"
            if 'SeriesDescription' in dicom_h[0]:
                MEDimg.series_description = dicom_h[0].SeriesDescription
            else:
                MEDimg.series_description = dicom_h[0].Modality

            # save MEDimage class instance as a pickle object
            if self.save and self.path_save:
                save_MEDimage(MEDimg, self.path_save)

        except Exception as e:
            print('patientID: ' + dicom_hi[0].PatientID + ' error: ' + str(e))
            return MEDimg
        
        return MEDimg
