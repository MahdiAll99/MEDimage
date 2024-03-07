#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import warnings
from typing import List, Union

import numpy as np
import pydicom
import ray

from ..utils.imref import imref3d

warnings.simplefilter("ignore")

from pathlib import Path

from ..MEDscan import MEDscan
from ..processing.segmentation import get_roi
from ..utils.save_MEDscan import save_MEDscan


class ProcessDICOM():
    """
    Class to process dicom files and extract imaging volume and 3D masks from it
    in order to oganize the data in a MEDscan class object.
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
            path_save (Union[str, Path]): Path to the folder where the MEDscan object will be saved.
            save (bool): Whether to save the MEDscan object or not.
        
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
            np.median(np.abs(np.diff(image_patient_positions_x))), 
            np.median(np.abs(np.diff(image_patient_positions_y))), 
            np.median(np.abs(np.diff(image_patient_positions_z)))
        ]
        index = dist.index(max(dist))
        if index == 0:
            orientation = 'Sagittal'
        elif index == 1:
            orientation = 'Coronal'
        else:
            orientation = 'Axial'
        
        return orientation

    def __merge_slice_pixel_arrays(self, slice_datasets):
        first_dataset = slice_datasets[0]
        num_rows = first_dataset.Rows
        num_columns = first_dataset.Columns
        num_slices = len(slice_datasets)

        sorted_slice_datasets = self.__sort_by_slice_spacing(slice_datasets)

        if any(self.__requires_rescaling(d) for d in sorted_slice_datasets):
            voxels = np.empty(
                (num_columns, num_rows, num_slices), dtype=np.float32)
            for k, dataset in enumerate(sorted_slice_datasets):
                slope = float(getattr(dataset, 'RescaleSlope', 1))
                intercept = float(getattr(dataset, 'RescaleIntercept', 0))
                voxels[:, :, k] = dataset.pixel_array.T.astype(
                    np.float32)*slope + intercept
        else:
            dtype = first_dataset.pixel_array.dtype
            voxels = np.empty((num_columns, num_rows, num_slices), dtype=dtype)
            for k, dataset in enumerate(sorted_slice_datasets):
                voxels[:, :, k] = dataset.pixel_array.T

        return voxels

    def __requires_rescaling(self, dataset):
        return hasattr(dataset, 'RescaleSlope') or hasattr(dataset, 'RescaleIntercept')

    def __ijk_to_patient_xyz_transform_matrix(self, slice_datasets):
        first_dataset = self.__sort_by_slice_spacing(slice_datasets)[0]
        image_orientation = first_dataset.ImageOrientationPatient
        row_cosine, column_cosine, slice_cosine = self.__extract_cosines(
            image_orientation)

        row_spacing, column_spacing = first_dataset.PixelSpacing
        slice_spacing = self.__slice_spacing(slice_datasets)

        transform = np.identity(4, dtype=np.float32)
        rotation = np.identity(3, dtype=np.float32)
        scaling = np.identity(3, dtype=np.float32)

        transform[:3, 0] = row_cosine*column_spacing
        transform[:3, 1] = column_cosine*row_spacing
        transform[:3, 2] = slice_cosine*slice_spacing

        transform[:3, 3] = first_dataset.ImagePositionPatient

        rotation[:3, 0] = row_cosine
        rotation[:3, 1] = column_cosine
        rotation[:3, 2] = slice_cosine

        rotation = np.transpose(rotation)

        scaling[0, 0] = column_spacing
        scaling[1, 1] = row_spacing
        scaling[2, 2] = slice_spacing

        return transform, rotation, scaling

    def __validate_slices_form_uniform_grid(self, slice_datasets):
        """
        Perform various data checks to ensure that the list of slices form a
        evenly-spaced grid of data.
        Some of these checks are probably not required if the data follows the
        DICOM specification, however it seems pertinent to check anyway.
        """
        invariant_properties = [
            'Modality',
            'SOPClassUID',
            'SeriesInstanceUID',
            'Rows',
            'Columns',
            'ImageOrientationPatient',
            'PixelSpacing',
            'PixelRepresentation',
            'BitsAllocated',
            'BitsStored',
            'HighBit',
        ]

        for property_name in invariant_properties:
            self.__slice_attribute_equal(slice_datasets, property_name)

        self.__validate_image_orientation(slice_datasets[0].ImageOrientationPatient)

        slice_positions = self.__slice_positions(slice_datasets)
        self.__check_for_missing_slices(slice_positions)

    def __validate_image_orientation(self, image_orientation):
        """
        Ensure that the image orientation is supported
        - The direction cosines have magnitudes of 1 (just in case)
        - The direction cosines are perpendicular
        """

        row_cosine, column_cosine, slice_cosine = self.__extract_cosines(
            image_orientation)

        if not self.__almost_zero(np.dot(row_cosine, column_cosine), 1e-4):
            raise ValueError(
                "Non-orthogonal direction cosines: {}, {}".format(row_cosine, column_cosine))
        elif not self.__almost_zero(np.dot(row_cosine, column_cosine), 1e-8):
            warnings.warn("Direction cosines aren't quite orthogonal: {}, {}".format(
                row_cosine, column_cosine))

        if not self.__almost_one(np.linalg.norm(row_cosine), 1e-4):
            raise ValueError(
                "The row direction cosine's magnitude is not 1: {}".format(row_cosine))
        elif not self.__almost_one(np.linalg.norm(row_cosine), 1e-8):
            warnings.warn(
                "The row direction cosine's magnitude is not quite 1: {}".format(row_cosine))

        if not self.__almost_one(np.linalg.norm(column_cosine), 1e-4):
            raise ValueError(
                "The column direction cosine's magnitude is not 1: {}".format(column_cosine))
        elif not self.__almost_one(np.linalg.norm(column_cosine), 1e-8):
            warnings.warn(
                "The column direction cosine's magnitude is not quite 1: {}".format(column_cosine))
        sys.stderr.flush()

    def __is_close(self, a, b, rel_tol=1e-9, abs_tol=0.0):
        return abs(a-b) <= max(rel_tol*max(abs(a), abs(b)), abs_tol)

    def __almost_zero(self, value, abs_tol):
        return self.__is_close(value, 0.0, abs_tol=abs_tol)

    def __almost_one(self, value, abs_tol):
        return self.__is_close(value, 1.0, abs_tol=abs_tol)

    def __extract_cosines(self, image_orientation):
        row_cosine = np.array(image_orientation[:3])
        column_cosine = np.array(image_orientation[3:])
        slice_cosine = np.cross(row_cosine, column_cosine)
        return row_cosine, column_cosine, slice_cosine

    def __slice_attribute_equal(self, slice_datasets, property_name):
        initial_value = getattr(slice_datasets[0], property_name, None)
        for slice_idx, dataset in enumerate(slice_datasets[1:]):
            value = getattr(dataset, property_name, None)
            if value != initial_value:
                msg = f'Slice {slice_idx+1} have different value for {property_name}: {value} != {initial_value}'
                warnings.warn(msg)

    def __slice_positions(self, slice_datasets):
        image_orientation = slice_datasets[0].ImageOrientationPatient
        row_cosine, column_cosine, slice_cosine = self.__extract_cosines(
            image_orientation)
        return [np.dot(slice_cosine, d.ImagePositionPatient) for d in slice_datasets]

    def __check_for_missing_slices(self, slice_positions):
        slice_positions_diffs = np.diff(sorted(slice_positions))
        if not np.allclose(slice_positions_diffs, slice_positions_diffs[0], atol=0, rtol=1e-5):
            msg = "The slice spacing is non-uniform. Slice spacings:\n{}"
            warnings.warn(msg.format(slice_positions_diffs))
            sys.stderr.flush()
        if not np.allclose(slice_positions_diffs, slice_positions_diffs[0], atol=0, rtol=1e-1):
            raise ValueError('The slice spacing is non-uniform. It appears there are extra slices from another scan')

    def __slice_spacing(self, slice_datasets):
        if len(slice_datasets) > 1:
            slice_positions = self.__slice_positions(slice_datasets)
            slice_positions_diffs = np.diff(sorted(slice_positions))
            return np.mean(slice_positions_diffs)

        return 0.0

    def __sort_by_slice_spacing(self, slice_datasets):
        slice_spacing = self.__slice_positions(slice_datasets)
        return [d for (s, d) in sorted(zip(slice_spacing, slice_datasets))]

    def combine_slices(self, slice_datasets: List[pydicom.dataset.FileDataset]) -> List[np.ndarray]:
        """
        Given a list of pydicom datasets for an image series, stitch them together into a
        three-dimensional numpy array of iamging data. Also calculate a 4x4 affine transformation
        matrix that converts the ijk-pixel-indices into the xyz-coordinates in the
        DICOM patient's coordinate system and 4x4 rotation and scaling matrix.
        If any of the DICOM images contain either the
        `Rescale Slope <https://dicom.innolitics.com/ciods/ct-image/ct-image/00281053>`__ or the
        `Rescale Intercept <https://dicom.innolitics.com/ciods/ct-image/ct-image/00281052>`__
        attributes they will be applied to each slice individually.
        This function requires that the datasets:

        - Be in same series (have the same
          `Series Instance UID <https://dicom.innolitics.com/ciods/ct-image/general-series/0020000e>`__,
          `Modality <https://dicom.innolitics.com/ciods/ct-image/general-series/00080060>`__,
          and `SOP Class UID <https://dicom.innolitics.com/ciods/ct-image/sop-common/00080016>`__).
        - The binary storage of each slice must be the same (have the same
          `Bits Allocated <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280100>`__,
          `Bits Stored <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280101>`__,
          `High Bit <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280102>`__, and
          `Pixel Representation <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280103>`__).
        - The image slice must approximately form a grid. This means there can not
          be any missing internal slices (missing slices on the ends of the dataset
          are not detected). It also means that  each slice must have the same
          `Rows <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280010>`__,
          `Columns <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280011>`__,
          `Pixel Spacing <https://dicom.innolitics.com/ciods/ct-image/image-plane/00280030>`__, and
          `Image Orientation (Patient) <https://dicom.innolitics.com/ciods/ct-image/image-plane/00200037>`__
          attribute values.
        - The direction cosines derived from the
          `Image Orientation (Patient) <https://dicom.innolitics.com/ciods/ct-image/image-plane/00200037>`__
          attribute must, within 1e-4, have a magnitude of 1.  The cosines must
          also be approximately perpendicular (their dot-product must be within
          1e-4 of 0).  Warnings are displayed if any of theseapproximations are
          below 1e-8, however, since we have seen real datasets with values up to
          1e-4, we let them pass.
        - The `Image Position (Patient) <https://dicom.innolitics.com/ciods/ct-image/image-plane/00200032>`__
          values must approximately form a line.
        
        If any of these conditions are not met, a `dicom_numpy.DicomImportException` is raised.

        Args:
            slice_datasets (List[pydicom.dataset.FileDataset]): List of dicom headers.
        Returns:
            List[numpy.ndarray]: List of numpy arrays containing the data extracted the dicom files
            (voxels, translation, rotation and scaling matrix).
        """

        if not slice_datasets:
            raise ValueError("Must provide at least one DICOM dataset")

        self.__validate_slices_form_uniform_grid(slice_datasets)

        voxels = self.__merge_slice_pixel_arrays(slice_datasets)
        transform, rotation, scaling = self.__ijk_to_patient_xyz_transform_matrix(
            slice_datasets)

        return voxels, transform, rotation, scaling

    def process_files(self):
        """
        Reads DICOM files (imaging volume + ROIs) in the instance data path 
        and then organizes it in the MEDscan class.
        
        Args:
            None.
        
        Returns:
            medscan (MEDscan): Instance of a MEDscan class.
        """
        
        return self.process_files_wrapper.remote(self)
    
    @ray.remote
    def process_files_wrapper(self) -> MEDscan:
        """
        Wrapper function to process the files.
        """

        # PARTIAL PARSING OF ARGUMENTS
        if self.path_images is None:
            raise ValueError('At least two arguments must be provided')

        # INITIALIZATION
        medscan = MEDscan()

        # IMAGING DATA AND ROI DEFINITION (if applicable)
        # Reading DICOM images and headers
        dicom_hi = [pydicom.dcmread(str(dicom_file), force=True)
                for dicom_file in self.path_images]

        try:
            # Determination of the scan orientation
            medscan.data.orientation = self.__get_dicom_scan_orientation(dicom_hi)

            # IMPORTANT NOTE: extract_voxel_data using combine_slices from dicom_numpy
            # missing slices and oblique restrictions apply see the reference:
            # https://dicom-numpy.readthedocs.io/en/latest/index.html#dicom_numpy.combine_slices
            try:
                voxel_ndarray, ijk_to_xyz, rotation_m, scaling_m = self.combine_slices(dicom_hi)
            except ValueError as e:
                raise ValueError(f'Invalid DICOM data for combine_slices(). Error: {e}')

            # Alignment of scan coordinates for MR scans
            # (inverse of ImageOrientationPatient rotation matrix)
            if not np.allclose(rotation_m, np.eye(rotation_m.shape[0])):
                medscan.data.volume.scan_rot = rotation_m

            medscan.data.volume.array = voxel_ndarray
            medscan.type = dicom_hi[0].Modality + 'scan'

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

            # Update the spatial reference in the MEDscan class
            medscan.data.volume.spatialRef = spatial_ref
            
            # DICOM HEADERS OF IMAGING DATA
            dicom_h = [
                pydicom.dcmread(str(dicom_file),stop_before_pixels=True,force=True) for dicom_file in self.path_images
                ]
            for i in range(0, len(dicom_h)):
                dicom_h[i].remove_private_tags()
            medscan.dicomH = dicom_h

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

                    medscan.data.ROI.update_roi_name(key=contour_num,
                                                    roi_name=dicom_rs_full[rs].StructureSetROISequence[roi].ROIName)
                    medscan.data.ROI.update_indexes(key=contour_num,
                                                    indexes=None)
                    medscan.data.ROI.update_name_set(key=contour_num,
                                                    name_set=name_set)
                    medscan.data.ROI.update_name_set_info(key=contour_num,
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
                        if n_closed_contour == 0:
                            print(f'Warning: no contour data found for ROI: \
                                  {dicom_rs_full[rs].StructureSetROISequence[roi].ROIName}')
                        else:
                            # Save the XYZ points in the MEDscan class
                            medscan.data.ROI.update_indexes(
                                                        key=contour_num, 
                                                        indexes=np.concatenate(
                                                                (points, 
                                                                np.reshape(ind_closed_contour, (len(ind_closed_contour), 1))),
                                                        axis=1)
                                                        )
                            # Compute the ROI box
                            _, roi_obj = get_roi(
                                            medscan,
                                            name_roi='{' + dicom_rs_full[rs].StructureSetROISequence[roi].ROIName + '}',
                                            box_string='full'
                                            )

                            # Save the ROI box non-zero indexes in the MEDscan class
                            medscan.data.ROI.update_indexes(key=contour_num, indexes=np.nonzero(roi_obj.data.flatten()))

                    except Exception as e:
                        if 'SeriesDescription' in dicom_h[0]:
                            print(f'patientID: {dicom_hi[0].PatientID} Modality: {dicom_hi[0].SeriesDescription} error: \
                                {str(e)} n_roi: {str(roi)}  n_rs: {str(rs)}')
                        else:
                            print(f'patientID: {dicom_hi[0].PatientID} Modality: {dicom_hi[0].Modality} error: \
                            {str(e)} n_roi: {str(roi)}  n_rs: {str(rs)}')
                        medscan.data.ROI.update_indexes(key=contour_num, indexes=np.NaN)
                    contour_num += 1

            # Save additional scan information in the MEDscan class
            medscan.data.set_patient_position(patient_position=dicom_h[0].PatientPosition)
            medscan.patientID = str(dicom_h[0].PatientID)
            medscan.format = "dicom"
            if 'SeriesDescription' in dicom_h[0]:
                medscan.series_description = dicom_h[0].SeriesDescription
            else:
                medscan.series_description = dicom_h[0].Modality

            # save MEDscan class instance as a pickle object
            if self.save and self.path_save:
                name_complete = save_MEDscan(medscan, self.path_save)
                del medscan
            else:
                series_description = medscan.series_description.translate({ord(ch): '-' for ch in '/\\ ()&:*'})
                name_id = medscan.patientID.translate({ord(ch): '-' for ch in '/\\ ()&:*'})

                # final saving name
                name_complete = name_id + '__' + series_description + '.' + medscan.type + '.npy'

        except Exception as e:
            if 'SeriesDescription' in dicom_hi[0]:
                print(f'patientID: {dicom_hi[0].PatientID} Modality: {dicom_hi[0].SeriesDescription} error: {str(e)}')
            else:
                print(f'patientID: {dicom_hi[0].PatientID} Modality: {dicom_hi[0].Modality} error: {str(e)}')
            return ''
        
        return name_complete
