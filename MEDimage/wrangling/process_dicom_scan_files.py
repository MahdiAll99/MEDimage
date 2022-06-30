#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

import numpy as np
import pydicom
import ray
from MEDimage.utils.imref import imref3d

warnings.simplefilter("ignore")

from pathlib import Path

from MEDimage.MEDimage import MEDimage

from ..processing.get_roi import get_roi
from ..utils.save_MEDimage import save_MEDimage


@ray.remote
def process_dicom_scan_files(
                            path_images: Path, 
                            path_rs: Path = None,
                            path_save: Path = None,
                            save: bool = True
                            ) -> MEDimage:
    """
    Reads DICOM data according to the path info found in the
    input cells, and then organizes it in the MEDimage class.
    
    Args:
        path_save (Path): String specifying the full path to the directory where to
            save all the MEDimage class created by the current
            function.
        path_images (Path): Cell of strings, where each string specifies the full path
            to a DICOM image of single volume.
        path_rs: (Path, optional). Cell of strings, where each string specifies the
            full path to a DICOM RTstruct of a single volume.
                --> Options:- cell_path_rs{1} from readAllDICOM.m
                            - Empty array or cell ([],{})
                            - No argument
    Returns:
        MEDimg (MEDimage): Instance of a MEDimage class.
    """
    # Since we created a worker, we need to add code path to the system
    from .combine_slices import combine_slices

    # PARTIAL PARSING OF ARGUMENTS
    if path_images is None:
        raise ValueError('At least two arguments must be provided')

    # INITIALIZATION
    MEDimg = MEDimage()

    # IMAGING DATA AND ROI DEFINITION (if applicable)
    # Reading DICOM images and headers
    n_slices = len(path_images)
    dicom_hi = [pydicom.dcmread(str(dicom_file), force=True)
               for dicom_file in path_images]

    # Determination of the scan orientation
    try:
        mid = round(n_slices/2)
        dist = [abs(dicom_hi[mid+1].ImagePositionPatient[0] - dicom_hi[mid].ImagePositionPatient[0]),
                abs(dicom_hi[mid+1].ImagePositionPatient[1] - dicom_hi[mid].ImagePositionPatient[1]),
                abs(dicom_hi[mid+1].ImagePositionPatient[2] - dicom_hi[mid].ImagePositionPatient[2])]

        index = dist.index(max(dist))
        if index == 0:
            orientation = 'Sagittal'
        elif index == 1:
            orientation = 'Coronal'
        else:
            orientation = 'Axial'

        MEDimg.scan.orientation = orientation

        # IMPORTANT NOTE: extract_voxel_data using combine_slices from dicom_numpy
        # missing slices and oblique restrictions apply see the reference:
        # https://dicom-numpy.readthedocs.io/en/latest/index.html#dicom_numpy.combine_slices
        try:
            voxel_ndarray, ijk_to_xyz, rotation_m, scaling_m = combine_slices(dicom_hi)
        except ValueError:
            raise ValueError('Invalid DICOM data for dicom_numpy.combine_slices')

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

        MEDimg.scan.volume.spatialRef = spatial_ref
        
        # DICOM HEADERS OF IMAGING DATA
        dicom_h = [
            pydicom.dcmread(str(dicom_file),
                            stop_before_pixels=True,
                            force=True)
            for dicom_file in path_images]

        for i in range(0, len(dicom_h)):
            dicom_h[i].remove_private_tags()

        MEDimg.dicomH = dicom_h

        # DICOM RTstruct (if applicable)
        if path_rs is not None and len(path_rs) > 0:
            dicom_rs_full = [
                pydicom.dcmread(str(dicom_file),
                                stop_before_pixels=True,
                                force=True)
                for dicom_file in path_rs
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

                MEDimg.scan.ROI.update_ROIname(key=contour_num,
                                                ROIname=dicom_rs_full[rs].StructureSetROISequence[roi].ROIName)
                MEDimg.scan.ROI.update_indexes(key=contour_num,
                                                indexes=None)
                MEDimg.scan.ROI.update_nameSet(key=contour_num,
                                                nameSet=name_set)
                MEDimg.scan.ROI.update_nameSetInfo(key=contour_num,
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
                    MEDimg.scan.ROI.update_indexes(
                                                key=contour_num, 
                                                indexes=np.concatenate(
                                                        (points, 
                                                        np.reshape(ind_closed_contour, (len(ind_closed_contour), 1))),
                                                axis=1)
                                                )
                    _, roi_obj = get_roi(
                                    MEDimg,
                                    name_roi='{' + dicom_rs_full[rs].StructureSetROISequence[roi].ROIName + '}',
                                    box_string='full'
                                    )

                    MEDimg.scan.ROI.update_indexes(key=contour_num, indexes=np.nonzero(roi_obj.data.flatten()))

                except Exception as e:
                    print('patientID: ' + dicom_hi[0].patientID + ' error: ' + str(e) + ' n_roi: ' + str(roi) + ' n_rs:' + str(rs))
                    MEDimg.scan.ROI.update_indexes(key=contour_num, indexes=np.NaN)
                contour_num += 1

        MEDimg.scan.patientPosition = dicom_h[0].PatientPosition
        MEDimg.patientID = str(dicom_h[0].PatientID)

        # save MEDimage class instance as a pickle object
        if save and path_save:
            try:
                save_MEDimage(MEDimg, dicom_h[0].SeriesDescription, path_save)
            except:
                save_MEDimage(MEDimg, dicom_h[0].Modality, path_save)

    except Exception as e:
        print('patientID: ' + dicom_hi[0].patientID + ' error: ' + str(e))
        return MEDimg
    
    return MEDimg
