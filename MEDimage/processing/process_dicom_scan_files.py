#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

import numpy as np
import pydicom
import pydicom.errors
import pydicom.misc
import pydicom.uid
import ray

warnings.simplefilter("ignore")

from pathlib import Path

from MEDimage.MEDimage import MEDimage
from MEDimage.MEDimageProcessing import MEDimageProcessing
from MEDimage.processing.getROI import getROI

from ..utils.save_MEDimage import save_MEDimage


@ray.remote
def process_dicom_scan_files(
                            pathImages: Path, 
                            pathRS: Path = None,
                            pathSave: Path = None
                            ) -> MEDimage:
    """
    Reads DICOM data according to the path info found in the
    input cells, and then organizes it in the MEDimage class.
    
    Args:
        pathSave (Path): String specifying the full path to the directory where to
            save all the MEDimage class created by the current
            function.
        pathImages (Path): Cell of strings, where each string specifies the full path
            to a DICOM image of single volume.
        pathRS: (Path, optional). Cell of strings, where each string specifies the
            full path to a DICOM RTstruct of a single volume.
                --> Options:- cellPathRS{1} from readAllDICOM.m
                            - Empty array or cell ([],{})
                            - No argument
        pathREG: (Path, optional). Cell of strings, where each string specifies the
            full path to a DICOM REG of a single volume.
                --> Options:- cellPathREG{1} from readAllDICOM.m
                            - Empty array or cell ([],{})
                            - No argument if 'pathRD', 'pathRP' and
                                'nameSave' are also not provided
        pathRD: (Path, optional). Cell of strings, where each string specifies the
            full path to a DICOM RTdose of a single volume.
                --> Options:- cellPathRD{1} from readAllDICOM.m
                            - Empty array or cell ([],{})
                            - No argument if 'pathRP' and 'nameSave' are also
                                not provided
        pathRP: (Path, optional). Cell of strings, where each string specifies the
            full path to a DICOM RTplan of a single volume.
                --> Options:- cellPathRP{1} from readAllDICOM.m
                            - Empty array or cell ([],{})
                            - No argument if 'nameSave' is also not provided
        nameSave: (str, optional). String specifying with what name the pickle object 
            file will be saved. If defined as 'modality', the Modality field
            of the DICOM headers of the imaging volume will also be used
            for 'nameSave'. The saving format is the following:
            '(PatientID)_(nameSave).(modality)scan.mat'
                --> Options:- User-defined. Ex: 'myScanName'
                            - No argument (default: 'SeriesDescription'
                                field of DICOM headers of imaging volume)
                            - 'modality'

    Returns:
        MEDimg (MEDimage): Instance of a MEDimage class.
    """

    # Since we created a worker, we need to add code path to the system
    import MEDimage.utils.combineSlices as cs
    import MEDimage.utils.imref as ref

    # PARTIAL PARSING OF ARGUMENTS
    if pathImages is None:
        raise ValueError('At least two arguments must be provided')

    # INITIALIZATION
    MEDimg = MEDimage()

    # IMAGING DATA AND ROI DEFINITION (if applicable)
    # Reading DICOM images and headers
    nSlices = len(pathImages)
    dicomHI = [pydicom.dcmread(str(dicom_file), force=True)
               for dicom_file in pathImages]

    # Determination of the scan orientation
    try:
        mid = round(nSlices/2)
        dist = [abs(dicomHI[mid+1].ImagePositionPatient[0] -
                    dicomHI[mid].ImagePositionPatient[0]),
                abs(dicomHI[mid+1].ImagePositionPatient[1] -
                    dicomHI[mid].ImagePositionPatient[1]),
                abs(dicomHI[mid+1].ImagePositionPatient[2] -
                    dicomHI[mid].ImagePositionPatient[2])]

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
            voxel_ndarray, ijk_to_xyz, rotation_m, scaling_m = cs.combineSlices(dicomHI)
        except ValueError:
            # invalid DICOM data
            raise ValueError('Invalid DICOM data for dicom_numpy.combine_slices')

        # Alignment of scan coordinates for MR scans
        # (inverse of ImageOrientationPatient rotation matrix)
        if not np.allclose(rotation_m, np.eye(rotation_m.shape[0])):
            MEDimg.scan.volume.scanRot = rotation_m

        MEDimg.scan.volume.data = voxel_ndarray
        MEDimg.type = dicomHI[0].Modality + 'scan'

        # 7. Creation of imref3d object
        pixelX = scaling_m[0, 0]
        pixelY = scaling_m[1, 1]
        sliceS = scaling_m[2, 2]
        min_grid = rotation_m@ijk_to_xyz[:3, 3]
        min_Xgrid = min_grid[0]
        min_Ygrid = min_grid[1]
        min_Zgrid = min_grid[2]
        size_image = np.shape(voxel_ndarray)
        spatialRef = ref.imref3d(size_image, pixelX, pixelY, sliceS)
        spatialRef.XWorldLimits = (np.array(spatialRef.XWorldLimits) -
                                   (spatialRef.XWorldLimits[0] -
                                    (min_Xgrid-pixelX/2))).tolist()
        spatialRef.YWorldLimits = (np.array(spatialRef.YWorldLimits) -
                                   (spatialRef.YWorldLimits[0] -
                                    (min_Ygrid-pixelY/2))).tolist()
        spatialRef.ZWorldLimits = (np.array(spatialRef.ZWorldLimits) -
                                   (spatialRef.ZWorldLimits[0] -
                                    (min_Zgrid-sliceS/2))).tolist()

        # Converting the results into lists
        spatialRef.ImageSize = spatialRef.ImageSize.tolist()
        spatialRef.XIntrinsicLimits = spatialRef.XIntrinsicLimits.tolist()
        spatialRef.YIntrinsicLimits = spatialRef.YIntrinsicLimits.tolist()
        spatialRef.ZIntrinsicLimits = spatialRef.ZIntrinsicLimits.tolist()

        MEDimg.scan.volume.spatialRef = spatialRef
        
        # DICOM HEADERS OF IMAGING DATA
        dicomH = [
            pydicom.dcmread(str(dicom_file),
                            stop_before_pixels=True,
                            force=True)
            for dicom_file in pathImages]

        for i in range(0, len(dicomH)):
            dicomH[i].remove_private_tags()

        MEDimg.dicomH = dicomH

        # DICOM RTstruct (if applicable)
        if pathRS is not None and len(pathRS) > 0:
            dicomRS_Full = [
                pydicom.dcmread(str(dicom_file),
                                stop_before_pixels=True,
                                force=True)
                for dicom_file in pathRS
            ]

            for i in range(0, len(dicomRS_Full)):
                dicomRS_Full[i].remove_private_tags()

        # GATHER XYZ POINTS OF ROIs USING RTstruct
        nRS = len(dicomRS_Full) if type(dicomRS_Full) is list else dicomRS_Full
        contourNum = 0
        for rs in range(nRS):
            nROI = len(dicomRS_Full[rs].StructureSetROISequence)
            for roi in range(nROI):
                if roi!=0:
                    if dicomRS_Full[rs].StructureSetROISequence[roi].ROIName == dicomRS_Full[rs].StructureSetROISequence[roi-1].ROIName:
                        continue
                points = []
                nameSetStrings = ['StructureSetName', 'StructureSetDescription',
                                  'SeriesDescription', 'SeriesInstanceUID']
                for name_field in nameSetStrings:
                    if name_field in dicomRS_Full[rs]:
                        nameSet = getattr(dicomRS_Full[rs], name_field)
                        nameSetInfo = name_field
                        break

                MEDimg.scan.ROI.update_ROIname(key=contourNum,
                                                ROIname=dicomRS_Full[rs].StructureSetROISequence[roi].ROIName)
                MEDimg.scan.ROI.update_indexes(key=contourNum,
                                                indexes=None)
                MEDimg.scan.ROI.update_nameSet(key=contourNum,
                                                nameSet=nameSet)
                MEDimg.scan.ROI.update_nameSetInfo(key=contourNum,
                                                nameSetInfo=nameSetInfo)
                
                try:
                    nClosedContours = len(dicomRS_Full[rs].ROIContourSequence[roi].ContourSequence)
                    indClosedContour = []
                    for s in range(0, nClosedContours):
                        pts_temp = dicomRS_Full[rs].ROIContourSequence[roi].ContourSequence[s].ContourData
                        # points stored in the RTstruct file for a given closed
                        # contour (beware: there can be multiple closed contours
                        # on a given slice).
                        nPoints = int(len(pts_temp) / 3)
                        # and isnumeric(pts_temp) SE THIS LINE TO TRANSLATE
                        if len(pts_temp) > 0:
                            indClosedContour = indClosedContour + np.tile(s, nPoints).tolist()
                            if type(points) == list:
                                points = np.reshape(
                                    np.transpose(pts_temp),
                                    (nPoints, 3))
                            else:
                                points = np.concatenate(
                                        (points, np.reshape(np.transpose(pts_temp), (nPoints, 3))),
                                        axis=0
                                        )
                    MEDimg.scan.ROI.update_indexes(
                                                key=contourNum, 
                                                indexes=np.concatenate(
                                                        (points, 
                                                        np.reshape(indClosedContour, (len(indClosedContour), 1))),
                                                axis=1)
                                                )

                    MEDImageProcess = MEDimageProcessing(MEDimg=MEDimg)

                    _, roiObj = getROI(
                                    MEDImageProcess,
                                    nameROI='{' + dicomRS_Full[rs].StructureSetROISequence[roi].ROIName + '}',
                                    boxString='full'
                                    )

                    MEDimg.scan.ROI.update_indexes(key=contourNum, indexes=np.nonzero(roiObj.data.flatten()))

                except Exception as e:
                    print('patientID: ' + dicomHI[0].PatientID + ' error: ' + str(e) + ' nROI: ' + str(roi) + ' nRS:' + str(rs))
                    MEDimg.scan.ROI.update_indexes(key=contourNum, indexes=np.NaN)
                contourNum += 1

        MEDimg.scan.patientPosition = MEDimg.dicomH[0].PatientPosition
        MEDimg.patientID = dicomH[0].PatientID

        # save MEDimage class instance as a pickle object
        if pathSave:
            save_MEDimage(MEDimg, dicomH[0].SeriesDescription, pathSave)

    except Exception as e:
        print('patientID: ' + dicomHI[0].PatientID + ' error: ' + str(e))
        return MEDimg
    
    return MEDimg
