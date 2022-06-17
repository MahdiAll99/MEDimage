#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle

from MEDimage.MEDimage import MEDimage


def initMEDimage(name_read, path_read, roi_type, im_params, log_file):
    """
    Initializes the MEDimage class and the child classes.

    Args:
        name_read (str): name of the scan that will be used to
            initialize the MEDimage class and its children.
        path_read (Path): Path to the scan file.
        roi_type (str): ROI type.
        im_params (Dict): Dict of the test parameters.
        log_file (str): Name of the log file that will be used.

    Returns: 
        Derived classes (MEDimageProcessing and MEDimageComputeRadiomics).

    """ 
    if name_read.endswith('.npy'):
        # MEDimage instance is now in Workspace
        with open(path_read / name_read, 'rb') as f: MEDimg = pickle.load(f)

        MEDimg = MEDimage(MEDimg, log_file)

        # Initialize processing & computation parameters
        MEDimg.init_params(imParamScan=im_params,
                                    imParamFilter=im_params['imParamFilter'],
                                    roi_type=roi_type)

        return MEDimg

    # Set up NIFTI Image path 
    nifti_image = path_read / name_read

    # MEDimage instance is now in Workspace
    MEDimg = MEDimage()

    # Initialization using NIFTI file :
    MEDimg.init_from_nifti(NiftiImagePath=nifti_image)

    # spatial_ref Creation : 
    MEDimg.scan.volume.spatial_ref_from_NIFTI(nifti_image)

    # Initialize processing & computation parameters
    MEDimg.init_Params(imParamScan=im_params,
                                imParamFilter=im_params['imParamFilter'],
                                roi_type=roi_type)

    return MEDimg
