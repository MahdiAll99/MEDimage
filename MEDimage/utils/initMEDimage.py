#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle

from MEDimage.MEDimage import MEDimage
from MEDimage.MEDimageComputeRadiomics import MEDimageComputeRadiomics
from MEDimage.MEDimageProcessing import MEDimageProcessing


def initMEDimage(nameRead, pathRead, roiType, imParams, log_file):
    """
    Initializes the MEDimage class and the child classes.

    Args:
        nameRead (str): name of the scan that will be used to
            initialize the MEDimage class and its children.
        pathRead (Path): Path to the scan file.
        roiType (str): ROI type.
        imParams (Dict): Dict of the test parameters.
        log_file (str): Name of the log file that will be used.

    Returns: 
        Derived classes (MEDimageProcessing and MEDimageComputeRadiomics).

    """ 
    if nameRead.endswith('.npy'):
        # MEDimage instance is now in Workspace
        with open(pathRead / nameRead, 'rb') as f: MEDimg = pickle.load(f)

        # MEDimageProcess instance is now in Workspace
        MEDimageProcess = MEDimageProcessing(MEDimg=MEDimg, log_file=pathRead / log_file)

        # MEDimageComputeRadiomics instance is now in Workspace
        MEDimageCR = MEDimageComputeRadiomics(MEDimg=MEDimg, log_file=MEDimageProcess.log_file)

        # Initialize processing & computation parameters
        MEDimageProcess.init_Params(imParamScan=imParams,
                                    imParamFilter=imParams['imParamFilter'],
                                    roiType=roiType)

        # Clear MEDimage instance # Clear up RAM
        MEDimg = None 

        return MEDimageProcess, MEDimageCR

    # Set up NIFTI Image path 
    NiftiImage = pathRead / nameRead

    # MEDimage instance is now in Workspace
    MEDimg = MEDimage()

    # Initialization using NIFTI file :
    MEDimg.init_from_nifti(NiftiImagePath=NiftiImage)

    # spatialRef Creation : 
    MEDimg.scan.volume.spatialRef_from_NIFTI(NiftiImage)

    # MEDimageProcess instance is now in Workspace
    MEDimageProcess = MEDimageProcessing(MEDimg=MEDimg, log_file=pathRead / log_file)

    # MEDimageComputeRadiomics instance is now in Workspace
    MEDimageCR = MEDimageComputeRadiomics(MEDimg=MEDimg, log_file=MEDimageProcess.log_file)

    # Initialize processing & computation parameters
    MEDimageProcess.init_Params(imParamScan=imParams,
                                imParamFilter=imParams['imParamFilter'],
                                roiType=roiType)

    # Clear MEDimage instance, clear up RAM
    MEDimg = None

    return MEDimageProcess, MEDimageCR
