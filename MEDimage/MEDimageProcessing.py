import logging
import os
import warnings
from pathlib import Path

import numpy as np

from MEDimage.MEDimage import MEDimage

_logger = logging.getLogger(__name__)


class MEDimageProcessing(MEDimage):
    """Organizes all processing parameters (patientID, imaging data, scan type...). 

    Args:
        MEDimg (MEDimage, optional): A MEDimage instance.
        log_file (str, optional): Name of the file that will be used
            for logging. MUST END WITH '.txt'

    Attributes:
        Params (Dict): Dict of parameters.
        results (Dict): Dict of results.
        nScale (int): Number of times we resample the voxel spacing.
        nAlgo (int): Number of texture discretisation algorithms.
        nGl (int): Number of gray levels.
        nExp (int): Equals to `nScale * nAlgo * nGl`.

    """
    __shared_state = {}
    
    def __init__(self, MEDimg=None, log_file=None):
        self.__dict__ = self.__shared_state
        super().__init__(MEDimg)
        self.Params = {}  # maybe it should be a class variable instead of attribute
        self.results = {} # maybe it should be a class variable instead of attribute
        self.nScale = 0 
        self.nAlgo = 0
        self.nGl = 0
        self.nExp = 0
        self.Continue = False

        if log_file is None: 
            warnings.warn("Log file is invalid, a log file will be created !") 
            self.log_file = Path(os.getcwd()) / 'log_file.txt'
        else: 
            self.log_file = log_file

    def init_Params(self, imParamScan, imParamFilter, **kwargs):

        try:
            boxString = 'box10'
            # 10 voxels in all three dimensions are added to the smallest
            # bounding box. This setting is used to speed up interpolation
            # processes (mostly) prior to the computation of radiomics
            # features. Optional argument in the function computeRadiomics.
            # PARSING LAST ARGUMENT

            # get default scan parameters from imParamScan
            radiomics = {}
            radiomics.update({'image': {}})
            scaleNonText = imParamScan['image']['interp']['scaleNonText']
            volInterp = imParamScan['image']['interp']['volInterp']
            roiInterp = imParamScan['image']['interp']['roiInterp']
            glRound = imParamScan['image']['interp']['glRound']
            roiPV = imParamScan['image']['interp']['roiPV']
            im_range = imParamScan['image']['reSeg']['range'] if 'range' in imParamScan['image']['reSeg'] else None
            outliers = imParamScan['image']['reSeg']['outliers']
            IH = imParamScan['image']['discretisation']['IH']
            IVH = imParamScan['image']['discretisation']['IVH']
            scaleText = imParamScan['image']['interp']['scaleText']
            algo = imParamScan['image']['discretisation']['texture']['type']
            grayLevels = imParamScan['image']['discretisation']['texture']['val']
            if self.type == 'PTscan':
                _computeSUVmap = imParamScan['image']['computeSUVmap']
            else :
                _computeSUVmap = False
            im_type = imParamScan['image']['type'] # TODO: Discover the usage of this variable!
            # Variable used to determine if there is 'arbitrary' (e.g., MRI)
            # or 'definite' (e.g., CT) intensities.
            intensity = imParamScan['image']['intensity']

            if 'distCorrection' in imParamScan['image']:
                distCorrection = imParamScan['image']['distCorrection']
            else:
                distCorrection = False

            if 'computeDiagFeatures' in imParamScan['image']:
                computeDiagFeatures = imParamScan['image']['computeDiagFeatures']
            else:
                computeDiagFeatures = False

            if computeDiagFeatures:  # If computeDiagFeatures is true.
                boxString = 'full'  # This is required for proper comparison.
                self.Params['boxString'] = boxString
            
            self.Params['radiomics'] = radiomics
            self.Params['filter'] = imParamFilter
            self.Params['radiomics']['imParam'] = imParamScan
            self.Params['scaleNonText'] = scaleNonText
            self.Params['volInterp'] = volInterp
            self.Params['roiInterp'] = roiInterp
            self.Params['glRound'] = glRound
            self.Params['roiPV'] = roiPV
            self.Params['im_range'] = im_range
            self.Params['outliers'] = outliers
            self.Params['IH'] = IH
            self.Params['IVH'] = IVH
            self.Params['scaleText'] = scaleText
            self.Params['algo'] = algo
            self.Params['grayLevels'] = grayLevels
            self.Params['im_type'] = im_type
            self.Params['intensity'] = intensity
            self.Params['computeDiagFeatures'] = computeDiagFeatures
            self.Params['distCorrection'] = distCorrection
            self.Params['boxString'] = boxString
            self.Params['scaleName'] = ''
            self.Params['IHname'] = ''
            self.Params['IVHname'] = ''

            for key, value in kwargs.items():
                try:
                    self.Params[key] = value
                except:
                    pass

            if self.Params['boxString'] is None:
                # boxString argument is optional. If not present, we use the full box.
                self.Params['boxString'] = 'full'

            # *******************************
            # ** SETTING UP userSetMinVal  **
            # *******************************

            if self.Params['im_range'] is not None and type(self.Params['im_range']) is list and self.Params['im_range']:
                userSetMinVal = self.Params['im_range'][0]
                if userSetMinVal == -np.inf:
                    # In case no re-seg im_range is defined for the FBS algorithm,
                    # the minimum value of ROI will be used (not recommended).
                    userSetMinVal = []
            else:
                # In case no re-seg im_range is defined for the FBS algorithm,
                # the minimum value of ROI will be used (not recommended).
                userSetMinVal = [] 

            self.Params['userSetMinVal'] = userSetMinVal
            self.nScale = len(self.Params['scaleText'])
            self.nAlgo = len(self.Params['algo'])
            self.nGl = len(self.Params['grayLevels'][0])
            self.nExp = self.nScale * self.nAlgo * self.nGl

            if self.type == 'PTscan' and _computeSUVmap:
                try:
                    self.scan.volume.data = self.computeSUVmap(self.scan.volume.data,self.dicomH[0])
                
                except Exception as e :
                    message = "\n ERROR COMPUTING SUV MAP - SOME FEATURES " \
                            "WILL BE INVALID: \n {}".format(e)
                    print(message)
                    _logger.error(message)
                    self.Continue = True

        except Exception as e:
            message = "\n ERROR IN INITIALIZATION OF RADIOMICS FEATURE " \
                      "COMPUTATION\n {}".format(e)
            print(message)
            _logger.error(message)
            self.Continue = True
