import logging
import math
import os
import warnings
from pathlib import Path

import numpy as np

from MEDimage.MEDimageFilter import Gabor, LaplacianOfGaussian, Laws, Mean, Wavelet
from MEDimage.MEDimage import MEDimage

_logger = logging.getLogger(__name__)


class MEDimageProcessing(MEDimage):
    """Organizes all processing parameters (patient_id, imaging data, scan type...). 

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
            box_string = 'box10'
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
                _computeSUVmap = imParamScan['image']['compute_suv_map']
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
                box_string = 'full'  # This is required for proper comparison.
                self.Params['box_string'] = box_string
            
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
            self.Params['box_string'] = box_string
            self.Params['scaleName'] = ''
            self.Params['IHname'] = ''
            self.Params['IVHname'] = ''

            for key, value in kwargs.items():
                try:
                    self.Params[key] = value
                except:
                    pass

            if self.Params['box_string'] is None:
                # box_string argument is optional. If not present, we use the full box.
                self.Params['box_string'] = 'full'

            # *******************************
            # ** SETTING UP user_set_min_val  **
            # *******************************

            if self.Params['im_range'] is not None and type(self.Params['im_range']) is list and self.Params['im_range']:
                user_set_min_val = self.Params['im_range'][0]
                if user_set_min_val == -np.inf:
                    # In case no re-seg im_range is defined for the FBS algorithm,
                    # the minimum value of ROI will be used (not recommended).
                    user_set_min_val = []
            else:
                # In case no re-seg im_range is defined for the FBS algorithm,
                # the minimum value of ROI will be used (not recommended).
                user_set_min_val = [] 

            self.Params['user_set_min_val'] = user_set_min_val
            self.nScale = len(self.Params['scaleText'])
            self.nAlgo = len(self.Params['algo'])
            self.nGl = len(self.Params['grayLevels'][0])
            self.nExp = self.nScale * self.nAlgo * self.nGl

            if self.type == 'PTscan' and _computeSUVmap:
                try:
                    self.scan.volume.data = self.compute_suv_map(self.scan.volume.data,self.dicom_h[0])
                
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

    def applyFilter(self, filterType, vol_obj):
        """Applies filter (depending on the filter name) on the given volume object
        and returns new filtred image.
        
        Args:
            filterType (str): Name of the filter to use (Mean, Laws, Wavelet...).
            vol_obj (imref3d): Volume object containing the data that will be filterd.

        Returns:
            ndarray: vol_obj: 3D array of filtered imaging data.
        """
        
        VOLEX_LENGTH = self.Params['scaleNonText'][0]
        input = np.expand_dims(vol_obj.data.astype(np.float64), axis=0)   # Convert to shape : (B, W, H, D)
        params = self.Params['filter']

        if filterType.lower() == "mean":
            params = params['Mean']
            _filter = Mean(ndims=params['ndims'], size=params['size'], padding=params['padding'])
            result = _filter.convolve(input)

        elif filterType.lower() == "log":
            params = params['LoG']
            sigma = params['sigma'] / VOLEX_LENGTH
            length = 2 * int(4 * params['sigma'] + 0.5) + 1
            _filter = LaplacianOfGaussian(ndims=params['ndims'], size=length, sigma=sigma, padding=params['padding'])
            result = _filter.convolve(input, 
                            orthogonal_rot=params['orthogonal_rot']
                            )

        elif filterType.lower() == "laws":
            params = params['Laws']
            _filter = Laws(params['config'], 
                            energy_distance=params['energy_distance'], 
                            rot_invariance=params['rot_invariance'], 
                            padding=params['padding']
                            )
            result = _filter.convolve(input, 
                            orthogonal_rot=params['orthogonal_rot'],
                            energy_image=params['energy_image']
                            )
            if params['energy_image']:
                result = result[1]
        
        elif filterType.lower() == "gabor":
            params = params['Gabor']
            sigma = params['sigma'] / VOLEX_LENGTH
            lamb = params['lambda'] / VOLEX_LENGTH
            size = 2 * int(7 * params['sigma'] + 0.5) + 1
            if type(params["theta"]) is str and params["theta"].startswith('Pi/'):
                theta = math.pi / int(params["theta"].split('/')[1])
            else:
                theta = float(params["theta"])

            _filter = Gabor(size=size, 
                            sigma=sigma, 
                            lamb=lamb,
                            gamma=params['gamma'], 
                            theta=-theta,
                            rot_invariance=params['rot_invariance'],
                            padding=params['padding']
                            )
            result = _filter.convolve(input, orthogonal_rot=params['orthogonal_rot'])
        
        elif filterType.lower().startswith("wavelet"):
            params = params[filterType]
            _filter = Wavelet(ndims=params['ndims'], 
                            wavelet_name=params['basis_function'],
                            rot_invariance=params['rot_invariance'],
                            padding=params['padding']
                            )
            result = _filter.convolve(input, _filter=params['subband'], level=params['level'])
        
        else:
            raise ValueError(
                    r'Filter name should either be: "mean", "log", "laws", "gabor" or "wavelet".')

        vol_obj.data = np.squeeze(result)

        return vol_obj
