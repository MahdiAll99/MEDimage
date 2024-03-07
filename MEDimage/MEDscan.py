import logging
import os
from json import dump
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from numpyencoder import NumpyEncoder
from PIL import Image

from .utils.image_volume_obj import image_volume_obj
from .utils.imref import imref3d
from .utils.json_utils import load_json


class MEDscan(object):
    """Organizes all scan data (patientID, imaging data, scan type...). 

    Attributes:
        patientID (str): Patient ID.
        type (str): Scan type (MRscan, CTscan...).
        format (str): Scan file format. Either 'npy' or 'nifti'.
        dicomH (pydicom.dataset.FileDataset): DICOM header.
        data (MEDscan.data): Instance of MEDscan.data inner class.

    """

    def __init__(self, medscan=None) -> None:
        """Constructor of the MEDscan class

        Args:
            medscan(MEDscan): A MEDscan class instance.
        
        Returns:
            None
        """
        try:
            self.patientID = medscan.patientID
        except:
            self.patientID = ""
        try:
            self.type = medscan.type
        except:
            self.type = ""
        try:
            self.series_description = medscan.series_description
        except:
            self.series_description = ""
        try:
            self.format = medscan.format
        except:
            self.format = ""
        try:
            self.dicomH = medscan.dicomH
        except:
            self.dicomH = []
        try:
            self.data = medscan.data
        except:
            self.data = self.data()
        try:
            self.params = medscan.params
        except:
            self.params = self.Params()
        try:
            self.radiomics = medscan.radiomics
        except:
            self.radiomics = self.Radiomics()

        self.skip = False

    def __init_process_params(self, im_params: Dict) -> None:
        """Initializes the processing params from a given Dict.
        
        Args:
            im_params(Dict): Dictionary of different processing params.
        
        Returns:
            None.
        """
        if self.type == 'CTscan' and 'imParamCT' in im_params:
            im_params = im_params['imParamCT']
        elif self.type == 'MRscan' and 'imParamMR' in im_params:
            im_params = im_params['imParamMR']
        elif self.type == 'PTscan' and 'imParamPET' in im_params:
            im_params = im_params['imParamPET']
        else:
            raise ValueError(f"The given parameters dict is not valid, no params found for {self.type} modality")
        
        # re-segmentation range processing
        if(im_params['reSeg']['range'] and im_params['reSeg']['range'][1] == "inf"):
            im_params['reSeg']['range'][1] = np.inf

        if 'box_string' in im_params:
            box_string = im_params['box_string']
        else:
            # By default, we add 10 voxels in all three dimensions are added to the smallest
            # bounding box. This setting is used to speed up interpolation
            # processes (mostly) prior to the computation of radiomics
            # features. Optional argument in the function computeRadiomics.
            box_string = 'box10'
        if 'compute_diag_features' in im_params:
            compute_diag_features = im_params['compute_diag_features']
        else:
            compute_diag_features = False
        if compute_diag_features:  # If compute_diag_features is true.
            box_string = 'full'  # This is required for proper comparison.

        self.params.process.box_string = box_string

        # get default scan parameters from im_param_scan
        self.params.process.scale_non_text = im_params['interp']['scale_non_text']
        self.params.process.vol_interp  = im_params['interp']['vol_interp']
        self.params.process.roi_interp = im_params['interp']['roi_interp']
        self.params.process.gl_round = im_params['interp']['gl_round']
        self.params.process.roi_pv  = im_params['interp']['roi_pv']
        self.params.process.im_range = im_params['reSeg']['range'] if 'range' in im_params['reSeg'] else None
        self.params.process.outliers = im_params['reSeg']['outliers']
        self.params.process.ih = im_params['discretisation']['IH']
        self.params.process.ivh = im_params['discretisation']['IVH']
        self.params.process.scale_text = im_params['interp']['scale_text']
        self.params.process.algo = im_params['discretisation']['texture']['type'] if 'type' in im_params['discretisation']['texture'] else []
        self.params.process.gray_levels = im_params['discretisation']['texture']['val'] if 'val' in im_params['discretisation']['texture'] else [[]]
        self.params.process.im_type = self.type

        # Voxels dimension
        self.params.process.n_scale = len(self.params.process.scale_text)
        # Setting up discretisation params
        self.params.process.n_algo = len(self.params.process.algo)
        self.params.process.n_gl = len(self.params.process.gray_levels[0])
        self.params.process.n_exp = self.params.process.n_scale * self.params.process.n_algo * self.params.process.n_gl

        # Setting up user_set_min_value
        if self.params.process.im_range is not None and type(self.params.process.im_range) is list and self.params.process.im_range:
            user_set_min_value = self.params.process.im_range[0]
            if user_set_min_value == -np.inf:
                # In case no re-seg im_range is defined for the FBS algorithm,
                # the minimum value of ROI will be used (not recommended).
                user_set_min_value = []
        else:
            # In case no re-seg im_range is defined for the FBS algorithm,
            # the minimum value of ROI will be used (not recommended).
            user_set_min_value = [] 
        self.params.process.user_set_min_value = user_set_min_value

        # box_string argument is optional. If not present, we use the full box.
        if self.params.process.box_string is None:
            self.params.process.box_string = 'full'
        
        # set filter type for the modality
        if 'filter_type' in im_params:
            self.params.filter.filter_type = im_params['filter_type']
        
        # Set intensity type
        if 'intensity_type' in im_params and im_params['intensity_type'] != "":
            self.params.process.intensity_type = im_params['intensity_type']
        elif self.params.filter.filter_type != "":
            self.params.process.intensity_type = 'filtered'
        elif self.type == 'MRscan':
            self.params.process.intensity_type = 'arbitrary'
        else:
            self.params.process.intensity_type = 'definite'

    def __init_extraction_params(self, im_params: Dict):
        """Initializes the extraction params from a given Dict.
        
        Args:
            im_params(Dict): Dictionary of different extraction params.
        
        Returns:
            None.
        """
        if self.type == 'CTscan' and 'imParamCT' in im_params:
            im_params = im_params['imParamCT']
        elif self.type == 'MRscan' and 'imParamMR' in im_params:
            im_params = im_params['imParamMR']
        elif self.type == 'PTscan' and 'imParamPET' in im_params:
            im_params = im_params['imParamPET']
        else:
            raise ValueError(f"The given parameters dict is not valid, no params found for {self.type} modality")
        
        # glcm features extraction params
        if 'glcm' in im_params:
            if 'dist_correction' in im_params['glcm']:
                self.params.radiomics.glcm.dist_correction = im_params['glcm']['dist_correction']
            else:
                self.params.radiomics.glcm.dist_correction = False
            if 'merge_method' in im_params['glcm']:
                self.params.radiomics.glcm.merge_method = im_params['glcm']['merge_method']
            else:
                self.params.radiomics.glcm.merge_method = "vol_merge"
        else:
            self.params.radiomics.glcm.dist_correction = False
            self.params.radiomics.glcm.merge_method = "vol_merge"

        # glrlm features extraction params
        if 'glrlm' in im_params:
            if 'dist_correction' in im_params['glrlm']:
                self.params.radiomics.glrlm.dist_correction = im_params['glrlm']['dist_correction']
            else:
                self.params.radiomics.glrlm.dist_correction = False
            if 'merge_method' in im_params['glrlm']:
                self.params.radiomics.glrlm.merge_method = im_params['glrlm']['merge_method']
            else:
                self.params.radiomics.glrlm.merge_method = "vol_merge"
        else:
            self.params.radiomics.glrlm.dist_correction = False
            self.params.radiomics.glrlm.merge_method = "vol_merge"


        # ngtdm features extraction params
        if 'ngtdm' in im_params:
            if 'dist_correction' in im_params['ngtdm']:
                self.params.radiomics.ngtdm.dist_correction = im_params['ngtdm']['dist_correction']
            else:
                self.params.radiomics.ngtdm.dist_correction = False
        else:
            self.params.radiomics.ngtdm.dist_correction = False

    def __init_filter_params(self, filter_params: Dict) -> None:
        """Initializes the filtering params from a given Dict.

        Args:
            filter_params(Dict): Dictionary of the filtering parameters.
        
        Returns:
            None.
        """
        if 'imParamFilter' in filter_params:
            filter_params = filter_params['imParamFilter']
        
        # Initializae filter attribute
        self.params.filter = self.params.Filter()

        # mean filter params
        if 'mean' in filter_params:
            self.params.filter.mean.init_from_json(filter_params['mean'])

        # log filter params
        if 'log' in filter_params:
            self.params.filter.log.init_from_json(filter_params['log'])

        # laws filter params
        if 'laws' in filter_params:
            self.params.filter.laws.init_from_json(filter_params['laws'])

        # gabor filter params
        if 'gabor' in filter_params:
            self.params.filter.gabor.init_from_json(filter_params['gabor'])

        # wavelet filter params
        if 'wavelet' in filter_params:
            self.params.filter.wavelet.init_from_json(filter_params['wavelet'])

        # Textural filter params
        if 'textural' in filter_params:
            self.params.filter.textural.init_from_json(filter_params['textural'])

    def init_params(self, im_param_scan: Dict) -> None:
        """Initializes the Params class from a dictionary.

        Args:
            im_param_scan(Dict): Dictionary of different processing, extraction and filtering params.
        
        Returns:
            None.
        """
        try:
            # get default scan parameters from im_param_scan
            self.__init_filter_params(im_param_scan['imParamFilter'])
            self.__init_process_params(im_param_scan)
            self.__init_extraction_params(im_param_scan)

            # compute suv map for PT scans
            if self.type == 'PTscan':
                _compute_suv_map = im_param_scan['imParamPET']['compute_suv_map']
            else :
                _compute_suv_map = False
            
            if self.type == 'PTscan' and _compute_suv_map and self.format != 'nifti':
                try:
                    from .processing.compute_suv_map import compute_suv_map
                    self.data.volume.array = compute_suv_map(self.data.volume.array, self.dicomH[0])
                except Exception as e :
                    message = f"\n ERROR COMPUTING SUV MAP - SOME FEATURES WILL BE INVALID: \n {e}"
                    logging.error(message)
                    print(message)
                    self.skip = True
            
            # initialize radiomics structure
            self.radiomics.image = {}
            self.radiomics.params = im_param_scan
            self.params.radiomics.scale_name = ''
            self.params.radiomics.ih_name = ''
            self.params.radiomics.ivh_name = ''
            
        except Exception as e:
            message = f"\n ERROR IN INITIALIZATION OF RADIOMICS FEATURE COMPUTATION\n {e}"
            logging.error(message)
            print(message)
            self.skip = True

    def init_ntf_calculation(self, vol_obj: image_volume_obj) -> None:
        """
        Initializes all the computation parameters for non-texture features  as well as the results dict.

        Args:
            vol_obj(image_volume_obj): Imaging volume.
        
        Returns:
            None.
        """
        try:
            if sum(self.params.process.scale_non_text) == 0:  # In case the user chose to not interpolate
                self.params.process.scale_non_text = [
                                        vol_obj.spatialRef.PixelExtentInWorldX,
                                        vol_obj.spatialRef.PixelExtentInWorldY,
                                        vol_obj.spatialRef.PixelExtentInWorldZ]
            else:
                if len(self.params.process.scale_non_text) == 2:
                    # In case not interpolation is performed in
                    # the slice direction (e.g. 2D case)
                    self.params.process.scale_non_text = self.params.process.scale_non_text + \
                        [vol_obj.spatialRef.PixelExtentInWorldZ]

            # Scale name
            # Always isotropic resampling, so the first entry is ok.
            self.params.radiomics.scale_name = 'scale' + (str(self.params.process.scale_non_text[0])).replace('.', 'dot')

            # IH name
            if 'val' in self.params.process.ih:
                ih_val_name = 'bin' + (str(self.params.process.ih['val'])).replace('.', 'dot')
            else:
                ih_val_name = 'binNone'

            # The minimum value defines the computation.
            if self.params.process.ih['type'].find('FBS')>=0:
                if type(self.params.process.user_set_min_value) is list and self.params.process.user_set_min_value:
                    min_val_name = '_min' + \
                        ((str(self.params.process.user_set_min_value)).replace('.', 'dot')).replace('-', 'M')
                else:
                    # Otherwise, minimum value of ROI will be used (not recommended),
                    # so no need to report it.
                    min_val_name = ''
            else:
                min_val_name = ''
            self.params.radiomics.ih_name = self.params.radiomics.scale_name + \
                                            '_algo' + self.params.process.ih['type'] + \
                                            '_' + ih_val_name + min_val_name

            # IVH name
            if not self.params.process.ivh:  # CT case
                ivh_algo_name = 'algoNone'
                ivh_val_name = 'bin1'
                if self.params.process.im_range:  # The im_range defines the computation.
                    min_val_name = ((str(self.params.process.im_range[0])).replace(
                        '.', 'dot')).replace('-', 'M')
                    max_val_name = ((str(self.params.process.im_range[1])).replace(
                        '.', 'dot')).replace('-', 'M')
                    range_name = '_min' + min_val_name + '_max' + max_val_name
                else:
                    range_name = ''
            else:
                ivh_algo_name = 'algo' + self.params.process.ivh['type']
                if 'val' in self.params.process.ivh:
                    ivh_val_name = 'bin' + (str(self.params.process.ivh['val'])).replace('.', 'dot')
                else:
                    ivh_val_name = 'binNone'
                # The im_range defines the computation.
                if 'type' in self.params.process.ivh and self.params.process.ivh['type'].find('FBS') >=0:
                    if self.params.process.im_range:
                        min_val_name = ((str(self.params.process.im_range[0])).replace(
                            '.', 'dot')).replace('-', 'M')
                        max_val_name = ((str(self.params.process.im_range[1])).replace(
                            '.', 'dot')).replace('-', 'M')
                        if max_val_name == 'inf':
                            # In this case, the maximum value of the ROI is used,
                            # so no need to report it.
                            range_name = '_min' + min_val_name
                        elif min_val_name == '-inf':
                            # In this case, the minimum value of the ROI is used,
                            # so no need to report it.
                            range_name = '_max' + max_val_name
                        else:
                            range_name = '_min' + min_val_name + '_max' + max_val_name
                    else:  # min-max of ROI will be used, no need to report it.
                        range_name = ''
                else:  # min-max of ROI will be used, no need to report it.
                    range_name = ''
            self.params.radiomics.ivh_name = self.params.radiomics.scale_name + '_' + ivh_algo_name + '_' + ivh_val_name + range_name

            # Now initialize the attribute that will hold the computation results
            self.radiomics.image.update({ 
                            'morph_3D': {self.params.radiomics.scale_name: {}},
                            'locInt_3D': {self.params.radiomics.scale_name: {}},
                            'stats_3D': {self.params.radiomics.scale_name: {}},
                            'intHist_3D': {self.params.radiomics.ih_name: {}},
                            'intVolHist_3D': {self.params.radiomics.ivh_name: {}} 
                            })

        except Exception as e:
            message = f"\n PROBLEM WITH PRE-PROCESSING OF FEATURES IN init_ntf_calculation(): \n {e}"
            logging.error(message)
            print(message)
            self.radiomics.image.update(
                    {('scale' + (str(self.params.process.scale_non_text[0])).replace('.', 'dot')): 'ERROR_PROCESSING'})

    def init_tf_calculation(self, algo:int, gl:int, scale:int) -> None:
        """
        Initializes all the computation parameters for the texture-features as well as the results dict.

        Args:
            algo(int): Discretisation algorithms index.
            gl(int): gray-level index.
            scale(int): scale-text index.
        
        Returns:
            None.
        """
        # check glcm merge method
        glcm_merge_method = self.params.radiomics.glcm.merge_method
        if glcm_merge_method:
            if glcm_merge_method == 'average':
                glcm_merge_method = '_avg'
            elif glcm_merge_method == 'vol_merge':
                glcm_merge_method = '_comb'
            else:
                error_msg = f"{glcm_merge_method} Method not supported in glcm computation, \
                    only 'average' or 'vol_merge' are supported. \
                    Radiomics will be saved without any specific merge method."
                logging.warning(error_msg)
                print(error_msg)

        # check glrlm merge method
        glrlm_merge_method = self.params.radiomics.glrlm.merge_method
        if glrlm_merge_method:
            if glrlm_merge_method == 'average':
                glrlm_merge_method = '_avg'
            elif glrlm_merge_method == 'vol_merge':
                glrlm_merge_method = '_comb'
            else:
                error_msg = f"{glcm_merge_method} Method not supported in glrlm computation, \
                    only 'average' or 'vol_merge' are supported. \
                    Radiomics will be saved without any specific merge method"
                logging.warning(error_msg)
                print(error_msg)
        # set texture features names and updates radiomics dict
        self.params.radiomics.name_text_types = [
                                'glcm_3D' + glcm_merge_method, 
                                'glrlm_3D' + glrlm_merge_method, 
                                'glszm_3D', 
                                'gldzm_3D', 
                                'ngtdm_3D', 
                                'ngldm_3D']
        n_text_types = len(self.params.radiomics.name_text_types)
        if not ('texture' in self.radiomics.image):
            self.radiomics.image.update({'texture': {}})
            for t in range(n_text_types):
                self.radiomics.image['texture'].update({self.params.radiomics.name_text_types[t]: {}})

        # scale name
        # Always isotropic resampling, so the first entry is ok.
        scale_name = 'scale' + (str(self.params.process.scale_text[scale][0])).replace('.', 'dot')
        if hasattr(self.params.radiomics, "scale_name"):
            setattr(self.params.radiomics, 'scale_name', scale_name)
        else:
            self.params.radiomics.scale_name = scale_name

        # Discretisation name
        gray_levels_name = (str(self.params.process.gray_levels[algo][gl])).replace('.', 'dot')

        if 'FBS' in self.params.process.algo[algo]:  # The minimum value defines the computation.
            if type(self.params.process.user_set_min_value) is list and self.params.process.user_set_min_value:
                min_val_name = '_min' + ((str(self.params.process.user_set_min_value)).replace('.', 'dot')).replace('-', 'M')
            else:
                # Otherwise, minimum value of ROI will be used (not recommended),
                # so no need to report it.
                min_val_name = ''
        else:
            min_val_name = ''

        if 'equal'in self.params.process.algo[algo]:
            # The number of gray-levels used for equalization is currently
            # hard-coded to 64 in equalization.m
            discretisation_name = 'algo' + self.params.process.algo[algo] + '256_bin' + gray_levels_name + min_val_name
        else:
            discretisation_name = 'algo' + self.params.process.algo[algo] + '_bin' + gray_levels_name + min_val_name

        # Processing full name
        processing_name = scale_name + '_' + discretisation_name
        if hasattr(self.params.radiomics, "processing_name"):
            setattr(self.params.radiomics, 'processing_name', processing_name)
        else:
            self.params.radiomics.processing_name = processing_name

    def init_from_nifti(self, nifti_image_path: Path) -> None:
        """Initializes the MEDscan class using a NIfTI file.

        Args:
            nifti_image_path (Path): NIfTI file path.

        Returns:
            None.
        
        """
        self.patientID = os.path.basename(nifti_image_path).split("_")[0]
        self.type = os.path.basename(nifti_image_path).split(".")[-3]
        self.format = "nifti"
        self.data.set_orientation(orientation="Axial")
        self.data.set_patient_position(patient_position="HFS")
        self.data.ROI.get_roi_from_path(roi_path=os.path.dirname(nifti_image_path), 
                                        id=Path(nifti_image_path).name.split("(")[0])
        self.data.volume.array = nib.load(nifti_image_path).get_fdata()
        # RAS to LPS
        self.data.volume.convert_to_LPS()
        self.data.volume.scan_rot = None
    
    def update_radiomics(
                        self, int_vol_hist_features: Dict = {}, 
                        morph_features: Dict = {}, loc_int_features: Dict = {}, 
                        stats_features: Dict = {}, int_hist_features: Dict = {},
                        glcm_features: Dict = {}, glrlm_features: Dict = {},
                        glszm_features: Dict = {}, gldzm_features: Dict = {}, 
                        ngtdm_features: Dict = {}, ngldm_features: Dict = {}) -> None:
        """Updates the results attribute with the extracted features.

        Args:
            int_vol_hist_features(Dict, optional): Dictionary of the intensity volume histogram features.
            morph_features(Dict, optional): Dictionary of the morphological features.
            loc_int_features(Dict, optional): Dictionary of the intensity local intensity features.
            stats_features(Dict, optional): Dictionary of the statistical features.
            int_hist_features(Dict, optional): Dictionary of the intensity histogram features.
            glcm_features(Dict, optional): Dictionary of the GLCM features.
            glrlm_features(Dict, optional): Dictionary of the GLRLM features.
            glszm_features(Dict, optional): Dictionary of the GLSZM features.
            gldzm_features(Dict, optional): Dictionary of the GLDZM features.
            ngtdm_features(Dict, optional): Dictionary of the NGTDM features.
            ngldm_features(Dict, optional): Dictionary of the NGLDM features.
        Returns:
            None.
        """
        # check glcm merge method
        glcm_merge_method = self.params.radiomics.glcm.merge_method
        if glcm_merge_method:
            if glcm_merge_method == 'average':
                glcm_merge_method = '_avg'
            elif glcm_merge_method == 'vol_merge':
                glcm_merge_method = '_comb'

        # check glrlm merge method
        glrlm_merge_method = self.params.radiomics.glrlm.merge_method
        if glrlm_merge_method:
            if glrlm_merge_method == 'average':
                glrlm_merge_method = '_avg'
            elif glrlm_merge_method == 'vol_merge':
                glrlm_merge_method = '_comb'

        # Non-texture Features
        if int_vol_hist_features:
            self.radiomics.image['intVolHist_3D'][self.params.radiomics.ivh_name] = int_vol_hist_features
        if morph_features:
            self.radiomics.image['morph_3D'][self.params.radiomics.scale_name] = morph_features
        if loc_int_features:
            self.radiomics.image['locInt_3D'][self.params.radiomics.scale_name] = loc_int_features
        if stats_features:
            self.radiomics.image['stats_3D'][self.params.radiomics.scale_name] = stats_features
        if int_hist_features:
            self.radiomics.image['intHist_3D'][self.params.radiomics.ih_name] = int_hist_features
        
        # Texture Features
        if glcm_features:
            self.radiomics.image['texture'][
                'glcm_3D' + glcm_merge_method][self.params.radiomics.processing_name] = glcm_features
        if glrlm_features:
            self.radiomics.image['texture'][
                'glrlm_3D' + glrlm_merge_method][self.params.radiomics.processing_name] = glrlm_features
        if glszm_features:
            self.radiomics.image['texture']['glszm_3D'][self.params.radiomics.processing_name] = glszm_features
        if gldzm_features:
            self.radiomics.image['texture']['gldzm_3D'][self.params.radiomics.processing_name] = gldzm_features
        if ngtdm_features:
            self.radiomics.image['texture']['ngtdm_3D'][self.params.radiomics.processing_name] = ngtdm_features
        if ngldm_features:
            self.radiomics.image['texture']['ngldm_3D'][self.params.radiomics.processing_name] = ngldm_features

    def save_radiomics(
                    self, scan_file_name: List, 
                    path_save: Path, roi_type: str, 
                    roi_type_label: str, patient_num: int = None) -> None:
        """
        Saves extracted radiomics features in a JSON file.

        Args:
            scan_file_name(List): List of scan files.
            path_save(Path): Saving path.
            roi_type(str): Type of the ROI.
            roi_type_label(str): Label of the ROI type.
            patient_num(int): Index of scan.
        
        Returns:
            None.
        """
        if path_save.name != f'features({roi_type})':
            if not (path_save / f'features({roi_type})').exists():
                (path_save / f'features({roi_type})').mkdir()
                path_save = Path(path_save / f'features({roi_type})')
            else:
                path_save = Path(path_save) / f'features({roi_type})'
        else:
            path_save = Path(path_save)
        params = {}
        params['roi_type'] = roi_type_label
        params['patientID'] = self.patientID
        params['vox_dim'] = list([
                                self.data.volume.spatialRef.PixelExtentInWorldX, 
                                self.data.volume.spatialRef.PixelExtentInWorldY,
                                self.data.volume.spatialRef.PixelExtentInWorldZ
                                ])
        self.radiomics.update_params(params)
        if type(scan_file_name) is str:
            index_dot = scan_file_name.find('.')
            ext = scan_file_name.find('.npy')
            name_save = scan_file_name[:index_dot] + \
                        '(' + roi_type_label + ')' + \
                        scan_file_name[index_dot : ext]
        elif patient_num is not None:
            index_dot = scan_file_name[patient_num].find('.')
            ext = scan_file_name[patient_num].find('.npy')
            name_save = scan_file_name[patient_num][:index_dot] + \
                        '(' + roi_type_label + ')' + \
                        scan_file_name[patient_num][index_dot : ext]
        else:
            raise ValueError("`patient_num` must be specified or `scan_file_name` must be str")

        with open(path_save / f"{name_save}.json", "w") as fp:   
            dump(self.radiomics.to_json(), fp, indent=4, cls=NumpyEncoder)


    class Params:
        """Organizes all processing, filtering and features extraction parameters"""

        def __init__(self) -> None:
            """
            Organizes all processing, filtering and features extraction
            """
            self.process = self.Process()
            self.filter = self.Filter()
            self.radiomics = self.Radiomics()


        class Process:
            """Organizes all processing parameters."""
            def __init__(self, **kwargs) -> None:
                """
                Constructor of the `Process` class.
                """
                self.algo = kwargs['algo'] if 'algo' in kwargs else None
                self.box_string = kwargs['box_string'] if 'box_string' in kwargs else None
                self.gl_round = kwargs['gl_round'] if 'gl_round' in kwargs else None
                self.gray_levels = kwargs['gray_levels'] if 'gray_levels' in kwargs else None
                self.ih = kwargs['ih'] if 'ih' in kwargs else None
                self.im_range = kwargs['im_range'] if 'im_range' in kwargs else None
                self.im_type = kwargs['im_type'] if 'im_type' in kwargs else None
                self.intensity_type = kwargs['intensity_type'] if 'intensity_type' in kwargs else None
                self.ivh = kwargs['ivh'] if 'ivh' in kwargs else None
                self.n_algo = kwargs['n_algo'] if 'n_algo' in kwargs else None
                self.n_exp = kwargs['n_exp'] if 'n_exp' in kwargs else None
                self.n_gl = kwargs['n_gl'] if 'n_gl' in kwargs else None
                self.n_scale = kwargs['n_scale'] if 'n_scale' in kwargs else None
                self.outliers = kwargs['outliers'] if 'outliers' in kwargs else None
                self.scale_non_text = kwargs['scale_non_text'] if 'scale_non_text' in kwargs else None
                self.scale_text = kwargs['scale_text'] if 'scale_text' in kwargs else None
                self.roi_interp = kwargs['roi_interp'] if 'roi_interp' in kwargs else None
                self.roi_pv = kwargs['roi_pv'] if 'roi_pv' in kwargs else None
                self.user_set_min_value = kwargs['user_set_min_value'] if 'user_set_min_value' in kwargs else None
                self.vol_interp = kwargs['vol_interp'] if 'vol_interp' in kwargs else None

            def init_from_json(self, path_to_json: Union[Path, str]) -> None:
                """
                Updates class attributes from json file.

                Args:
                    path_to_json(Union[Path, str]): Path to the JSON file with processing parameters.
                
                Returns:
                    None.
                """
                __params = load_json(Path(path_to_json))

                self.algo = __params['algo'] if 'algo' in __params else self.algo
                self.box_string = __params['box_string'] if 'box_string' in __params else self.box_string
                self.gl_round = __params['gl_round'] if 'gl_round' in __params else self.gl_round
                self.gray_levels = __params['gray_levels'] if 'gray_levels' in __params else self.gray_levels
                self.ih = __params['ih'] if 'ih' in __params else self.ih
                self.im_range = __params['im_range'] if 'im_range' in __params else self.im_range
                self.im_type = __params['im_type'] if 'im_type' in __params else self.im_type
                self.ivh = __params['ivh'] if 'ivh' in __params else self.ivh
                self.n_algo = __params['n_algo'] if 'n_algo' in __params else self.n_algo
                self.n_exp = __params['n_exp'] if 'n_exp' in __params else self.n_exp
                self.n_gl = __params['n_gl'] if 'n_gl' in __params else self.n_gl
                self.n_scale = __params['n_scale'] if 'n_scale' in __params else self.n_scale
                self.outliers = __params['outliers'] if 'outliers' in __params else self.outliers
                self.scale_non_text = __params['scale_non_text'] if 'scale_non_text' in __params else self.scale_non_text
                self.scale_text = __params['scale_text'] if 'scale_text' in __params else self.scale_text
                self.roi_interp = __params['roi_interp'] if 'roi_interp' in __params else self.roi_interp
                self.roi_pv = __params['roi_pv'] if 'roi_pv' in __params else self.roi_pv
                self.user_set_min_value = __params['user_set_min_value'] if 'user_set_min_value' in __params else self.user_set_min_value
                self.vol_interp = __params['vol_interp'] if 'vol_interp' in __params else self.vol_interp


        class Filter:
            """Organizes all filtering parameters"""
            def __init__(self, filter_type: str = "") -> None:
                """
                Constructor of the Filter class.

                Args:
                    filter_type(str): Type of the filter that will be used (Must be 'mean', 'log', 'laws',
                        'gabor' or 'wavelet').
                
                Returns:
                    None.
                """
                self.filter_type = filter_type
                self.mean = self.Mean()
                self.log = self.Log()
                self.gabor = self.Gabor()
                self.laws = self.Laws()
                self.wavelet = self.Wavelet()
                self.textural = self.Textural()


            class Mean:
                """Organizes the Mean filter parameters"""
                def __init__(
                        self, ndims: int = 0, name_save: str = '', 
                        padding: str = '', size: int = 0, orthogonal_rot: bool = False
                ) -> None:
                    """
                    Constructor of the Mean class.

                    Args:
                        ndims(int): Filter dimension. 
                        name_save(str): Specific name added to final extraction results file. 
                        padding(str): padding mode. 
                        size(int): Filter size. 
                                       
                    Returns:
                        None.
                    """
                    self.name_save = name_save
                    self.ndims = ndims
                    self.orthogonal_rot = orthogonal_rot
                    self.padding = padding
                    self.size = size

                def init_from_json(self, params: Dict) -> None:
                    """
                    Updates class attributes from json file.

                    Args:
                        params(Dict): Dictionary of the Mean filter parameters.
                    
                    Returns:
                        None.
                    """
                    self.name_save = params['name_save']
                    self.ndims = params['ndims']
                    self.padding = params['padding']
                    self.size = params['size']
                    self.orthogonal_rot = params['orthogonal_rot']


            class Log:
                """Organizes the Log filter parameters"""
                def __init__(
                        self, ndims: int = 0, sigma: float = 0.0, 
                        padding: str = '', orthogonal_rot: bool = False, 
                        name_save: str = ''
                ) -> None:
                    """
                    Constructor of the Log class.

                    Args:
                        ndims(int): Filter dimension. 
                        sigma(float): Float of the sigma value.
                        padding(str): padding mode.
                        orthogonal_rot(bool): If True will compute average response over orthogonal planes. 
                        name_save(str): Specific name added to final extraction results file. 
                                       
                    Returns:
                        None.
                    """
                    self.name_save = name_save
                    self.ndims = ndims
                    self.orthogonal_rot = orthogonal_rot
                    self.padding = padding
                    self.sigma = sigma

                def init_from_json(self, params: Dict) -> None:
                    """
                    Updates class attributes from json file.

                    Args:
                        params(Dict): Dictionary of the Log filter parameters.
                    
                    Returns:
                        None.
                    """
                    self.name_save = params['name_save']
                    self.ndims = params['ndims']
                    self.orthogonal_rot = params['orthogonal_rot']
                    self.padding = params['padding']
                    self.sigma = params['sigma']


            class Gabor:
                """Organizes the gabor filter parameters"""
                def __init__(
                        self, sigma: float = 0.0, _lambda: float = 0.0,  
                        gamma: float = 0.0, theta: str = '', rot_invariance: bool = False,
                        orthogonal_rot: bool= False, name_save: str = '',
                        padding: str = ''
                ) -> None:
                    """
                    Constructor of the Gabor class.

                    Args:
                        sigma(float): Float of the sigma value.
                        _lambda(float): Float of the lambda value.
                        gamma(float): Float of the gamma value.
                        theta(str): String of the theta angle value.
                        rot_invariance(bool): If True the filter will be rotation invariant.
                        orthogonal_rot(bool): If True will compute average response over orthogonal planes.
                        name_save(str): Specific name added to final extraction results file.
                        padding(str): padding mode.
                                       
                    Returns:
                        None.
                    """
                    self._lambda = _lambda
                    self.gamma = gamma
                    self.name_save = name_save
                    self.orthogonal_rot = orthogonal_rot
                    self.padding = padding
                    self.rot_invariance = rot_invariance
                    self.sigma = sigma
                    self.theta = theta

                def init_from_json(self, params: Dict) -> None:
                    """
                    Updates class attributes from json file.

                    Args:
                        params(Dict): Dictionary of the gabor filter parameters.
                    
                    Returns:
                        None.
                    """
                    self._lambda = params['lambda']
                    self.gamma = params['gamma']
                    self.name_save = params['name_save']
                    self.orthogonal_rot = params['orthogonal_rot']
                    self.padding = params['padding']
                    self.rot_invariance = params['rot_invariance']
                    self.sigma = params['sigma']
                    if type(params["theta"]) is str:
                        if params["theta"].lower().startswith('pi/'):
                            self.theta = np.pi / int(params["theta"].split('/')[1])
                        elif params["theta"].lower().startswith('-'):
                            if params["theta"].lower().startswith('-pi/'):
                                self.theta = -np.pi / int(params["theta"].split('/')[1])
                            else:
                                nom, denom = params["theta"].replace('-', '').replace('Pi', '').split('/')
                                self.theta = -np.pi*int(nom) / int(denom)
                    else:
                        self.theta = float(params["theta"])


            class Laws:
                """Organizes the laws filter parameters"""
                def __init__(
                        self, config: List = [], energy_distance: int = 0, 
                        energy_image: bool = False, rot_invariance: bool = False, 
                        orthogonal_rot: bool = False, name_save: str = '', padding: str = ''
                ) -> None:
                    """
                    Constructor of the Laws class.

                    Args:
                        config(List): Configuration of the Laws filter, for ex: ['E5', 'L5', 'E5'].
                        energy_distance(int): Chebyshev distance.
                        energy_image(bool): If True will compute the Laws texture energy image.
                        rot_invariance(bool): If True the filter will be rotation invariant.
                        orthogonal_rot(bool): If True will compute average response over orthogonal planes.
                        name_save(str): Specific name added to final extraction results file.
                        padding(str): padding mode.
                                       
                    Returns:
                        None.
                    """
                    self.config = config
                    self.energy_distance = energy_distance
                    self.energy_image = energy_image
                    self.name_save = name_save
                    self.orthogonal_rot = orthogonal_rot
                    self.padding = padding
                    self.rot_invariance = rot_invariance

                def init_from_json(self, params: Dict) -> None:
                    """
                    Updates class attributes from json file.

                    Args:
                        params(Dict): Dictionary of the laws filter parameters.
                    
                    Returns:
                        None.
                    """
                    self.config = params['config']
                    self.energy_distance = params['energy_distance']
                    self.energy_image = params['energy_image']
                    self.name_save = params['name_save']
                    self.orthogonal_rot = params['orthogonal_rot']
                    self.padding = params['padding']
                    self.rot_invariance = params['rot_invariance']


            class Wavelet:
                """Organizes the Wavelet filter parameters"""
                def __init__(
                        self, ndims: int = 0, name_save: str = '', 
                        basis_function: str = '', subband: str = '', level: int = 0, 
                        rot_invariance: bool = False, padding: str = ''
                ) -> None:
                    """
                    Constructor of the Wavelet class.

                    Args:
                        ndims(int): Dimension of the filter.
                        name_save(str): Specific name added to final extraction results file.
                        basis_function(str): Wavelet basis function.
                        subband(str): Wavelet subband.
                        level(int): Decomposition level.
                        rot_invariance(bool): If True the filter will be rotation invariant.
                        padding(str): padding mode.
                                       
                    Returns:
                        None.
                    """
                    self.basis_function = basis_function
                    self.level = level
                    self.ndims = ndims
                    self.name_save = name_save
                    self.padding = padding
                    self.rot_invariance = rot_invariance
                    self.subband = subband

                def init_from_json(self, params: Dict) -> None:
                    """
                    Updates class attributes from json file.

                    Args:
                        params(Dict): Dictionary of the wavelet filter parameters.
                    
                    Returns:
                        None.
                    """
                    self.basis_function = params['basis_function']
                    self.level = params['level']
                    self.ndims = params['ndims']
                    self.name_save = params['name_save']
                    self.padding = params['padding']
                    self.rot_invariance = params['rot_invariance']
                    self.subband = params['subband']
            

            class Textural:
                """Organizes the Textural filters parameters"""
                def __init__(
                        self,
                        family: str = '',
                        size: int = 0,
                        discretization: dict = {},
                        local: bool = False,
                        name_save: str = ''
                ) -> None:
                    """
                    Constructor of the Textural class.

                    Args:
                        family (str, optional): The family of the textural filter.
                        size (int, optional): The filter size.
                        discretization (dict, optional): The discretization parameters.
                        local (bool, optional): If true, the discretization will be computed locally, else globally.
                        name_save (str, optional): Specific name added to final extraction results file.
                                       
                    Returns:
                        None.
                    """
                    self.family = family
                    self.size = size
                    self.discretization = discretization
                    self.local = local
                    self.name_save = name_save

                def init_from_json(self, params: Dict) -> None:
                    """
                    Updates class attributes from json file.

                    Args:
                        params(Dict): Dictionary of the wavelet filter parameters.
                    
                    Returns:
                        None.
                    """
                    self.family = params['family']
                    self.size = params['size']
                    self.discretization = params['discretization']
                    self.local = params['local']
                    self.name_save = params['name_save']


        class Radiomics:
            """Organizes the radiomics extraction parameters"""
            def __init__(self, **kwargs) -> None:
                """
                Constructor of the Radiomics class.
                """
                self.ih_name = kwargs['ih_name'] if 'ih_name' in kwargs else None
                self.ivh_name = kwargs['ivh_name'] if 'ivh_name' in kwargs else None
                self.glcm = self.GLCM()
                self.glrlm = self.GLRLM()
                self.ngtdm = self.NGTDM()
                self.name_text_types = kwargs['name_text_types'] if 'name_text_types' in kwargs else None
                self.processing_name = kwargs['processing_name'] if 'processing_name' in kwargs else None
                self.scale_name = kwargs['scale_name'] if 'scale_name' in kwargs else None


            class GLCM:
                """Organizes the GLCM features extraction parameters"""
                def __init__(
                        self, 
                        dist_correction: Union[bool, str] = False,
                        merge_method: str = "vol_merge"
                ) -> None:
                    """
                    Constructor of the GLCM class

                    Args:
                        dist_correction(Union[bool, str]): norm for distance weighting, must be 
                            "manhattan", "euclidean" or "chebyshev". If True the norm for distance weighting 
                            is gonna be "euclidean".
                        merge_method(str): merging method which determines how features are
                            calculated. Must be "average", "slice_merge", "dir_merge" and "vol_merge".
                                       
                    Returns:
                        None.
                    """
                    self.dist_correction = dist_correction
                    self.merge_method = merge_method


            class GLRLM:
                """Organizes the GLRLM features extraction parameters"""
                def __init__(
                        self, 
                        dist_correction: Union[bool, str] = False,
                        merge_method: str = "vol_merge"
                ) -> None:
                    """
                    Constructor of the GLRLM class

                    Args:
                        dist_correction(Union[bool, str]): If True the norm for distance weighting is gonna be "euclidean".
                        merge_method(str): merging method which determines how features are
                            calculated. Must be "average", "slice_merge", "dir_merge" and "vol_merge".

                    Returns:
                        None.
                    """
                    self.dist_correction = dist_correction
                    self.merge_method = merge_method


            class NGTDM:
                """Organizes the NGTDM features extraction parameters"""
                def __init__(
                        self, 
                        dist_correction: Union[bool, str] = None
                ) -> None:
                    """
                    Constructor of the NGTDM class

                    Args:
                        dist_correction(Union[bool, str]): If True the norm for distance weighting is gonna be "euclidean".

                    Returns:
                        None.
                    """
                    self.dist_correction = dist_correction


    class Radiomics:
        """Organizes all the extracted features.
        """
        def __init__(self, image: Dict = None, params: Dict = None) -> None:
            """Constructor of the Radiomics class
            Args:
                image(Dict): Dict of the extracted features.
                params(Dict): Dict of the parameters used in features extraction (roi type, voxels diemension...)
            
            Returns:
                None
            """
            self.image = image if image else {}
            self.params = params if params else {}

        def update_params(self, params: Dict) -> None:
            """Updates `params` attribute from a given Dict
                Args:
                    params(Dict): Dict of the parameters used in features extraction (roi type, voxels diemension...)
                
                Returns:
                    None
            """
            self.params['roi_type'] = params['roi_type']
            self.params['patientID'] = params['patientID']
            self.params['vox_dim'] = params['vox_dim']

        def to_json(self) -> Dict:
            """Summarizes the class attributes in a Dict
                Args:
                    None
                
                Returns:
                    Dict: Dictionay of radiomics structure (extracted features and extraction params)
            """
            radiomics = {
                'image': self.image,
                'params': self.params
            }
            return radiomics


    class data:
        """Organizes all imaging data (volume and ROI). 

        Attributes:
            volume (object): Instance of MEDscan.data.volume inner class.
            ROI (object): Instance of MEDscan.data.ROI inner class.
            orientation (str): Imaging data orientation (axial, sagittal or coronal).
            patient_position (str): Patient position specifies the position of the 
                patient relative to the imaging equipment space (HFS, HFP...).

        """
        def __init__(self, orientation: str=None, patient_position: str=None) -> None:
            """Constructor of the scan class

            Args:
                orientation (str, optional): Imaging data orientation (axial, sagittal or coronal).
                patient_position (str, optional): Patient position specifies the position of the 
                    patient relative to the imaging equipment space (HFS, HFP...).
            
            Returns:
                None.
            """
            self.volume = self.volume() 
            self.volume_process = self.volume_process()
            self.ROI = self.ROI()
            self.orientation = orientation
            self.patient_position = patient_position

        def set_patient_position(self, patient_position):
            self.patient_position = patient_position

        def set_orientation(self, orientation):
            self.orientation = orientation
        
        def set_volume(self, volume):
            self.volume = volume
        
        def set_ROI(self, *args):
            self.ROI = self.ROI(args)

        def get_roi_from_indexes(self, key: int) -> np.ndarray:
            """
            Extracts ROI data using the saved indexes (Indexes of non-null values).

            Args:
                key (int): Key of ROI indexes list (A volume can have multiple ROIs).

            Returns:
                ndarray: n-dimensional array of ROI data.
            
            """
            roi_volume = np.zeros_like(self.volume.array).flatten()
            roi_volume[self.ROI.get_indexes(key)] = 1
            return roi_volume.reshape(self.volume.array.shape)

        def get_indexes_by_roi_name(self, roi_name : str) -> np.ndarray:
            """
            Extract ROI data using the ROI name.

            Args:
                roi_name (str): String of the ROI name (A volume can have multiple ROIs).

            Returns:
                ndarray: n-dimensional array of the ROI data.
            
            """
            roi_name_key = list(self.ROI.roi_names.values()).index(roi_name)
            roi_volume = np.zeros_like(self.volume.array).flatten()
            roi_volume[self.ROI.get_indexes(roi_name_key)] = 1
            return roi_volume.reshape(self.volume.array.shape)

        def display(self, _slice: int = None, roi: Union[str, int] = 0) -> None:
            """Displays slices from imaging data with the ROI contour in XY-Plane.

            Args:
                _slice (int, optional): Index of the slice you want to plot.
                roi (Union[str, int], optional): ROI name or index. If not specified will use the first ROI.

            Returns:
                None.
            
            """
            # extract slices containing ROI
            size_m = self.volume.array.shape
            i = np.arange(0, size_m[0])
            j = np.arange(0, size_m[1])
            k = np.arange(0, size_m[2])
            ind_mask = np.nonzero(self.get_roi_from_indexes(roi))
            J, I, K = np.meshgrid(i, j, k, indexing='ij')
            I = I[ind_mask]
            J = J[ind_mask]
            K = K[ind_mask]
            slices = np.unique(K)

            vol_data = self.volume.array.swapaxes(0, 1)[:, :, slices]
            roi_data = self.get_roi_from_indexes(roi).swapaxes(0, 1)[:, :, slices]        
            
            rows = int(np.round(np.sqrt(len(slices))))
            columns = int(np.ceil(len(slices) / rows))
            
            plt.set_cmap(plt.gray())
            
            # plot only one slice
            if _slice:
                fig, ax =  plt.subplots(1, 1, figsize=(10, 5))
                ax.axis('off')
                ax.set_title(_slice)
                ax.imshow(vol_data[:, :, _slice])
                im = Image.fromarray((roi_data[:, :, _slice]))
                ax.contour(im, colors='red', linewidths=0.4, alpha=0.45)
                lps_ax = fig.add_subplot(1, columns, 1)
            
            # plot multiple slices containing an ROI.
            else:
                fig, axs =  plt.subplots(rows, columns+1, figsize=(20, 10))
                s = 0
                for i in range(0,rows):
                    for j in range(0,columns):
                        axs[i,j].axis('off')
                        if s < len(slices):
                            axs[i,j].set_title(str(s))
                            axs[i,j].imshow(vol_data[:, :, s])
                            im = Image.fromarray((roi_data[:, :, s]))
                            axs[i,j].contour(im, colors='red', linewidths=0.4, alpha=0.45)
                        s += 1
                    axs[i,columns].axis('off')
                lps_ax = fig.add_subplot(1, columns+1, axs.shape[1])

            fig.suptitle('XY-Plane')
            fig.tight_layout()
            
            # add the coordinates system
            lps_ax.axis([-1.5, 1.5, -1.5, 1.5])
            lps_ax.set_title("Coordinates system")
            
            lps_ax.quiver([-0.5], [0], [1.5], [0], scale_units='xy', angles='xy', scale=1.0, color='green')
            lps_ax.quiver([-0.5], [0], [0], [-1.5], scale_units='xy', angles='xy', scale=3, color='blue')
            lps_ax.quiver([-0.5], [0], [1.5], [1.5], scale_units='xy', angles='xy', scale=3, color='red')
            lps_ax.text(1.0, 0, "L")
            lps_ax.text(-0.3, -0.5, "P")
            lps_ax.text(0.3, 0.4, "S")

            lps_ax.set_xticks([])
            lps_ax.set_yticks([])

            plt.show()

        def display_process(self, _slice: int = None, roi: Union[str, int] = 0) -> None:
            """Displays slices from imaging data with the ROI contour in XY-Plane.

            Args:
                _slice (int, optional): Index of the slice you want to plot.
                roi (Union[str, int], optional): ROI name or index. If not specified will use the first ROI.

            Returns:
                None.
            
            """
            # extract slices containing ROI
            size_m = self.volume_process.array.shape
            i = np.arange(0, size_m[0])
            j = np.arange(0, size_m[1])
            k = np.arange(0, size_m[2])
            ind_mask = np.nonzero(self.get_roi_from_indexes(roi))
            J, I, K = np.meshgrid(j, i, k, indexing='ij')
            I = I[ind_mask]
            J = J[ind_mask]
            K = K[ind_mask]
            slices = np.unique(K)

            vol_data = self.volume_process.array.swapaxes(0, 1)[:, :, slices]
            roi_data = self.get_roi_from_indexes(roi).swapaxes(0, 1)[:, :, slices]        
            
            rows = int(np.round(np.sqrt(len(slices))))
            columns = int(np.ceil(len(slices) / rows))
            
            plt.set_cmap(plt.gray())
            
            # plot only one slice
            if _slice:
                fig, ax =  plt.subplots(1, 1, figsize=(10, 5))
                ax.axis('off')
                ax.set_title(_slice)
                ax.imshow(vol_data[:, :, _slice])
                im = Image.fromarray((roi_data[:, :, _slice]))
                ax.contour(im, colors='red', linewidths=0.4, alpha=0.45)
                lps_ax = fig.add_subplot(1, columns, 1)
            
            # plot multiple slices containing an ROI.
            else:
                fig, axs =  plt.subplots(rows, columns+1, figsize=(20, 10))
                s = 0
                for i in range(0,rows):
                    for j in range(0,columns):
                        axs[i,j].axis('off')
                        if s < len(slices):
                            axs[i,j].set_title(str(s))
                            axs[i,j].imshow(vol_data[:, :, s])
                            im = Image.fromarray((roi_data[:, :, s]))
                            axs[i,j].contour(im, colors='red', linewidths=0.4, alpha=0.45)
                        s += 1
                    axs[i,columns].axis('off')
                lps_ax = fig.add_subplot(1, columns+1, axs.shape[1])

            fig.suptitle('XY-Plane')
            fig.tight_layout()
            
            # add the coordinates system
            lps_ax.axis([-1.5, 1.5, -1.5, 1.5])
            lps_ax.set_title("Coordinates system")
            
            lps_ax.quiver([-0.5], [0], [1.5], [0], scale_units='xy', angles='xy', scale=1.0, color='green')
            lps_ax.quiver([-0.5], [0], [0], [-1.5], scale_units='xy', angles='xy', scale=3, color='blue')
            lps_ax.quiver([-0.5], [0], [1.5], [1.5], scale_units='xy', angles='xy', scale=3, color='red')
            lps_ax.text(1.0, 0, "L")
            lps_ax.text(-0.3, -0.5, "P")
            lps_ax.text(0.3, 0.4, "S")

            lps_ax.set_xticks([])
            lps_ax.set_yticks([])

            plt.show()


        class volume:
            """Organizes all volume data and information related to imaging volume. 

            Attributes:
                spatialRef (imref3d): Imaging data orientation (axial, sagittal or coronal).
                scan_rot (ndarray): Array of the rotation applied to the XYZ points of the ROI.
                array (ndarray): n-dimensional of the imaging data.

            """
            def __init__(self, spatialRef: imref3d=None, scan_rot: str=None, array: np.ndarray=None) -> None:
                """Organizes all volume data and information. 

                Args:
                    spatialRef (imref3d, optional): Imaging data orientation (axial, sagittal or coronal).
                    scan_rot (ndarray, optional): Array of the rotation applied to the XYZ points of the ROI.
                    array (ndarray, optional): n-dimensional of the imaging data.

                """
                self.spatialRef = spatialRef
                self.scan_rot = scan_rot
                self.array = array

            def update_spatialRef(self, spatialRef_value):
                self.spatialRef = spatialRef_value
            
            def update_scan_rot(self, scan_rot_value):
                self.scan_rot = scan_rot_value
            
            def update_transScanToModel(self, transScanToModel_value):
                self.transScanToModel = transScanToModel_value
            
            def update_array(self, array):
                self.array = array

            def convert_to_LPS(self):
                """Convert Imaging data to LPS (Left-Posterior-Superior) coordinates system.
                <https://www.slicer.org/wiki/Coordinate_systems>.

                Returns:
                    None.

                """
                # flip x
                self.array = np.flip(self.array, 0)
                # flip y
                self.array = np.flip(self.array, 1)
            
            def spatialRef_from_nifti(self, nifti_image_path: Union[Path, str]) -> None:
                """Computes the imref3d spatialRef using a NIFTI file and
                updates the `spatialRef` attribute.

                Args:
                    nifti_image_path (str): String of the NIFTI file path.

                Returns:
                    None.

                """
                # Loading the nifti file:
                nifti_image_path = Path(nifti_image_path)
                nifti = nib.load(nifti_image_path)
                nifti_data = self.array

                # spatialRef Creation
                pixelX = nifti.affine[0, 0]
                pixelY = nifti.affine[1, 1]
                sliceS = nifti.affine[2, 2]
                min_grid = nifti.affine[:3, 3]
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

                # update spatialRef
                self.update_spatialRef(spatialRef)

            def convert_spatialRef(self):
                """converts the `spatialRef` attribute from RAS to LPS coordinates system.
                <https://www.slicer.org/wiki/Coordinate_systems>.

                Args:
                    None.

                Returns:
                    None.

                """
                # swap x and y data
                temp = self.spatialRef.ImageExtentInWorldX
                self.spatialRef.ImageExtentInWorldX = self.spatialRef.ImageExtentInWorldY
                self.spatialRef.ImageExtentInWorldY = temp

                temp = self.spatialRef.PixelExtentInWorldX
                self.spatialRef.PixelExtentInWorldX = self.spatialRef.PixelExtentInWorldY
                self.spatialRef.PixelExtentInWorldY = temp

                temp = self.spatialRef.XIntrinsicLimits
                self.spatialRef.XIntrinsicLimits = self.spatialRef.YIntrinsicLimits
                self.spatialRef.YIntrinsicLimits = temp

                temp = self.spatialRef.XWorldLimits
                self.spatialRef.XWorldLimits = self.spatialRef.YWorldLimits
                self.spatialRef.YWorldLimits = temp
                del temp

        class volume_process:
            """Organizes all volume data and information. 

            Attributes:
                spatialRef (imref3d): Imaging data orientation (axial, sagittal or coronal).
                scan_rot (ndarray): Array of the rotation applied to the XYZ points of the ROI.
                data (ndarray): n-dimensional of the imaging data.

            """
            def __init__(self, spatialRef: imref3d = None, 
                        scan_rot: List = None, array: np.ndarray = None,
                        user_string: str = "") -> None:
                """Organizes all volume data and information. 

                Args:
                    spatialRef (imref3d, optional): Imaging data orientation (axial, sagittal or coronal).
                    scan_rot (ndarray, optional): Array of the rotation applied to the XYZ points of the ROI.
                    array (ndarray, optional): n-dimensional of the imaging data.
                    user_string(str, optional): string explaining the processed data in the class.
                
                Returns:
                    None.

                """
                self.array = array
                self.scan_rot = scan_rot
                self.spatialRef = spatialRef
                self.user_string = user_string

            def update_processed_data(self, array: np.ndarray, user_string: str = "") -> None:
                if user_string:
                    self.user_string = user_string
                self.array = array

            def save(self, name_save: str, path_save: Union[Path, str])-> None:
                """Saves the processed data locally.

                Args:
                    name_save(str): Saving name of the processed data.
                    path_save(Union[Path, str]): Path to where save the processed data.
                
                Returns:
                    None.
                """
                path_save = Path(path_save)
                if not name_save:
                    name_save = self.user_string

                if not name_save.endswith('.npy'):
                    name_save += '.npy'

                with open(path_save / name_save, 'wb') as f:
                    np.save(f, self.array)

            def load(
                    self, 
                    file_name: str, 
                    loading_path: Union[Path, str], 
                    update: bool=True
                ) -> Union[None, np.ndarray]:
                """Saves the processed data locally.

                Args:
                    file_name(str): Name file of the processed data to load.
                    loading_path(Union[Path, str]): Path to the processed data to load.
                    update(bool, optional): If True, updates the class attrtibutes with loaded data.

                Returns:
                    None.
                """
                loading_path = Path(loading_path)

                if not file_name.endswith('.npy'):
                    file_name += '.npy'

                with open(loading_path / file_name, 'rb') as f:
                    if update:
                        self.update_processed_data(np.load(f, allow_pickle=True))
                    else:
                        return np.load(f, allow_pickle=True)


        class ROI:
            """Organizes all ROI data and information. 

            Attributes:
                indexes (Dict): Dict of the ROI indexes for each ROI name.
                roi_names (Dict): Dict of the ROI names.
                nameSet (Dict): Dict of the User-defined name for Structure Set for each ROI name.
                nameSetInfo (Dict): Dict of the names of the structure sets that define the areas of 
                    significance. Either 'StructureSetName', 'StructureSetDescription', 'SeriesDescription' 
                    or 'SeriesInstanceUID'.
            
            """
            def __init__(self, indexes: Dict=None, roi_names: Dict=None) -> None:
                """Constructor of the ROI class.

                Args:
                    indexes (Dict, optional): Dict of the ROI indexes for each ROI name.
                    roi_names (Dict, optional): Dict of the ROI names.
                
                Returns:
                    None.
                """
                self.indexes = indexes if indexes else {}
                self.roi_names = roi_names if roi_names else {}
                self.nameSet = roi_names if roi_names else {}
                self.nameSetInfo = roi_names if roi_names else {}

            def get_indexes(self, key):
                if not self.indexes or key is None:
                    return {}
                else:
                    return self.indexes[str(key)]

            def get_roi_name(self, key):
                if not self.roi_names or key is None:
                    return {}
                else:
                    return self.roi_names[str(key)]

            def get_name_set(self, key):
                if not self.nameSet or key is None:
                    return {}
                else:
                    return self.nameSet[str(key)]

            def get_name_set_info(self, key):
                if not self.nameSetInfo or key is None:
                    return {}
                else:
                    return self.nameSetInfo[str(key)]

            def update_indexes(self, key, indexes):
                try: 
                    self.indexes[str(key)] = indexes
                except:
                    Warning.warn("Wrong key given in update_indexes()")

            def update_roi_name(self, key, roi_name):
                try:
                    self.roi_names[str(key)] = roi_name
                except:
                    Warning.warn("Wrong key given in update_roi_name()")

            def update_name_set(self, key, name_set):
                try:
                    self.nameSet[str(key)] = name_set
                except:
                    Warning.warn("Wrong key given in update_name_set()")

            def update_name_set_info(self, key, nameSetInfo):
                try:
                    self.nameSetInfo[str(key)] = nameSetInfo
                except:
                    Warning.warn("Wrong key given in update_name_set_info()")
            
            def convert_to_LPS(self, data: np.ndarray) -> np.ndarray:
                """Converts the given volume to LPS coordinates system. For 
                more details please refer here : https://www.slicer.org/wiki/Coordinate_systems
                Args:
                    data(ndarray) : Volume data in RAS to convert to to LPS

                Returns:
                    ndarray: n-dimensional of `data` in LPS.
                """
                # flip x
                data = np.flip(data, 0)
                # flip y
                data = np.flip(data, 1)

                return data

            def get_roi_from_path(self, roi_path: Union[Path, str], id: str):
                """Extracts all ROI data from the given path for the given
                patient ID and updates all class attributes with the new extracted data.

                Args:
                    roi_path(Union[Path, str]): Path where the ROI data is stored.
                    id(str): ID containing patient ID and the modality type, to identify the right file.
                
                Returns:
                    None.
                """
                self.indexes = {}
                self.roi_names = {}
                self.nameSet = {}
                self.nameSetInfo = {}
                roi_index = 0
                list_of_patients = os.listdir(roi_path)

                for file in list_of_patients:
                    # Load the patient's ROI nifti files :
                    if file.startswith(id) and file.endswith('nii.gz') and 'ROI' in file.split("."):
                        roi = nib.load(roi_path + "/" + file)
                        roi_data = self.convert_to_LPS(data=roi.get_fdata())
                        roi_name = file[file.find("(")+1 : file.find(")")]
                        name_set = file[file.find("_")+2 : file.find("(")]
                        self.update_indexes(key=roi_index, indexes=np.nonzero(roi_data.flatten()))
                        self.update_name_set(key=roi_index, name_set=name_set)
                        self.update_roi_name(key=roi_index, roi_name=roi_name)
                        roi_index += 1
