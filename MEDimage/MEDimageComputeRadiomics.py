import logging
import os
from json import dump

from numpyencoder import NumpyEncoder

from MEDimage.MEDimageProcessing import MEDimageProcessing

_logger = logging.getLogger(__name__)


class MEDimageComputeRadiomics(MEDimageProcessing):

    def __init__(self, MEDimg=None, log_file=None):
        super().__init__(MEDimg=MEDimg, log_file=log_file)
        self.scaleName = ''
        self.processing_name = ''
        self.name_text_types = []
    
    def init_filters_params(self, imParamScan, **kwargs):
        im_filter = True
        ih_filter = imParamScan['filter']['discretisation']['IH']
        ivh_filter = imParamScan['filter']['discretisation']['IVH']
        algo_filter = imParamScan['filter']['discretisation']['texture']['type']
        gray_levels_filter = imParamScan['filter']['discretisation']['texture']['val']
        intensity_filter = imParamScan['filter']['intensity']
        filters_type = imParamScan['filter']['ToCompute']

        filter_params = {}
        filter_params['im_filter'] = im_filter
        filter_params['ih_filter'] = ih_filter
        filter_params['ivh_filter'] = ivh_filter
        filter_params['algo_filter'] = algo_filter
        filter_params['gray_levels_filter'] = gray_levels_filter
        filter_params['intensity_filter'] = intensity_filter
        filter_params['filters_type'] = filters_type

        for key,value in kwargs.items():
            try:
                filter_params[key] = value
            except:
                pass

        return im_filter, ih_filter, ivh_filter, algo_filter, gray_levels_filter, intensity_filter, filters_type

    def init_nft_calculation(self, vol_obj):
        """
        -------------------------------------------------------------------------
        DESCRIPTION:
        Initializes all the computation for NON-TEXTURE FEATURES parameters 
        as well as the results dict.
        -------------------------------------------------------------------------
        """
        try:
            Params = self.Params
            if sum(Params['scaleNonText']) == 0:  # In case the user chose to not interpolate
                        Params['scaleNonText'] = [vol_obj.spatial_ref.PixelExtentInWorldX,
                                        vol_obj.spatial_ref.PixelExtentInWorldY,
                                        vol_obj.spatial_ref.PixelExtentInWorldZ]
            else:
                if len(Params['scaleNonText']) == 2:
                    # In case not interpolation is performed in
                    # the slice direction (e.g. 2D case)
                    Params['scaleNonText'] = Params['scaleNonText'] + \
                        [vol_obj.spatial_ref.PixelExtentInWorldZ]

            # Scale name
            # Always isotropic resampling, so the first entry is ok.
            self.Params['scaleName'] = 'scale'+(str(Params['scaleNonText'][0])).replace('.', 'dot')

            # IH name
            ih_val_name = 'bin' + (str(Params['IH']['val'])).replace('.', 'dot')

            # The minimum value defines the computation.
            if Params['IH']['type'].find('FBS')>=0:
                if type(Params['user_set_min_val']) is list and Params['user_set_min_val']:
                    min_val_name = '_min' + \
                        ((str(Params['user_set_min_val'])).replace('.', 'dot')).replace('-', 'M')
                else:
                    # Otherwise, minimum value of ROI will be used (not recommended),
                    # so no need to report it.
                    min_val_name = ''
            else:
                min_val_name = ''

            self.Params['IHname'] = self.Params['scaleName'] + '_algo' + Params['IH']['type'] + '_' + ih_val_name + min_val_name

            # IVH name
            if not Params['IVH']:  # CT case
                ivh_algo_name = 'algoNone'
                ivh_val_name = 'bin1'
                if Params['im_range']:  # The im_range defines the computation.
                    min_val_name = ((str(Params['im_range'][0])).replace(
                        '.', 'dot')).replace('-', 'M')
                    max_val_name = ((str(Params['im_range'][1])).replace(
                        '.', 'dot')).replace('-', 'M')
                    range_name = '_min' + min_val_name + '_max' + max_val_name
                else:
                    range_name = ''
            else:
                ivh_algo_name = 'algo' + Params['IVH']['type']
                ivh_val_name = 'bin' + (str(Params['IVH']['val'])).replace('.', 'dot')
                # The im_range defines the computation.
                if 'type' in Params['IVH'] and Params['IVH']['type'].find('FBS') >=0:
                    if Params['im_range']:
                        min_val_name = ((str(Params['im_range'][0])).replace(
                            '.', 'dot')).replace('-', 'M')
                        max_val_name = ((str(Params['im_range'][1])).replace(
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

            self.Params['IVHname'] = self.Params['scaleName'] + '_' + ivh_algo_name + '_' + ivh_val_name + range_name
            self.Params = Params

            #Now initialize the attribute that will hold the computation results
            self.results = { 'morph_3D': { self.Params['scaleName'] : {} },
                    'locInt_3D': { self.Params['scaleName'] : {} },
                    'stats_3D': { self.Params['scaleName'] : {} },
                    'intHist_3D': { self.Params['IHname'] : {} },
                    'intVolHist_3D': { self.Params['IVHname'] : {} } }

        except Exception as e:
            message = "\n PROBLEM WITH PRE-PROCESSING OF FEATURES IN init_nft_calculation(): " \
                    "\n {}".format(e)
            _logger.error(message)
            print(message)

            self.Params['radiomics']['image'].update(
                {('scale'+(str(self.Params['scaleNonText'][0])).replace('.', 'dot')): 'ERROR_PROCESSING'})

    def init_tf_calculation(self, Algo:int, Gl:int, Scale:int):

        self.name_text_types = ['glcm_3Dmrg', 'glrlm_3Dmrg',
                        'glszm_3D', 'gldzm_3D', 'ngtdm_3D', 'ngldm_3D']
        n_text_types = len(self.name_text_types)

        if 'texture' in self.Params['radiomics']['image']:
            #Params Dict is already initialized
            pass
        else:
            self.Params['radiomics']['image'].update({'texture': {}})

            for t in range(n_text_types):
                self.Params['radiomics']['image']['texture'].update({self.name_text_types[t]: {}})
        # Scale name
        # Always isotropic resampling, so the first entry is ok.
        scaleName = 'scale'+(str(self.Params['scaleText'][Scale][0])).replace('.', 'dot')

        # Discretisation name
        gray_levels_name = (str(self.Params['grayLevels'][Algo][Gl])).replace('.', 'dot')

        if 'FBS' in self.Params['algo'][Algo]:  # The minimum value defines the computation.
            if type(self.Params['user_set_min_val']) is list and self.Params['user_set_min_val']:
                min_val_name = '_min' + \
                    ((str(self.Params['user_set_min_val'])).replace('.', 'dot')).replace('-', 'M')
            else:
                # Otherwise, minimum value of ROI will be used (not recommended),
                # so no need to report it.
                min_val_name = ''
        else:
            min_val_name = ''

        if 'equal'in self.Params['algo'][Algo]:
            # The number of gray-levels used for equalization is currently
            # hard-coded to 64 in equalization.m
            discretisation_name = 'algo' + self.Params['algo'][Algo] + '256_bin' + gray_levels_name + min_val_name
        else:
            discretisation_name = 'algo' + self.Params['algo'][Algo] + '_bin' + gray_levels_name + min_val_name

        # Processing full name
        processing_name = scaleName + '_' + discretisation_name
        
        self.results.update( {'glcm_3Dmrg': {processing_name: {}},
                                'glrlm_3Dmrg': {processing_name: {}},
                                'glszm_3D': {processing_name: {}},
                                'gldzm_3D': {processing_name: {}},
                                'ngtdm_3D': {processing_name: {}},
                                'ngldm_3D': {processing_name: {}}} )

        setattr(self, 'scaleName', scaleName)
        setattr(self, 'processing_name', processing_name)

    def update_radiomics(self, int_vol_hist, morph, local_intensity, stats, int_hist,
                        glcm, glrlm, glszm, gldzm, ngtdm, ngldm):
        texture_features = ['intVolHist_3D','morph_3D','locInt_3D','stats_3D','intHist_3D']
        #Non-Texture Features
        self.results['intVolHist_3D'][self.Params['IVHname']] = int_vol_hist
        self.results['morph_3D'][self.Params['scaleName']] = morph
        self.results['locInt_3D'][self.Params['scaleName']] = local_intensity
        self.results['stats_3D'][self.Params['scaleName']] = stats
        self.results['intHist_3D'][self.Params['IHname']] = int_hist

        #Done with non-texture features, update Params with the new results
        for TF in texture_features:
            self.Params['radiomics']['image'][TF] = self.results[TF]
        
        #Texture Features
        self.results['glcm_3Dmrg'][self.processing_name] = glcm
        self.results['glrlm_3Dmrg'][self.processing_name] = glrlm
        self.results['glszm_3D'][self.processing_name] = glszm
        self.results['gldzm_3D'][self.processing_name] = gldzm
        self.results['ngtdm_3D'][self.processing_name] = ngtdm
        self.results['ngldm_3D'][self.processing_name] = ngldm
        #update the radiomics parameters with all the results of the calculated texture features
        for t in range(len(self.name_text_types)):
            self.Params['radiomics']['image']['texture'][self.name_text_types[t]].update(
                self.results[self.name_text_types[t]])

    def save_radiomics_structure(self, scan_file_name, path_save, type_of_roi, label_of_roi_type, patient_num):
        """
        Saves extracted radiomics features in a JSON file.
        """
        self.Params['radiomics']['imParam']['roi_type'] = type_of_roi
        self.Params['radiomics']['imParam']['patient_id'] = self.patient_id
        self.Params['radiomics']['imParam']['vox_dim'] = list([self.scan.volume.spatial_ref.PixelExtentInWorldX, 
                                                            self.scan.volume.spatial_ref.PixelExtentInWorldY,
                                                            self.scan.volume.spatial_ref.PixelExtentInWorldZ])

        ind_dot = scan_file_name[patient_num].find('.')
        ext = scan_file_name[patient_num].find('.npy')
        name_save = scan_file_name[patient_num][:ind_dot] + \
            '(' + label_of_roi_type + ')' + scan_file_name[patient_num][ind_dot:ext]

        os.chdir(path_save)

        # IMPORTANT: HERE, WE COULD ADD SOME CODE TO APPEND A NEW "radiomics"
        # STRUCTURE TO AN EXISTING ONE WITH THE SAME NAME IN "path_save"
        with open(f"{name_save}.json", "w") as fp:   
            dump(self.Params['radiomics'], fp, indent=4, cls=NumpyEncoder)
