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
        self.processingName = ''
        self.nameTextTypes = []
    
    def init_filters_params(self, imParamScan, **kwargs):
        im_filter = True
        IHfilter = imParamScan['filter']['discretisation']['IH']
        IVHfilter = imParamScan['filter']['discretisation']['IVH']
        algoFilter = imParamScan['filter']['discretisation']['texture']['type']
        grayLevelsFilter = imParamScan['filter']['discretisation']['texture']['val']
        intensityFilter = imParamScan['filter']['intensity']
        filtersType = imParamScan['filter']['ToCompute']

        Filter_Params = {}
        Filter_Params['im_filter'] = im_filter
        Filter_Params['IHfilter'] = IHfilter
        Filter_Params['IVHfilter'] = IVHfilter
        Filter_Params['algoFilter'] = algoFilter
        Filter_Params['grayLevelsFilter'] = grayLevelsFilter
        Filter_Params['intensityFilter'] = intensityFilter
        Filter_Params['filtersType'] = filtersType

        for key,value in kwargs.items():
            try:
                Filter_Params[key] = value
            except:
                pass

        return im_filter, IHfilter, IVHfilter, algoFilter, grayLevelsFilter, intensityFilter, filtersType

    def init_NTF_Calculation(self, volObj):
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
                        Params['scaleNonText'] = [volObj.spatialRef.PixelExtentInWorldX,
                                        volObj.spatialRef.PixelExtentInWorldY,
                                        volObj.spatialRef.PixelExtentInWorldZ]
            else:
                if len(Params['scaleNonText']) == 2:
                    # In case not interpolation is performed in
                    # the slice direction (e.g. 2D case)
                    Params['scaleNonText'] = Params['scaleNonText'] + \
                        [volObj.spatialRef.PixelExtentInWorldZ]

            # Scale name
            # Always isotropic resampling, so the first entry is ok.
            self.Params['scaleName'] = 'scale'+(str(Params['scaleNonText'][0])).replace('.', 'dot')

            # IH name
            IHvalName = 'bin' + (str(Params['IH']['val'])).replace('.', 'dot')

            # The minimum value defines the computation.
            if Params['IH']['type'].find('FBS')>=0:
                if type(Params['userSetMinVal']) is list and Params['userSetMinVal']:
                    minValName = '_min' + \
                        ((str(Params['userSetMinVal'])).replace('.', 'dot')).replace('-', 'M')
                else:
                    # Otherwise, minimum value of ROI will be used (not recommended),
                    # so no need to report it.
                    minValName = ''
            else:
                minValName = ''

            self.Params['IHname'] = self.Params['scaleName'] + '_algo' + Params['IH']['type'] + '_' + IHvalName + minValName

            # IVH name
            if not Params['IVH']:  # CT case
                IVHAlgoName = 'algoNone'
                IVHvalName = 'bin1'
                if Params['im_range']:  # The im_range defines the computation.
                    minValName = ((str(Params['im_range'][0])).replace(
                        '.', 'dot')).replace('-', 'M')
                    maxValName = ((str(Params['im_range'][1])).replace(
                        '.', 'dot')).replace('-', 'M')
                    rangeName = '_min' + minValName + '_max' + maxValName
                else:
                    rangeName = ''
            else:
                IVHAlgoName = 'algo' + Params['IVH']['type']
                IVHvalName = 'bin' + (str(Params['IVH']['val'])).replace('.', 'dot')
                # The im_range defines the computation.
                if 'type' in Params['IVH'] and Params['IVH']['type'].find('FBS') >=0:
                    if Params['im_range']:
                        minValName = ((str(Params['im_range'][0])).replace(
                            '.', 'dot')).replace('-', 'M')
                        maxValName = ((str(Params['im_range'][1])).replace(
                            '.', 'dot')).replace('-', 'M')
                        if maxValName == 'inf':
                            # In this case, the maximum value of the ROI is used,
                            # so no need to report it.
                            rangeName = '_min' + minValName
                        elif minValName == '-inf':
                            # In this case, the minimum value of the ROI is used,
                            # so no need to report it.
                            rangeName = '_max' + maxValName
                        else:
                            rangeName = '_min' + minValName + '_max' + maxValName

                    else:  # min-max of ROI will be used, no need to report it.
                        rangeName = ''

                else:  # min-max of ROI will be used, no need to report it.
                    rangeName = ''

            self.Params['IVHname'] = self.Params['scaleName'] + '_' + IVHAlgoName + '_' + IVHvalName + rangeName
            self.Params = Params

            #Now initialize the attribute that will hold the computation results
            self.results = { 'morph_3D': { self.Params['scaleName'] : {} },
                    'locInt_3D': { self.Params['scaleName'] : {} },
                    'stats_3D': { self.Params['scaleName'] : {} },
                    'intHist_3D': { self.Params['IHname'] : {} },
                    'intVolHist_3D': { self.Params['IVHname'] : {} } }

        except Exception as e:
            message = "\n PROBLEM WITH PRE-PROCESSING OF FEATURES IN init_NTF_Calculation(): " \
                    "\n {}".format(e)
            _logger.error(message)
            print(message)

            self.Params['radiomics']['image'].update(
                {('scale'+(str(self.Params['scaleNonText'][0])).replace('.', 'dot')): 'ERROR_PROCESSING'})

    def init_TF_Calculation(self, Algo:int, Gl:int, Scale:int):

        self.nameTextTypes = ['glcm_3Dmrg', 'glrlm_3Dmrg',
                        'glszm_3D', 'gldzm_3D', 'ngtdm_3D', 'ngldm_3D']
        nTextTypes = len(self.nameTextTypes)

        if 'texture' in self.Params['radiomics']['image']:
            #Params Dict is already initialized
            pass
        else:
            self.Params['radiomics']['image'].update({'texture': {}})

            for t in range(nTextTypes):
                self.Params['radiomics']['image']['texture'].update({self.nameTextTypes[t]: {}})
        # Scale name
        # Always isotropic resampling, so the first entry is ok.
        scaleName = 'scale'+(str(self.Params['scaleText'][Scale][0])).replace('.', 'dot')

        # Discretisation name
        grayLevelsName = (str(self.Params['grayLevels'][Algo][Gl])).replace('.', 'dot')

        if 'FBS' in self.Params['algo'][Algo]:  # The minimum value defines the computation.
            if type(self.Params['userSetMinVal']) is list and self.Params['userSetMinVal']:
                minValName = '_min' + \
                    ((str(self.Params['userSetMinVal'])).replace('.', 'dot')).replace('-', 'M')
            else:
                # Otherwise, minimum value of ROI will be used (not recommended),
                # so no need to report it.
                minValName = ''
        else:
            minValName = ''

        if 'equal'in self.Params['algo'][Algo]:
            # The number of gray-levels used for equalization is currently
            # hard-coded to 64 in equalization.m
            discretisationName = 'algo' + self.Params['algo'][Algo] + '256_bin' + grayLevelsName + minValName
        else:
            discretisationName = 'algo' + self.Params['algo'][Algo] + '_bin' + grayLevelsName + minValName

        # Processing full name
        processingName = scaleName + '_' + discretisationName
        
        self.results.update( {'glcm_3Dmrg': {processingName: {}},
                                'glrlm_3Dmrg': {processingName: {}},
                                'glszm_3D': {processingName: {}},
                                'gldzm_3D': {processingName: {}},
                                'ngtdm_3D': {processingName: {}},
                                'ngldm_3D': {processingName: {}}} )

        setattr(self, 'scaleName', scaleName)
        setattr(self, 'processingName', processingName)

    def updateRadiomics(self, IntVolHistFeatures, MORPHFeatures, LocalIntensityFeatures, StatsFeatures, IntHistFeatures,
                        GLCMFeatures, GLRLMFeatures, GLSZMFeatures, GLDZMFeatures, NGTDMFeatures, NGLDMFeatures):
        TextureFeatures = ['intVolHist_3D','morph_3D','locInt_3D','stats_3D','intHist_3D']
        #Non-Texture Features
        self.results['intVolHist_3D'][self.Params['IVHname']] = IntVolHistFeatures
        self.results['morph_3D'][self.Params['scaleName']] = MORPHFeatures
        self.results['locInt_3D'][self.Params['scaleName']] = LocalIntensityFeatures
        self.results['stats_3D'][self.Params['scaleName']] = StatsFeatures
        self.results['intHist_3D'][self.Params['IHname']] = IntHistFeatures

        #Done with non-texture features, update Params with the new results
        for TF in TextureFeatures:
            self.Params['radiomics']['image'][TF] = self.results[TF]
        
        #Texture Features
        self.results['glcm_3Dmrg'][self.processingName] = GLCMFeatures
        self.results['glrlm_3Dmrg'][self.processingName] = GLRLMFeatures
        self.results['glszm_3D'][self.processingName] = GLSZMFeatures
        self.results['gldzm_3D'][self.processingName] = GLDZMFeatures
        self.results['ngtdm_3D'][self.processingName] = NGTDMFeatures
        self.results['ngldm_3D'][self.processingName] = NGLDMFeatures
        #update the radiomics parameters with all the results of the calculated texture features
        for t in range(len(self.nameTextTypes)):
            self.Params['radiomics']['image']['texture'][self.nameTextTypes[t]].update(
                self.results[self.nameTextTypes[t]])

    def save_radiomics_structure(self, scan_file_name, path_save, type_of_roi, label_of_roi_type, patient_num):
        """
        Saves extracted radiomics features in a JSON file.
        """
        self.Params['radiomics']['imParam']['roiType'] = type_of_roi
        self.Params['radiomics']['imParam']['patientID'] = self.patientID
        self.Params['radiomics']['imParam']['voxDim'] = list([self.scan.volume.spatialRef.PixelExtentInWorldX, 
                                                            self.scan.volume.spatialRef.PixelExtentInWorldY,
                                                            self.scan.volume.spatialRef.PixelExtentInWorldZ])

        indDot = scan_file_name[patient_num].find('.')
        ext = scan_file_name[patient_num].find('.npy')
        nameSave = scan_file_name[patient_num][:indDot] + \
            '(' + label_of_roi_type + ')' + scan_file_name[patient_num][indDot:ext]

        os.chdir(path_save)

        # IMPORTANT: HERE, WE COULD ADD SOME CODE TO APPEND A NEW "radiomics"
        # STRUCTURE TO AN EXISTING ONE WITH THE SAME NAME IN "pathSave"
        with open(f"{nameSave}.json", "w") as fp:   
            dump(self.Params['radiomics'], fp, indent=4, cls=NumpyEncoder)