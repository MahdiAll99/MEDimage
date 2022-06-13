import logging
import os
from json import dump
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from numpyencoder import NumpyEncoder
from PIL import Image
from pydicom.dataset import FileDataset

from MEDimage.processing.compute_suv_map import compute_suv_map

from .utils.imref import imref3d


class MEDimage(object):
    """Organizes all scan data (patient_id, imaging data, scan type...). 

    Args:
        MEDimg (MEDimage, optional): A MEDimage instance.

    Attributes:
        patient_id (str): Patient ID.
        type (str): Scan type (MRscan, CTscan...).
        format (str): Scan file format. Either 'npy' or 'nifti'.
        dicom_h (pydicom.dataset.FileDataset): DICOM header.
        scan (MEDimage.scan): Instance of MEDimage.scan inner class.

    """

    def __init__(self, MEDimg=None, logger=None) -> None:
        try:
            self.patient_id = MEDimg.patient_id
        except:
            self.patient_id = ""
        try:
            self.type = MEDimg.type
        except:
            self.type = ""
        try:
            self.format = MEDimg.format
        except:
            self.format = ""
        try:
            self.dicom_h = MEDimg.dicom_h
        except:
            self.dicom_h = []
        try:
            self.scan = MEDimg.scan
        except:
            self.scan = self.scan()
        
        # processing & computation attributes
        self.params = {}  # TODO: create object params
        self.results = {} # TODO: create object result
        self.nScale = 0 
        self.nAlgo = 0
        self.nGl = 0
        self.nExp = 0
        self.skip = False
        self.scale_name = ''
        self.processing_name = ''
        self.name_text_types = []
        if logger == None:
            self._logger = 'MEDimage.log'
        else:
            self._logger = logger
        logging.basicConfig(filename=self._logger, level=logging.DEBUG)

    def init_Params(self, imParamScan, imParamFilter, **kwargs):

        try:
            box_string = 'box10'
            # 10 voxels in all three dimensions are added to the smallest
            # bounding box. This setting is used to speed up interpolation
            # processes (mostly) prior to the computation of radiomics
            # features. Optional argument in the function computeRadiomics.

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
                _compute_SUV_map = imParamScan['image']['compute_suv_map']
            else :
                _compute_SUV_map = False
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
                self.params['box_string'] = box_string
            
            self.params['radiomics'] = radiomics
            self.params['filter'] = imParamFilter
            self.params['radiomics']['imParam'] = imParamScan
            self.params['scaleNonText'] = scaleNonText
            self.params['volInterp'] = volInterp
            self.params['roiInterp'] = roiInterp
            self.params['glRound'] = glRound
            self.params['roiPV'] = roiPV
            self.params['im_range'] = im_range
            self.params['outliers'] = outliers
            self.params['IH'] = IH
            self.params['IVH'] = IVH
            self.params['scaleText'] = scaleText
            self.params['algo'] = algo
            self.params['grayLevels'] = grayLevels
            self.params['im_type'] = im_type
            self.params['intensity'] = intensity
            self.params['computeDiagFeatures'] = computeDiagFeatures
            self.params['distCorrection'] = distCorrection
            self.params['box_string'] = box_string
            self.params['scaleName'] = ''
            self.params['IHname'] = ''
            self.params['IVHname'] = ''

            for key, value in kwargs.items():
                try:
                    self.params[key] = value
                except:
                    pass

            if self.params['box_string'] is None:
                # box_string argument is optional. If not present, we use the full box.
                self.params['box_string'] = 'full'

            # SETTING UP userSetMinVal
            if self.params['im_range'] is not None and type(self.params['im_range']) is list and self.params['im_range']:
                userSetMinVal = self.params['im_range'][0]
                if userSetMinVal == -np.inf:
                    # In case no re-seg im_range is defined for the FBS algorithm,
                    # the minimum value of ROI will be used (not recommended).
                    userSetMinVal = []
            else:
                # In case no re-seg im_range is defined for the FBS algorithm,
                # the minimum value of ROI will be used (not recommended).
                userSetMinVal = [] 
            self.params['userSetMinVal'] = userSetMinVal
            self.nScale = len(self.params['scaleText'])
            self.nAlgo = len(self.params['algo'])
            self.nGl = len(self.params['grayLevels'][0])
            self.nExp = self.nScale * self.nAlgo * self.nGl
            if self.type == 'PTscan' and _compute_SUV_map:
                try:
                    self.scan.volume.data = compute_suv_map(self.scan.volume.data, self.dicom_h[0])
                except Exception as e :
                    message = f"\n ERROR COMPUTING SUV MAP - SOME FEATURES WILL BE INVALID: \n {e}"
                    logging.error(message)
                    print(message)
                    self.skip = True

        except Exception as e:
            message = f"\n ERROR IN INITIALIZATION OF RADIOMICS FEATURE COMPUTATION\n {e}"
            logging.error(message)
            print(message)
            self.skip = True

    def init_ntf_calculation(self, vol_obj):
        """
        Initializes all the computation parameters for NON-TEXTURE FEATURES 
        as well as the results dict.
        """

        try:
            if sum(self.params['scaleNonText']) == 0:  # In case the user chose to not interpolate
                self.params['scaleNonText'] = [
                                        vol_obj.spatial_ref.PixelExtentInWorldX,
                                        vol_obj.spatial_ref.PixelExtentInWorldY,
                                        vol_obj.spatial_ref.PixelExtentInWorldZ]
            else:
                if len(self.params['scaleNonText']) == 2:
                    # In case not interpolation is performed in
                    # the slice direction (e.g. 2D case)
                    self.params['scaleNonText'] = self.params['scaleNonText'] + \
                        [vol_obj.spatial_ref.PixelExtentInWorldZ]

            # Scale name
            # Always isotropic resampling, so the first entry is ok.
            self.params['scaleName'] = 'scale' + (str(self.params['scaleNonText'][0])).replace('.', 'dot')

            # IH name
            IHvalName = 'bin' + (str(self.params['IH']['val'])).replace('.', 'dot')

            # The minimum value defines the computation.
            if self.params['IH']['type'].find('FBS')>=0:
                if type(self.params['userSetMinVal']) is list and self.params['userSetMinVal']:
                    minValName = '_min' + \
                        ((str(self.params['userSetMinVal'])).replace('.', 'dot')).replace('-', 'M')
                else:
                    # Otherwise, minimum value of ROI will be used (not recommended),
                    # so no need to report it.
                    minValName = ''
            else:
                minValName = ''

            self.params['IHname'] = self.params['scaleName'] + '_algo' + self.params['IH']['type'] + '_' + IHvalName + minValName

            # IVH name
            if not self.params['IVH']:  # CT case
                IVHAlgoName = 'algoNone'
                IVHvalName = 'bin1'
                if self.params['im_range']:  # The im_range defines the computation.
                    minValName = ((str(self.params['im_range'][0])).replace(
                        '.', 'dot')).replace('-', 'M')
                    maxValName = ((str(self.params['im_range'][1])).replace(
                        '.', 'dot')).replace('-', 'M')
                    rangeName = '_min' + minValName + '_max' + maxValName
                else:
                    rangeName = ''
            else:
                IVHAlgoName = 'algo' + self.params['IVH']['type']
                IVHvalName = 'bin' + (str(self.params['IVH']['val'])).replace('.', 'dot')
                # The im_range defines the computation.
                if 'type' in self.params['IVH'] and self.params['IVH']['type'].find('FBS') >=0:
                    if self.params['im_range']:
                        minValName = ((str(self.params['im_range'][0])).replace(
                            '.', 'dot')).replace('-', 'M')
                        maxValName = ((str(self.params['im_range'][1])).replace(
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

            self.params['IVHname'] = self.params['scaleName'] + '_' + IVHAlgoName + '_' + IVHvalName + rangeName

            # Now initialize the attribute that will hold the computation results
            self.results = { 
                            'morph_3D': {self.params['scaleName'] : {}},
                            'locInt_3D': {self.params['scaleName'] : {}},
                            'stats_3D': {self.params['scaleName'] : {}},
                            'intHist_3D': {self.params['IHname'] : {}},
                            'intVolHist_3D': {self.params['IVHname'] : {}} 
                            }

        except Exception as e:
            message = f"\n PROBLEM WITH PRE-PROCESSING OF FEATURES IN init_NTF_Calculation(): \n {e}"
            logging.error(message)
            print(message)

            self.params['radiomics']['image'].update(
                {('scale'+(str(self.params['scaleNonText'][0])).replace('.', 'dot')): 'ERROR_PROCESSING'})

    def init_tf_Calculation(self, Algo:int, Gl:int, Scale:int, params=None):
        """
        Initializes all the computation parameters for TEXTURE FEATURES 
        as well as the results dict.
        """

        if not hasattr(self, "params"):
            self.params = params

        self.nameTextTypes = ['glcm_3Dmrg', 'glrlm_3Dmrg', 'glszm_3D', 'gldzm_3D', 'ngtdm_3D', 'ngldm_3D']
        nTextTypes = len(self.nameTextTypes)

        if not ('texture' in self.params['radiomics']['image']):
            self.params['radiomics']['image'].update({'texture': {}})
            for t in range(nTextTypes):
                self.params['radiomics']['image']['texture'].update({self.nameTextTypes[t]: {}})

        # Scale name
        # Always isotropic resampling, so the first entry is ok.
        scaleName = 'scale' + (str(self.params['scaleText'][Scale][0])).replace('.', 'dot')

        # Discretisation name
        grayLevelsName = (str(self.params['grayLevels'][Algo][Gl])).replace('.', 'dot')

        if 'FBS' in self.params['algo'][Algo]:  # The minimum value defines the computation.
            if type(self.params['userSetMinVal']) is list and self.params['userSetMinVal']:
                minValName = '_min' + ((str(self.params['userSetMinVal'])).replace('.', 'dot')).replace('-', 'M')
            else:
                # Otherwise, minimum value of ROI will be used (not recommended),
                # so no need to report it.
                minValName = ''
        else:
            minValName = ''

        if 'equal'in self.params['algo'][Algo]:
            # The number of gray-levels used for equalization is currently
            # hard-coded to 64 in equalization.m
            discretisationName = 'algo' + self.params['algo'][Algo] + '256_bin' + grayLevelsName + minValName
        else:
            discretisationName = 'algo' + self.params['algo'][Algo] + '_bin' + grayLevelsName + minValName

        # Processing full name
        processingName = scaleName + '_' + discretisationName
        
        self.results.update({
                            'glcm_3Dmrg': {processingName: {}},
                            'glrlm_3Dmrg': {processingName: {}},
                            'glszm_3D': {processingName: {}},
                            'gldzm_3D': {processingName: {}},
                            'ngtdm_3D': {processingName: {}},
                            'ngldm_3D': {processingName: {}}
                            })

        if hasattr("scaleName"):
            setattr(self, 'scaleName', scaleName)
        else:
            self.scaleName = scaleName

        if hasattr("processingName"):
            setattr(self, 'processingName', processingName)
        else:
            self.processingName = processingName

    def init_from_nifti(self, NiftiImagePath) -> None:
        """Initializes the MEDimage class using a NIFTI file.

        Args:
            NiftiImagePath (Path): NIFTI file path.

        Returns:
            None.
        
        """
        self.patient_id = os.path.basename(NiftiImagePath).split("_")[0]
        self.type = os.path.basename(NiftiImagePath).split(".")[-3]
        self.format = "nifti"
        self.scan.set_orientation(orientation="Axial")
        self.scan.set_patientPosition(patientPosition="HFS")
        self.scan.ROI.get_ROI_from_path(ROI_path=os.path.dirname(NiftiImagePath), 
                                        ID=Path(NiftiImagePath).name.split("(")[0])
        self.scan.volume.data = nib.load(NiftiImagePath).get_fdata()
        # RAS to LPS
        self.scan.volume.convert_to_LPS()
        self.scan.volume.scanRot = None
    
    def update_radiomics(
                        self, int_vol_hist_features, morph_features, loc_int_features, 
                        stats_features, int_hist_features,
                        GLCM_features, GLRLM_features, GLSZM_features, 
                        GLDZM_features, NGTDM_features, NGLDM_features):
        """
        Updates the results attribute with the extracted features
        """
        TextureFeatures = ['intVolHist_3D','morph_3D','locInt_3D','stats_3D','intHist_3D']
        #Non-Texture Features
        self.results['intVolHist_3D'][self.params['IVHname']] = int_vol_hist_features
        self.results['morph_3D'][self.params['scaleName']] = morph_features
        self.results['locInt_3D'][self.params['scaleName']] = loc_int_features
        self.results['stats_3D'][self.params['scaleName']] = stats_features
        self.results['intHist_3D'][self.params['IHname']] = int_hist_features

        #Done with non-texture features, update params with the new results
        for feature in TextureFeatures:
            self.params['radiomics']['image'][feature] = self.results[feature]
        
        #Texture Features
        self.results['glcm_3Dmrg'][self.processingName] = GLCM_features
        self.results['glrlm_3Dmrg'][self.processingName] = GLRLM_features
        self.results['glszm_3D'][self.processingName] = GLSZM_features
        self.results['gldzm_3D'][self.processingName] = GLDZM_features
        self.results['ngtdm_3D'][self.processingName] = NGTDM_features
        self.results['ngldm_3D'][self.processingName] = NGLDM_features

        #update the radiomics parameters with all the results of the calculated texture features
        for t in range(len(self.nameTextTypes)):
            self.params['radiomics']['image']['texture'][self.nameTextTypes[t]].update(
                self.results[self.nameTextTypes[t]])

    def save_radiomics_structure(self, scan_file_name, path_save, type_of_roi, label_of_roi_type, patient_num):
        """
        Saves extracted radiomics features in a JSON file.
        """
        path_save = Path(path_save)

        self.params['radiomics']['imParam']['roi_type'] = type_of_roi
        self.params['radiomics']['imParam']['patient_id'] = self.patient_id
        self.params['radiomics']['imParam']['vox_dim'] = list([
                                                            self.scan.volume.spatial_ref.PixelExtentInWorldX, 
                                                            self.scan.volume.spatial_ref.PixelExtentInWorldY,
                                                            self.scan.volume.spatial_ref.PixelExtentInWorldZ
                                                            ])

        indDot = scan_file_name[patient_num].find('.')
        ext = scan_file_name[patient_num].find('.npy')
        name_save = scan_file_name[patient_num][:indDot] + \
            '(' + label_of_roi_type + ')' + scan_file_name[patient_num][indDot:ext]

        # IMPORTANT: HERE, WE COULD ADD SOME CODE TO APPEND A NEW "radiomics"
        # STRUCTURE TO AN EXISTING ONE WITH THE SAME NAME IN "path_save"
        with open(path_save / f"{name_save}.json", "w") as fp:   
            dump(self.params['radiomics'], fp, indent=4, cls=NumpyEncoder)

    class scan:
        """Organizes all imaging data (volume and ROI). 

        Args:
            orientation (str, optional): Imaging data orientation (axial, sagittal or coronal).
            patientPosition (str, optional): Patient position specifies the position of the 
                patient relative to the imaging equipment space (HFS, HFP...).

        Attributes:
            volume (object): Instance of MEDimage.scan.volume inner class.
            ROI (object): Instance of MEDimage.scan.ROI inner class.
            orientation (str): Imaging data orientation (axial, sagittal or coronal).
            patientPosition (str): Patient position specifies the position of the 
                patient relative to the imaging equipment space (HFS, HFP...).

        """
        def __init__(self, orientation=None, patientPosition=None):
            self.volume = self.volume() 
            self.ROI = self.ROI()
            self.orientation = orientation
            self.patientPosition = patientPosition

        def set_patientPosition(self, patientPosition):
            self.patientPosition = patientPosition

        def set_orientation(self, orientation):
            self.orientation = orientation
        
        def set_volume(self, volume):
            self.volume = volume
        
        def set_ROI(self, *args):
            self.ROI = self.ROI(args)

        def get_ROI_from_indexes(self, key):
            """
            Extract ROI data using the saved indexes (Indexes of 1's).

            Args:
                ket (int): ROI index (A volume can have multiple ROIs).

            Returns:
                ndarray: n-dimensional array of ROI data.
            
            """
            roi_volume = np.zeros_like(self.volume.data).flatten()
            roi_volume[self.ROI.get_indexes(key)] = 1
            return roi_volume.reshape(self.volume.data.shape)

        def get_indexes_by_ROIname(self, ROIname : str):
            """
            Extract ROI data using ROI name..

            Args:
                ROIname (str): String of the ROI name (A volume can have multiple ROIs).

            Returns:
                ndarray: n-dimensional array of ROI data.
            
            """
            ROIname_key = list(self.ROI.roi_names.values()).index(ROIname)
            roi_volume = np.zeros_like(self.volume.data).flatten()
            roi_volume[self.ROI.get_indexes(ROIname_key)] = 1
            return roi_volume.reshape(self.volume.data.shape)

        def display(self, _slice: int = None) -> None:
            """Displays slices from imaging data with the ROI contour in XY-Plane.

            Args:
                _slice (int, optional): Index of the slice you want to plot.

            Returns:
                None.
            
            """
            # extract slices containing ROI
            size_m = self.volume.data.shape
            i = np.arange(0, size_m[0])
            j = np.arange(0, size_m[1])
            k = np.arange(0, size_m[2])
            ind_mask = np.nonzero(self.get_ROI_from_indexes(0))
            J, I, K = np.meshgrid(j, i, k, indexing='ij')
            I = I[ind_mask]
            J = J[ind_mask]
            K = K[ind_mask]
            slices = np.unique(K)

            vol_data = self.volume.data.swapaxes(0, 1)[:, :, slices]
            roi_data = self.get_ROI_from_indexes(0).swapaxes(0, 1)[:, :, slices]        
            
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
            """Organizes all volume data and information. 

            Args:
                spatial_ref (imref3d, optional): Imaging data orientation (axial, sagittal or coronal).
                scanRot (ndarray, optional): Array of the rotation applied to the XYZ points of the ROI.
                data (ndarray, optional): n-dimensional of the imaging data.
                filtered_data (Dict[ndarray]): Dict of n-dimensional arrays of the filtered 
                        imaging data.

            Attributes:
                spatial_ref (imref3d): Imaging data orientation (axial, sagittal or coronal).
                scanRot (ndarray): Array of the rotation applied to the XYZ points of the ROI.
                data (ndarray): n-dimensional of the imaging data.
                filtered_data (Dict[ndarray]): Dict of n-dimensional arrays of the filtered 
                    imaging data.

            """
            def __init__(self, spatial_ref=None, scanRot=None, data=None, filtered_data={}):
                """Organizes all volume data and information. 

                Args:
                    spatial_ref (imref3d, optional): Imaging data orientation (axial, sagittal or coronal).
                    scanRot (ndarray, optional): Array of the rotation applied to the XYZ points of the ROI.
                    data (ndarray, optional): n-dimensional of the imaging data.
                    filtered_data (Dict[ndarray]): Dict of n-dimensional arrays of the filtered 
                        imaging data.

                """
                self.spatial_ref = spatial_ref
                self.scanRot = scanRot
                self.data = data
                self.filtered_data = filtered_data

            def update_spatial_ref(self, spatial_ref_value):
                self.spatial_ref = spatial_ref_value
            
            def update_scanRot(self, scanRot_value):
                self.scanRot = scanRot_value
            
            def update_transScanToModel(self, transScanToModel_value):
                self.transScanToModel = transScanToModel_value
            
            def update_data(self, data_value):
                self.data = data_value
            
            def update_filtered_data(self, filter_name, new_data):
                if not hasattr(self, 'filtered_data'):
                    self.filtered_data = {}
                self.filtered_data[filter_name] = new_data

            def save_filtered_data(self, name_save: str, path: Union[Path, str]):
                path = Path(path)
                if not name_save.endswith('.npy'):
                    name_save += '.npy'
                with open(path / name_save, 'wb') as f:
                    np.save(f, self.filtered_data)

            def load_filtered_data(self, filter_name, file_name, path, update=True):
                path = Path(path)
                if not file_name.endswith('.npy'):
                    file_name += '.npy'
                with open(path / file_name, 'rb') as f:
                    if update:
                        self.update_filtered_data(filter_name, np.load(f, allow_pickle=True))
                    else:
                        return np.load(f, allow_pickle=True)

            def convert_to_LPS(self):
                """Convert Imaging data to LPS (Left-Posterior-Superior) coordinates system.
                <https://www.slicer.org/wiki/Coordinate_systems>.

                Args:
                    ket (int): ROI index (A volume can have multiple ROIs).

                Returns:
                    None.

                """
                # flip x
                self.data = np.flip(self.data, 0)
                # flip y
                self.data = np.flip(self.data, 1)
                # to LPS
                self.data = self.data.swapaxes(0, 1) #TODO
            
            def spatial_ref_from_NIFTI(self, nifti_image_path):
                """Computes the imref3d spatial_ref using a NIFTI file and
                updates the spatial_ref attribute.

                Args:
                    NiftiImagePath (str): String of the NIFTI file path.

                Returns:
                    None.
                
                """
                # Loading the nifti file :
                nifti = nib.load(nifti_image_path)
                nifti_data = self.data

                # spatial_ref Creation
                pixel_x = nifti.affine[0, 0]
                pixel_y = nifti.affine[1, 1]
                slice_s = nifti.affine[2, 2]
                min_grid = nifti.affine[:3, 3]
                min_x_grid = min_grid[0]
                min_y_grid = min_grid[1]
                min_z_grid = min_grid[2]
                size_image = np.shape(nifti_data)
                spatial_ref = imref3d(size_image, abs(pixel_x), abs(pixel_y), abs(slice_s))
                spatial_ref.XWorldLimits = (np.array(spatial_ref.XWorldLimits) -
                                        (spatial_ref.XWorldLimits[0] -
                                            (min_x_grid-pixel_x/2))
                                        ).tolist()
                spatial_ref.YWorldLimits = (np.array(spatial_ref.YWorldLimits) -
                                        (spatial_ref.YWorldLimits[0] -
                                            (min_y_grid-pixel_y/2))
                                        ).tolist()
                spatial_ref.ZWorldLimits = (np.array(spatial_ref.ZWorldLimits) -
                                        (spatial_ref.ZWorldLimits[0] -
                                            (min_z_grid-slice_s/2))
                                        ).tolist()

                # Converting the results into lists
                spatial_ref.ImageSize = spatial_ref.ImageSize.tolist()
                spatial_ref.XIntrinsicLimits = spatial_ref.XIntrinsicLimits.tolist()
                spatial_ref.YIntrinsicLimits = spatial_ref.YIntrinsicLimits.tolist()
                spatial_ref.ZIntrinsicLimits = spatial_ref.ZIntrinsicLimits.tolist()

                # update spatial_ref
                self.update_spatial_ref(spatial_ref)

            def convert_spatial_ref(self):
                """converts the MEDimage spatial_ref from RAS to LPS coordinates system.
                <https://www.slicer.org/wiki/Coordinate_systems>.

                Args:
                    None.

                Returns:
                    None.

                """
                # swap x and y data
                temp = self.spatial_ref.ImageExtentInWorldX
                self.spatial_ref.ImageExtentInWorldX = self.spatial_ref.ImageExtentInWorldY
                self.spatial_ref.ImageExtentInWorldY = temp

                temp = self.spatial_ref.PixelExtentInWorldX
                self.spatial_ref.PixelExtentInWorldX = self.spatial_ref.PixelExtentInWorldY
                self.spatial_ref.PixelExtentInWorldY = temp

                temp = self.spatial_ref.XIntrinsicLimits
                self.spatial_ref.XIntrinsicLimits = self.spatial_ref.YIntrinsicLimits
                self.spatial_ref.YIntrinsicLimits = temp

                temp = self.spatial_ref.XWorldLimits
                self.spatial_ref.XWorldLimits = self.spatial_ref.YWorldLimits
                self.spatial_ref.YWorldLimits = temp
                del temp

        class ROI:
            """Organizes all ROI data and information. 

            Args:
                indexes (Dict, optional): Dict of the ROI indexes for each ROI name.
                roi_names (Dict, optional): Dict of the ROI names.

            Attributes:
                indexes (Dict): Dict of the ROI indexes for each ROI name.
                roi_names (Dict): Dict of the ROI names.
                name_set (Dict): Dict of the User-defined name for Structure Set for each ROI name.
                name_set_info (Dict): Dict of the names of the structure sets that define the areas of 
                    significance. Either 'StructureSetName', 'StructureSetDescription', 'series_description' 
                    or 'SeriesInstanceUID'.

            """
            def __init__(self, indexes=None, roi_names=None) -> None:
                self.indexes = indexes if indexes else {}
                self.roi_names = roi_names if roi_names else {}
                self.name_set = roi_names if roi_names else {}
                self.name_set_info = roi_names if roi_names else {}

            def get_indexes(self, key):
                if not self.indexes or key is None:
                    return {}
                else:
                    return self.indexes[str(key)]

            def get_ROIname(self, key):
                if not self.roi_names or key is None:
                    return {}
                else:
                    return self.roi_names[str(key)]

            def get_nameSet(self, key):
                if not self.name_set or key is None:
                    return {}
                else:
                    return self.name_set[str(key)]

            def get_nameSetInfo(self, key):
                if not self.name_set_info or key is None:
                    return {}
                else:
                    return self.name_set_info[str(key)]

            def update_indexes(self, key, indexes):
                try: 
                    self.indexes[str(key)] = indexes
                except:
                    Warning.warn("Wrong key given in update_indexes()")

            def update_ROIname(self, key, ROIname):
                try:
                    self.roi_names[str(key)] = ROIname
                except:
                    Warning.warn("Wrong key given in update_ROIname()")

            def update_nameSet(self, key, name_set):
                try:
                    self.name_set[str(key)] = name_set
                except:
                    Warning.warn("Wrong key given in update_nameSet()")

            def update_nameSetInfo(self, key, name_set_info):
                try:
                    self.name_set_info[str(key)] = name_set_info
                except:
                    Warning.warn("Wrong key given in update_nameSetInfo()")
            
            def convert_to_LPS(self, data):
                """
                -------------------------------------------------------------------------
                DESCRIPTION:
                This function converts the given volume to LPS coordinates system. For 
                more details please refer here : https://www.slicer.org/wiki/Coordinate_systems 
                -------------------------------------------------------------------------
                INPUTS:
                - data : given volume data in RAS to be converted to LPS
                -------------------------------------------------------------------------
                OUTPUTS:
                - data in LPS.
                -------------------------------------------------------------------------
                """
                # flip x
                data = np.flip(data, 0)
                # flip y
                data = np.flip(data, 1)
                # to LPS
                data = data.swapaxes(0, 1)

                return data

            def get_ROI_from_path(self, ROI_path, ID):
                """
                -------------------------------------------------------------------------
                DESCRIPTION:
                This function extracts all ROI data from the given path for the given
                patient ID and updates all class attributes with the new extracted data.
                This method is called only once for NIFTI formats per patient.
                -------------------------------------------------------------------------
                INPUTS:
                - ROI_path : Path where the ROI data is stored
                - ID : The ID contains patient ID and the modality type, which makes it
                possible for the method to extract the right data.
                -------------------------------------------------------------------------
                OUTPUTS:
                - NO OUTPUTS.
                -------------------------------------------------------------------------
                """
                self.indexes = {}
                self.roi_names = {}
                self.name_set = {}
                self.name_set_info = {}
                roi_index = 0
                ListOfPatients = os.listdir(ROI_path)

                for file in ListOfPatients:
                    # Load the patient's ROI nifti files :
                    if file.startswith(ID) and file.endswith('nii.gz') and 'ROI' in file.split("."):
                        roi = nib.load(ROI_path + "/" + file)
                        roi_data = self.convert_to_LPS(data=roi.get_fdata())
                        roi_name = file[file.find("(")+1:file.find(")")]
                        name_set = file[file.find("_")+2:file.find("(")]
                        self.update_indexes(key=roi_index, indexes=np.nonzero(roi_data.flatten()))
                        self.update_nameSet(key=roi_index, name_set=name_set)
                        self.update_ROIname(key=roi_index, ROIname=roi_name)
                        roi_index += 1
