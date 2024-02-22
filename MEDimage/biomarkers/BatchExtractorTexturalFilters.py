import logging
import math
import os
import pickle
import sys
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from time import time
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import ray
from tqdm import trange

import MEDimage


class BatchExtractorTexturalFilters(object):
    """
    Organizes all the patients/scans in batches to extract all the radiomic features
    """

    def __init__(
            self, 
            path_read: Union[str, Path],
            path_csv: Union[str, Path],
            path_params: Union[str, Path],
            path_save: Union[str, Path],
            n_batch: int = 4
    ) -> None:
        """
        constructor of the BatchExtractor class 
        """
        self._path_csv = Path(path_csv)
        self._path_params = Path(path_params)
        self._path_read = Path(path_read)
        self._path_save = Path(path_save)
        self.roi_types = []
        self.roi_type_labels = []
        self.n_bacth = n_batch
        self.glcm_features = [
            "Fcm_joint_max",
            "Fcm_joint_avg",
            "Fcm_joint_var",
            "Fcm_joint_entr",
            "Fcm_diff_avg",
            "Fcm_diff_var",
            "Fcm_diff_entr",
            "Fcm_sum_avg",
            "Fcm_sum_var",
            "Fcm_sum_entr",
            "Fcm_energy",
            "Fcm_contrast",
            "Fcm_dissimilarity",
            "Fcm_inv_diff",
            "Fcm_inv_diff_norm",
            "Fcm_inv_diff_mom",
            "Fcm_inv_diff_mom_norm",
            "Fcm_inv_var",
            "Fcm_corr",
            "Fcm_auto_corr",
            "Fcm_clust_tend",
            "Fcm_clust_shade",
            "Fcm_clust_prom",
            "Fcm_info_corr1",
            "Fcm_info_corr2"
        ]

    def __load_and_process_params(self) -> Dict:
        """Load and process the computing & batch parameters from JSON file"""
        # Load json parameters
        im_params = MEDimage.utils.json_utils.load_json(self._path_params)
        
        # Update class attributes
        self.roi_types.extend(im_params['roi_types'])
        self.roi_type_labels.extend(im_params['roi_type_labels'])
        self.n_bacth = im_params['n_batch'] if 'n_batch' in im_params else self.n_bacth

        return im_params

    def __compute_radiomics_one_patient(
            self,
            name_patient: str,
            roi_name: str,
            im_params: Dict,
            roi_type: str,
            roi_type_label: str,
            log_file: Union[Path, str],
            skip_existing: bool
        ) -> str:
        """
        Computes all radiomics features (Texture & Non-texture) for one patient/scan

        Args:
            name_patient(str): scan or patient full name. It has to respect the MEDimage naming convention:
                PatientID__ImagingScanName.ImagingModality.npy
            roi_name(str): name of the ROI that will  be used in computation.
            im_params(Dict): Dict of parameters/settings that will be used in the processing and computation.
            roi_type(str): Type of ROI used in the processing and computation (for identification purposes)
            roi_type_label(str): Label of the ROI used, to make it identifiable from other ROIs.
            log_file(Union[Path, str]): Path to the logging file.
            skip_existing(bool): True to skip the computation of the features for the scans that already have been computed.

        Returns:
            Union[Path, str]: Path to the updated logging file.
        """
        # Check if the features for the current filter have already been computed
        if skip_existing:
            list_feature = []
            # Find the glcm filters that have not been computed yet
            for i in range(len(self.glcm_features)):
                index_dot = name_patient.find('.')
                ext = name_patient.find('.npy')
                name_save = name_patient[:index_dot] + '(' + roi_type_label + ')' + name_patient[index_dot : ext] + ".json"
                name_roi_type = roi_type + '_' + self.glcm_features[i]
                path_to_check = Path(self._path_save / f'features({name_roi_type})')
                if not (path_to_check / name_save).exists():
                    list_feature.append(i)
            # If all the features have already been computed, skip the computation
            if len(list_feature) == 0:
                return log_file
        
        # Setting up logging settings
        logging.basicConfig(filename=log_file, level=logging.DEBUG, force=True)

        # start timer
        t_start = time()

        # Initialization
        message = f"\n***************** COMPUTING FEATURES: {name_patient} *****************"
        logging.info(message)

        # Load MEDscan instance
        try:
            with open(self._path_read / name_patient, 'rb') as f: medscan = pickle.load(f)
            medscan = MEDimage.MEDscan(medscan)
        except Exception as e:
            logging.error(f"\n ERROR LOADING PATIENT {name_patient}:\n {e}")
            return None

        # Init processing & computation parameters
        medscan.init_params(im_params)
        logging.debug('Parameters parsed, json file is valid.')

        # Get ROI (region of interest)
        logging.info("\n--> Extraction of ROI mask:")
        try:
            vol_obj_init, roi_obj_init = MEDimage.processing.get_roi_from_indexes(
                medscan,
                name_roi=roi_name,
                box_string=medscan.params.process.box_string
            )
        except:
            # if for the current scan ROI is not found, computation is aborted. 
            return log_file

        start = time()
        message = '--> Non-texture features pre-processing (interp + re-seg) for "Scale={}"'.\
            format(str(medscan.params.process.scale_non_text[0]))
        logging.info(message)

        # Interpolation
        # Intensity Mask
        vol_obj = MEDimage.processing.interp_volume(
            medscan=medscan,
            vol_obj_s=vol_obj_init,
            vox_dim=medscan.params.process.scale_non_text,
            interp_met=medscan.params.process.vol_interp,
            round_val=medscan.params.process.gl_round,
            image_type='image',
            roi_obj_s=roi_obj_init,
            box_string=medscan.params.process.box_string
        )
        # Morphological Mask
        roi_obj_morph = MEDimage.processing.interp_volume(
            medscan=medscan,
            vol_obj_s=roi_obj_init,
            vox_dim=medscan.params.process.scale_non_text,
            interp_met=medscan.params.process.roi_interp,
            round_val=medscan.params.process.roi_pv,
            image_type='roi', 
            roi_obj_s=roi_obj_init,
            box_string=medscan.params.process.box_string
        )

        # Re-segmentation
        # Intensity mask range re-segmentation
        roi_obj_int = deepcopy(roi_obj_morph)
        roi_obj_int.data = MEDimage.processing.range_re_seg(
            vol=vol_obj.data, 
            roi=roi_obj_int.data,
            im_range=medscan.params.process.im_range
        )
        # Intensity mask outlier re-segmentation
        roi_obj_int.data = np.logical_and(
            MEDimage.processing.outlier_re_seg(
                vol=vol_obj.data, 
                roi=roi_obj_int.data, 
                outliers=medscan.params.process.outliers
            ),
            roi_obj_int.data
        ).astype(int)
        logging.info(f"{time() - start}\n")

        # Reset timer
        start = time()

        # Image textural filtering
        logging.info("--> Image textural filtering:")

        # Preparation of computation :
        medscan.init_ntf_calculation(vol_obj)

        # ROI Extraction :
        try:
            vol_int_re = MEDimage.processing.roi_extract(
                vol=vol_obj.data, 
                roi=roi_obj_int.data
            )
        except Exception as e:
            print(name_patient, e)
            return log_file

        # Apply textural filter
        try:
            if medscan.params.process.user_set_min_value is None:
                medscan.params.process.user_set_min_value = np.nanmin(vol_int_re)
            vol_obj_all_features = MEDimage.filters.apply_filter(
                medscan, 
                vol_int_re, 
                user_set_min_val=medscan.params.process.user_set_min_value
            )
        except Exception as e:
            print(e)
            logging.error(f'PROBLEM WITH TEXTURAL FILTERING: {e}')
            return log_file
        
        # Initialize ray
        if ray.is_initialized():
            ray.shutdown()

        ray.init(local_mode=True, include_dashboard=True, num_cpus=self.n_bacth)

        # Loop through all the filters and extract the features for each filter
        ids = []
        nb_filters = len(list_feature)
        if nb_filters < self.n_bacth:
            self.n_bacth = nb_filters
        for i in range(self.n_bacth):
            # Extract the filtered volume
            filter_idx = list_feature[i]
            vol_obj.data = deepcopy(vol_obj_all_features[...,filter_idx])

            # Compute radiomics features
            logging.info(f"--> Computation of radiomics features for filter {filter_idx}:")

            ids.append(
                self.__compute_radiomics_filtered_volume.remote(
                    self,
                    medscan=medscan,
                    vol_obj=vol_obj,
                    roi_obj_int=roi_obj_int,
                    roi_obj_morph=roi_obj_morph,
                    name_patient=name_patient,
                    roi_name=roi_name,
                    roi_type=roi_type + '_' + self.glcm_features[filter_idx],
                    roi_type_label=roi_type_label,
                    log_file=log_file
                )
            )
        # Distribute the remaining tasks
        nb_job_left = nb_filters - self.n_bacth
        if nb_job_left > 0:
            for i in range(nb_filters - nb_job_left, nb_filters):
                ready, not_ready = ray.wait(ids, num_returns=1)
                ids = not_ready
                try:
                    log_file = ray.get(ready)[0]
                except:
                    pass
                # Extract the filtered volume
                filter_idx = list_feature[i]
                vol_obj.data = deepcopy(vol_obj_all_features[...,filter_idx])

                # Compute radiomics features
                logging.info(f"--> Computation of radiomics features for filter {filter_idx}:")

                ids.append(
                    self.__compute_radiomics_filtered_volume.remote(
                        self,
                        medscan=medscan,
                        vol_obj=vol_obj,
                        roi_obj_int=roi_obj_int,
                        roi_obj_morph=roi_obj_morph,
                        name_patient=name_patient,
                        roi_name=roi_name,
                        roi_type=roi_type + '_' + self.glcm_features[filter_idx],
                        roi_type_label=roi_type_label,
                        log_file=log_file
                    )
                )
        
        logging.info(f"TOTAL TIME:{time() - t_start} seconds\n\n")

        # Empty memory
        del medscan

    @ray.remote
    def __compute_radiomics_filtered_volume(
            self,
            medscan: MEDimage.MEDscan,
            vol_obj,
            roi_obj_int,
            roi_obj_morph,
            name_patient,
            roi_name,
            roi_type,
            roi_type_label,
            log_file
        ) -> Union[Path, str]:

        # time
        t_start = time()

        # ROI Extraction :
        vol_int_re = deepcopy(vol_obj.data)

        # check if ROI is empty
        if math.isnan(np.nanmax(vol_int_re)) and math.isnan(np.nanmin(vol_int_re)):
            logging.error(f'PROBLEM WITH INTENSITY MASK. ROI {roi_name} IS EMPTY.')
            return log_file
        
        # Computation of non-texture features
        logging.info("--> Computation of non-texture features:")

        # Morphological features extraction
        try:
            morph = MEDimage.biomarkers.morph.extract_all(
                vol=vol_obj.data, 
                mask_int=roi_obj_int.data, 
                mask_morph=roi_obj_morph.data,
                res=medscan.params.process.scale_non_text,
                intensity_type=medscan.params.process.intensity_type
            )
        except Exception as e:
            logging.error(f'PROBLEM WITH COMPUTATION OF MORPHOLOGICAL FEATURES {e}')
            morph = None

        # Local intensity features extraction
        try:
            local_intensity = MEDimage.biomarkers.local_intensity.extract_all(
                img_obj=vol_obj.data,
                roi_obj=roi_obj_int.data,
                res=medscan.params.process.scale_non_text,
                intensity_type=medscan.params.process.intensity_type
            )
        except Exception as e:
            logging.error(f'PROBLEM WITH COMPUTATION OF LOCAL INTENSITY FEATURES {e}')
            local_intensity = None

        # statistical features extraction
        try:
            stats = MEDimage.biomarkers.stats.extract_all(
                vol=vol_int_re,
                intensity_type=medscan.params.process.intensity_type
            )
        except Exception as e:
            logging.error(f'PROBLEM WITH COMPUTATION OF STATISTICAL FEATURES {e}')
            stats = None

        # Intensity histogram equalization of the imaging volume
        vol_quant_re, _ = MEDimage.processing.discretize(
            vol_re=vol_int_re,
            discr_type=medscan.params.process.ih['type'], 
            n_q=medscan.params.process.ih['val'], 
            user_set_min_val=medscan.params.process.user_set_min_value
        )
        
        # Intensity histogram features extraction
        try:
            int_hist = MEDimage.biomarkers.intensity_histogram.extract_all(
                vol=vol_quant_re
            )
        except Exception as e:
            logging.error(f'PROBLEM WITH COMPUTATION OF INTENSITY HISTOGRAM FEATURES {e}')
            int_hist = None
        
        # Intensity histogram equalization of the imaging volume
        if medscan.params.process.ivh and 'type' in medscan.params.process.ivh and 'val' in medscan.params.process.ivh:
            if medscan.params.process.ivh['type'] and medscan.params.process.ivh['val']:
                vol_quant_re, wd = MEDimage.processing.discretize(
                    vol_re=vol_int_re,
                    discr_type=medscan.params.process.ivh['type'], 
                    n_q=medscan.params.process.ivh['val'], 
                    user_set_min_val=medscan.params.process.user_set_min_value,
                    ivh=True
                )
        else:
            vol_quant_re = vol_int_re
            wd = 1

        # Intensity volume histogram features extraction
        try:
            int_vol_hist = MEDimage.biomarkers.int_vol_hist.extract_all(
                medscan=medscan,
                vol=vol_quant_re,
                vol_int_re=vol_int_re, 
                wd=wd
            )
        except:
            print("Error ivh:",name_patient)
            int_vol_hist = {'Fivh_V10': [],
                    'Fivh_V90': [],
                    'Fivh_I10': [],
                    'Fivh_I90': [],
                    'Fivh_V10minusV90': [],
                    'Fivh_I10minusI90': [],
                    'Fivh_auc': []
                    }

        # End of Non-Texture features extraction
        logging.info(f"End of non-texture features extraction: {time() - t_start}\n")

        # Computation of texture features
        logging.info("--> Computation of texture features:")

        # Compute radiomics features for each scale text
        count = 0
        logging.info(f"{time() - t_start}\n")
        
        # Compute features for each discretisation algorithm and for each grey-level  
        for a, n in product(range(medscan.params.process.n_algo), range(medscan.params.process.n_gl)):
            count += 1 
            start = time()
            message = '--> Computation of texture features in image ' \
                    'space for "Scale= {}", "Algo={}", "GL={}" ({}):'.format(
                        str(medscan.params.process.scale_text[0][1]),
                        medscan.params.process.algo[a],
                        str(medscan.params.process.gray_levels[a][n]),
                        str(count) + '/' + str(medscan.params.process.n_exp)
                        )
            logging.info(message)

            # Preparation of computation :
            medscan.init_tf_calculation(algo=a, gl=n, scale=0)

            # Discretisation :
            try:
                vol_quant_re, _ = MEDimage.processing.discretize(
                    vol_re=vol_int_re,
                    discr_type=medscan.params.process.algo[a], 
                    n_q=medscan.params.process.gray_levels[a][n], 
                    user_set_min_val=medscan.params.process.user_set_min_value
                )
            except Exception as e:
                logging.error(f'PROBLEM WITH DISCRETIZATION: {e}')
                vol_quant_re = None

            # GLCM features extraction
            try:
                glcm = MEDimage.biomarkers.glcm.extract_all(
                    vol=vol_quant_re, 
                    dist_correction=medscan.params.radiomics.glcm.dist_correction
                )
            except Exception as e:
                logging.error(f'PROBLEM WITH COMPUTATION OF GLCM FEATURES {e}')
                glcm = None

            # GLRLM features extraction
            try:
                glrlm = MEDimage.biomarkers.glrlm.extract_all(
                    vol=vol_quant_re,
                    dist_correction=medscan.params.radiomics.glrlm.dist_correction
                )
            except Exception as e:
                logging.error(f'PROBLEM WITH COMPUTATION OF GLRLM FEATURES {e}')
                glrlm = None

            # GLSZM features extraction
            try:
                glszm = MEDimage.biomarkers.glszm.extract_all(vol=vol_quant_re)
            except Exception as e:
                logging.error(f'PROBLEM WITH COMPUTATION OF GLSZM FEATURES {e}')
                glszm = None

            # GLDZM features extraction
            try:
                gldzm = MEDimage.biomarkers.gldzm.extract_all(
                    vol_int=vol_quant_re, 
                    mask_morph=roi_obj_morph.data
                )
            except Exception as e:
                logging.error(f'PROBLEM WITH COMPUTATION OF GLDZM FEATURES {e}')
                gldzm = None

            # NGTDM features extraction
            try:
                ngtdm = MEDimage.biomarkers.ngtdm.extract_all(
                    vol=vol_quant_re, 
                    dist_correction=medscan.params.radiomics.ngtdm.dist_correction
                )
            except Exception as e:
                logging.error(f'PROBLEM WITH COMPUTATION OF NGTDM FEATURES {e}')
                ngtdm = None

            # NGLDM features extraction
            try:
                ngldm = MEDimage.biomarkers.ngldm.extract_all(vol=vol_quant_re)
            except Exception as e:
                logging.error(f'PROBLEM WITH COMPUTATION OF NGLDM FEATURES {e}')
                ngldm = None
            
            # Update radiomics results class
            medscan.update_radiomics(
                int_vol_hist_features=int_vol_hist, 
                morph_features=morph,
                loc_int_features=local_intensity, 
                stats_features=stats, 
                int_hist_features=int_hist,
                glcm_features=glcm, 
                glrlm_features=glrlm, 
                glszm_features=glszm, 
                gldzm_features=gldzm, 
                ngtdm_features=ngtdm, 
                ngldm_features=ngldm
            )
                
        # End of texture features extraction
        logging.info(f"End of texture features extraction: {time() - start}\n")

        # Saving radiomics results
        medscan.save_radiomics(
            scan_file_name=name_patient,
            path_save=self._path_save,
            roi_type=roi_type,
            roi_type_label=roi_type_label,
        )
        
        logging.info(f"TOTAL TIME 1 FILTER:{time() - t_start} seconds\n\n")

        return log_file
    
    @ray.remote
    def __compute_radiomics_tables(
            self,
            table_tags: List,
            log_file: Union[str, Path],
            im_params: Dict,
            feature_name: str
        ) -> None:
        """
        Creates radiomic tables off of the saved dicts with the computed features and save it as CSV files

        Args:
            table_tags(List): Lists of information about scans, roi type and imaging space (or filter space)
            log_file(Union[str, Path]): Path to logging file.
            im_params(Dict): Dictionary of parameters.
        
        Returns:
            None.
        """
        n_tables = len(table_tags)

        for t in range(0, n_tables):
            scan = table_tags[t][0]
            roi_type = table_tags[t][1]
            roi_label = table_tags[t][2]
            im_space = table_tags[t][3]
            modality = table_tags[t][4]

            # extract parameters for the current modality
            if modality == 'CTscan' and 'imParamCT' in im_params:
                im_params_mod = im_params['imParamCT']
            elif modality== 'MRscan' and 'imParamMR' in im_params:
                im_params_mod = im_params['imParamMR']
            elif modality == 'PTscan' and 'imParamPET' in im_params:
                im_params_mod = im_params['imParamPET']
            # extract name save of the used filter
            if 'filter_type' in im_params_mod:
                filter_type = im_params_mod['filter_type']
                if filter_type in im_params['imParamFilter'] and 'name_save' in im_params['imParamFilter'][filter_type]:
                    name_save = im_params['imParamFilter'][filter_type]['name_save'] + '_' + feature_name
                else:
                    name_save= feature_name
            else:
                name_save= feature_name
            
            # set up table name
            if name_save:
                name_table = 'radiomics__' + scan + \
                '(' + roi_type + ')__'  + name_save + '.npy'   
            else:
                name_table = 'radiomics__' + scan + \
                '(' + roi_type + ')__' + im_space + '.npy'

            # Start timer
            start = time()
            logging.info("\n --> Computing radiomics table: {name_table}...")

            # Wildcard used to look only in the parent folder (save path),
            # no need to recursively look into sub-folders using '**/'.
            wildcard = '*_' + scan + '(' + roi_type + ')*.json'

            # Create radiomics table
            radiomics_table_dict = MEDimage.utils.create_radiomics_table(
                MEDimage.utils.get_file_paths(self._path_save / f'features({roi_label})', wildcard),
                im_space, 
                log_file
            )
            radiomics_table_dict['Properties']['Description'] = name_table

            # Save radiomics table
            save_path = self._path_save / f'features({roi_label})' / name_table
            np.save(save_path, [radiomics_table_dict])

            # Create CSV table and Definitions
            MEDimage.utils.write_radiomics_csv(save_path)

            logging.info(f"DONE\n {time() - start}\n")

        return log_file
    
    def __batch_all_patients(self, im_params: Dict, skip_existing) -> None:
        """
        Create batches of scans to process and compute radiomics features for every single scan.

        Args: 
            im_params(Dict): Dict of the processing & computation parameters.
            skip_existing(bool) : True to skip the computation of the features for the scans that already have been computed.

        Returns:
            None
        """
        # create a batch for each roi type
        n_roi_types = len(self.roi_type_labels)
        for r in range(0, n_roi_types):
            roi_type = self.roi_types[r]
            roi_type_label = self.roi_type_labels[r]
            print(f'\n --> Computing features for the "{roi_type_label}" roi type ...', end = '')

            # READING CSV EXPERIMENT TABLE
            tabel_roi = pd.read_csv(self._path_csv / ('roiNames_' + roi_type_label + '.csv'))
            tabel_roi['under'] = '_'
            tabel_roi['dot'] = '.'
            tabel_roi['npy'] = '.npy'
            name_patients = (pd.Series(
                tabel_roi[['PatientID', 'under', 'under',
                        'ImagingScanName',
                        'dot',
                        'ImagingModality',
                        'npy']].fillna('').values.tolist()).str.join('')).tolist()
            tabel_roi = tabel_roi.drop(columns=['under', 'under', 'dot', 'npy'])
            roi_names = tabel_roi.ROIname.tolist()

            # INITIALIZATION
            os.chdir(self._path_save)
            name_bacth_log = 'batchLog_' + roi_type_label
            p = Path.cwd().glob('*')
            files = [x for x in p if x.is_dir()]
            n_files = len(files)
            exist_file = name_bacth_log in [x.name for x in files]
            if exist_file and (n_files > 0):
                for i in range(0, n_files):
                    if (files[i].name == name_bacth_log):
                        mod_timestamp = datetime.fromtimestamp(
                            Path(files[i]).stat().st_mtime)
                        date = mod_timestamp.strftime("%d-%b-%Y_%HH%MM%SS")
                        new_name = name_bacth_log+'_'+date
                        if sys.platform == 'win32':
                            os.system('move ' + name_bacth_log + ' ' + new_name)
                        else:
                            os.system('mv ' + name_bacth_log + ' ' + new_name)

            os.makedirs(name_bacth_log, 0o777, True)
            path_batch = Path.cwd() / name_bacth_log

            # PRODUCE BATCH COMPUTATIONS
            n_patients = len(name_patients)

            # Produce a list log_file path.
            log_files = [path_batch / ('log_file_' + str(i) + '.log') for i in range(n_patients)]

            # Features computation for each patient (patients loop)
            for i in trange(n_patients):
                self.__compute_radiomics_one_patient(
                    name_patients[i],
                    roi_names[i], 
                    im_params,
                    roi_type,
                    roi_type_label,
                    log_files[i],
                    skip_existing
                )

            print('DONE')

    def __batch_all_tables(self, im_params: Dict):
        """
        Create batches of tables of the extracted features for every imaging scan type (CT, PET...).

        Args: 
            im_params(Dict): Dictionary of parameters.

        Returns:
            None
        """
        # INITIALIZATION
        os.chdir(self._path_save)
        name_batch_log = 'batchLog_tables'
        p = Path.cwd().glob('*')
        files = [x for x in p if x.is_dir()]
        n_files = len(files)
        exist_file = name_batch_log in [x.name for x in files]
        if exist_file and (n_files > 0):
            for i in range(0, n_files):
                if files[i].name == name_batch_log:
                    mod_timestamp = datetime.fromtimestamp(
                        Path(files[i]).stat().st_mtime)
                    date = mod_timestamp.strftime("%d-%b-%Y_%H:%M:%S")
                    new_name = name_batch_log+'_'+date
                    if sys.platform == 'win32':
                        os.system('move ' + name_batch_log + ' ' + new_name)
                    else:
                        os.system('mv ' + name_batch_log + ' ' + new_name)

        os.makedirs(name_batch_log, 0o777, True)
        path_batch = Path.cwd()
        
        # GETTING COMBINATIONS OF scan, roi_type and imageSpaces
        n_roi_types = len(self.roi_type_labels)
        
        # Get all scan names present for the given roi_type_label
        for f_idx in range(0, len(self.glcm_features)):
            # RE-INITIALIZATION
            table_tags = []
            for r in range(0, n_roi_types):
                label = self.roi_type_labels[r]
                wildcard = '*' + label + '*.json'
                roi_type = self.roi_types[r] + '_' + self.glcm_features[f_idx]
                file_paths = MEDimage.utils.get_file_paths(self._path_save / f'features({roi_type})', wildcard)
                n_files = len(file_paths)
                scans = [0] * n_files
                modalities = [0] * n_files
                for f in range(0, n_files):
                    rad_file_name = file_paths[f].stem
                    scans[f] = MEDimage.utils.get_scan_name_from_rad_name(rad_file_name)
                    modalities[f] = rad_file_name.split('.')[1]
                scans = s = (np.unique(np.array(scans))).tolist()
                n_scans = len(scans)
                # Get all scan names present for the given roi_type_label and scans
                for s in range(0, n_scans):
                    scan = scans[s]
                    modality = modalities[s]
                    wildcard = '*' + scan + '(' + label + ')*.json'
                    file_paths = MEDimage.utils.get_file_paths(self._path_save / f'features({roi_type})', wildcard)
                    n_files = len(file_paths)

                    # Finding the images spaces for a test file (assuming that all
                    # files for a given scan and roi_type_label have the same image spaces
                    radiomics = MEDimage.utils.json_utils.load_json(file_paths[0])
                    im_spaces = [key for key in radiomics.keys()]
                    im_spaces = im_spaces[:-1]
                    n_im_spaces = len(im_spaces)
                    # Constructing the table_tags variable
                    for i in range(0, n_im_spaces):
                        im_space = im_spaces[i]
                        table_tags = table_tags + [[scan, label, roi_type, im_space, modality]]

            # PRODUCE BATCH COMPUTATIONS
            n_tables = len(table_tags)
            self.n_bacth = self.n_bacth
            if self.n_bacth is None or self.n_bacth < 0:
                self.n_bacth = 1
            elif n_tables < self.n_bacth:
                self.n_bacth = n_tables

            # Produce a list log_file path.
            log_files = [path_batch / ('log_file_' + str(i) + '.txt') for i in range(self.n_bacth)]

            # Initialize ray
            if ray.is_initialized():
                ray.shutdown()

            ray.init(local_mode=True, include_dashboard=True, num_cpus=self.n_bacth)

            # Distribute the first tasks to all workers
            ids = [self.__compute_radiomics_tables.remote(
                                    self, 
                                    [table_tags[i]], 
                                    log_files[i],
                                    im_params,
                                    self.glcm_features[f_idx])
                    for i in range(self.n_bacth)]

            nb_job_left = n_tables - self.n_bacth

            for _ in trange(n_tables):
                ready, not_ready = ray.wait(ids, num_returns=1)
                ids = not_ready

                # We verify if error has occur during the process
                log_file = ray.get(ready)[0]

                # Distribute the remaining tasks
                if nb_job_left > 0:
                    idx = n_tables - nb_job_left
                    ids.extend([self.__compute_radiomics_tables.remote(
                                    self,
                                    [table_tags[idx]], 
                                    log_file,
                                    im_params,
                                    self.glcm_features[f_idx])])
                    nb_job_left -= 1

        print('DONE')

    def compute_radiomics(self, create_tables: bool = True, skip_existing: bool = False) -> None:
        """Compute all radiomic features for all scans in the CSV file (set in initialization) and organize it
        in JSON and CSV files

        Args:
            create_tables(bool) : True to create CSV tables for the extracted features and not save it in JSON only.
            skip_existing(bool) : True to skip the computation of the features for the scans that already have been computed.
        
        Returns:
            None.
        """

        # Load and process computing parameters
        im_params = self.__load_and_process_params()

        # Batch all scans from CSV file and compute radiomics for each scan
        self.__batch_all_patients(im_params, skip_existing)

        # Create a CSV file off of the computed features for all the scans
        if create_tables:
            self.__batch_all_tables(im_params)
