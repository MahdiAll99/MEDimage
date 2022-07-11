import logging
import os
import pickle
import sys
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from time import time
from typing import Dict, Union

import MEDimage
import numpy as np
import pandas as pd
import ray
from tqdm import trange


class BatchExtractor(object):
    """
    Organizes all the pateients/scans in batches to extract all the radiomic feautres
    """

    def __init__(
            self, 
            path_read: Union[str, Path],
            path_csv: Union[str, Path],
            path_params: Union[str, Path],
            path_save: Union[str, Path] = None,
            n_batch: int = 4
    ) -> None:
        """
        constructor of the BatchExtractor class 
        """
        self._path_csv = Path(path_csv)
        self._path_params = Path(path_params)
        self._path_read = Path(path_read)
        self._path_save = Path(path_save) if path_save is not None else None
        self.roi_types = []
        self.roi_type_labels = []
        self.n_bacth = n_batch

    def __load_and_process_params(self) -> Dict:
        """Load and process the computing & batch parameters from JSON file"""
        # Load json parameters
        im_params = MEDimage.utils.json_utils.load_json(self._path_params)
        
        # Update class aattributes
        self.roi_types.extend(im_params['roi_types'])
        self.roi_type_labels.extend(im_params['roi_type_labels'])
        self.n_bacth = im_params['n_batch'] if 'n_batch' in im_params else self.n_bacth

        # MRI parameters
        if(im_params['imParamMR']['reSeg']['range'] and im_params['imParamMR']['reSeg']['range'][1] == "inf"):
            im_params['imParamMR']['reSeg']['range'][1] = np.inf

        # CT parameters
        if(im_params['imParamCT']['reSeg']['range'] and im_params['imParamCT']['reSeg']['range'][1] == "inf"):
            im_params['imParamCT']['reSeg']['range'][1] = np.inf

        # PET parameters
        if(im_params['imParamPET']['reSeg']['range'] and im_params['imParamPET']['reSeg']['range'][1] == "inf"):
            im_params['imParamPET']['reSeg']['range'][1] = np.inf

        return im_params

    @ray.remote
    def __compute_radiomics_one_patient(
            self,
            name_patient,
            roi_name,
            im_params,
            roi_type,
            roi_type_label,
            log_file
        ) -> None:
        """
        Computes all radiomic features for one patient/scan
        """
        # Setting up logging settings
        logging.basicConfig(filename=log_file, level=logging.DEBUG)

        # start timer
        t_start = time()

        # Initialization
        message = f"\n***************** COMPUTING FEATURES: {name_patient} *****************"
        logging.info(message)

        # Load MEDimage instance
        try:
            with open(self._path_read / name_patient, 'rb') as f: MEDimg = pickle.load(f)
            MEDimg = MEDimage.MEDimage(MEDimg)
        except Exception as e:
            logging.error(f"\n ERROR LOADING PATIENT {name_patient}:\n {e}")
            return None

        # Init processing & computation parameters
        MEDimg.init_params(im_params)
        MEDimg.params.process.box_string = "full"

        # Get ROI (region of interest)
        logging.info("\n--> Extraction of ROI mask:")
        vol_obj_init, roi_obj_init = MEDimage.processing.get_roi_from_indexes(
            MEDimg,
            name_roi=roi_name,
            box_string=MEDimg.params.process.box_string
        )

        start = time()
        message = '--> Non-texture features pre-processing (interp + reSeg) for "Scale={}"'.\
            format(str(MEDimg.params.process.scale_non_text[0]))
        logging.info(message)

        # Interpolation
        # Intensity Mask
        vol_obj = MEDimage.processing.interp_volume(
            MEDimage=MEDimg,
            vol_obj_s=vol_obj_init,
            vox_dim=MEDimg.params.process.scale_non_text,
            interp_met=MEDimg.params.process.vol_interp,
            round_val=MEDimg.params.process.gl_round,
            image_type='image',
            roi_obj_s=roi_obj_init,
            box_string=MEDimg.params.process.box_string
        )
        # Morphological Mask
        roi_obj_morph = MEDimage.processing.interp_volume(
            MEDimage=MEDimg,
            vol_obj_s=roi_obj_init,
            vox_dim=MEDimg.params.process.scale_non_text,
            interp_met=MEDimg.params.process.roi_interp,
            round_val=MEDimg.params.process.roi_pv,
            image_type='roi', 
            roi_obj_s=roi_obj_init,
            box_string=MEDimg.params.process.box_string
        )

        # Re-segmentation
        # Intensity mask range re-segmentation
        roi_obj_int = deepcopy(roi_obj_morph)
        roi_obj_int.data = MEDimage.processing.range_re_seg(
            vol=vol_obj.data, 
            roi=roi_obj_int.data,
            im_range=MEDimg.params.process.im_range
        )
        # Intensity mask outlier re-segmentation
        roi_obj_int.data = np.logical_and(
            MEDimage.processing.outlier_re_seg(
                vol=vol_obj.data, 
                roi=roi_obj_int.data, 
                outliers=MEDimg.params.process.outliers
            ),
            roi_obj_int.data
        ).astype(int)
        logging.info("{}\n".format(time() - start))

        # Reset timer
        start = time()

        # Preparation of computation :
        MEDimg.init_ntf_calculation(vol_obj)

        # Image filtering
        if MEDimg.params.process.filter:
            vol_obj = MEDimage.filter.apply_mean(MEDimg, vol_obj)

        # ROI Extraction :
        vol_int_re = MEDimage.processing.roi_extract(
            vol=vol_obj.data, 
            roi=roi_obj_int.data
        )

        # Computation of non-texture features
        logging.info("--> Computation of non-texture features:")

        # Morphological features extraction
        morph = MEDimage.biomarkers.morph.extract_all(
            vol=vol_obj.data, 
            mask_int=roi_obj_int.data, 
            mask_morph=roi_obj_morph.data,
            res=MEDimg.params.process.scale_non_text,
            intensity=MEDimg.params.process.intensity
        )

        # Local intensity features extraction
        local_intensity = MEDimage.biomarkers.local_intensity.extract_all(
            img_obj=vol_obj.data,
            roi_obj=roi_obj_int.data,
            res=MEDimg.params.process.scale_non_text,
            intensity=MEDimg.params.process.intensity
        )

        # statistical features extraction
        stats = MEDimage.biomarkers.stats.extract_all(
            vol=vol_int_re,
            intensity=MEDimg.params.process.intensity
        )

        # Intensity histogram equalisation of the imaging volume
        vol_quant_re, _ = MEDimage.processing.discretisation(
            vol_re=vol_int_re,
            discr_type=MEDimg.params.process.ih['type'], 
            nq=MEDimg.params.process.ih['val'], 
            user_set_min_val=MEDimg.params.process.user_set_min_value
        )
        
        # Intensity histogram feratures extraction
        int_hist = MEDimage.biomarkers.intensity_histogram.extract_all(
            vol=vol_quant_re
        )
        
        # Intensity histogram equalisation of the imaging volume
        if 'type' in MEDimg.params.process.ivh and 'val' in MEDimg.params.process.ivh and MEDimg.params.process.ivh:
            vol_quant_re, wd = MEDimage.processing.discretisation(
                    vol_re=vol_int_re,
                    discr_type=MEDimg.params.process.ivh['type'], 
                    nq=MEDimg.params.process.ivh['val'], 
                    user_set_min_val=MEDimg.params.process.user_set_min_value,
                    ivh=True
            )
        else:
            vol_quant_re = vol_int_re
            wd = 1

        # Intensity volume histogram feratures extraction
        int_vol_hist = MEDimage.biomarkers.int_vol_hist.extract_all(
                    MEDimg=MEDimg,
                    vol=vol_quant_re,
                    vol_int_re=vol_int_re, 
                    wd=wd
        )

        # End of Non-Texture features extraction
        logging.info("{}\n".format(time() - start))

        # Computation of texture features
        logging.info("--> Computation of texture features:")

        # Compute radiomics features for each scale text
        count = 0
        for s in range(MEDimg.params.process.n_scale):
            start = time()
            message = '--> Texture features: pre-processing (interp + ' \
                    f'reSeg) for "Scale={str(MEDimg.params.process.scale_text[s][0])}": '
            logging.info(message)

            # Interpolation
            # Intensity Mask
            vol_obj = MEDimage.processing.interp_volume(
                MEDimage=MEDimg,
                vol_obj_s=vol_obj_init,
                vox_dim=MEDimg.params.process.scale_text[s],
                interp_met=MEDimg.params.process.vol_interp,
                round_val=MEDimg.params.process.gl_round,
                image_type='image', 
                roi_obj_s=roi_obj_init,
                box_string=MEDimg.params.process.box_string
            )
            # Morphological Mask
            roi_obj_morph = MEDimage.processing.interp_volume(
                MEDimage=MEDimg,
                vol_obj_s=roi_obj_init,
                vox_dim=MEDimg.params.process.scale_text[s],
                interp_met=MEDimg.params.process.roi_interp,
                round_val=MEDimg.params.process.roi_pv, 
                image_type='roi', 
                roi_obj_s=roi_obj_init,
                box_string=MEDimg.params.process.box_string
            )

            # Re-segmentation
            # Intensity mask range re-segmentation
            roi_obj_int = deepcopy(roi_obj_morph)
            roi_obj_int.data = MEDimage.processing.range_re_seg(
                vol=vol_obj.data, 
                roi=roi_obj_int.data,
                im_range=MEDimg.params.process.im_range
            )
            # Outlier Re-Segmentation
            roi_obj_int.data = np.logical_and(
                MEDimage.processing.outlier_re_seg(
                    vol=vol_obj.data, 
                    roi=roi_obj_int.data, 
                    outliers=MEDimg.params.process.outliers
                ),
                roi_obj_int.data
            ).astype(int)

            # Image filtering
            if MEDimg.params.process.filter:
                vol_obj = MEDimage.filter.apply_filter(MEDimg, vol_obj)

            logging.info("{}\n".format(time() - start))
            
            # Compute features for each discretisation algorithm and for each grey-level  
            for a, n in product(range(MEDimg.params.process.n_algo), range(MEDimg.params.process.n_gl)):
                count += 1 
                start = time()
                message = '--> Computation of texture features in image ' \
                        'space for "Scale= {}", "Algo={}", "GL={}" ({}):'.format(
                            str(MEDimg.params.process.scale_text[s][1]),
                            MEDimg.params.process.algo[a],
                            str(MEDimg.params.process.gray_levels[a][n]),
                            str(count) + '/' + str(MEDimg.params.process.n_exp)
                            )
                logging.info(message)

                # Preparation of computation :
                MEDimg.init_tf_calculation(algo=a, gl=n, scale=s)

                # ROI Extraction :
                vol_int_re = MEDimage.processing.roi_extract(
                    vol=vol_obj.data, 
                    roi=roi_obj_int.data)

                # Discretisation :
                vol_quant_re, _ = MEDimage.processing.discretisation(
                    vol_re=vol_int_re,
                    discr_type=MEDimg.params.process.algo[a], 
                    nq=MEDimg.params.process.gray_levels[a][n], 
                    user_set_min_val=MEDimg.params.process.user_set_min_value
                )

                # GLCM features extraction
                glcm = MEDimage.biomarkers.glcm.extract_all(
                    vol=vol_quant_re, 
                    distCorrection=MEDimg.params.radiomics.glcm.dist_correction)

                # GLRLM features extraction
                glrlm = MEDimage.biomarkers.glrlm.extract_all(
                    vol=vol_quant_re,
                    distCorrection=MEDimg.params.radiomics.glrlm.dist_correction)

                # GLSZM features extraction
                glszm = MEDimage.biomarkers.glszm.extract_all(
                    vol=vol_quant_re)

                # GLDZM features extraction
                gldzm = MEDimage.biomarkers.gldzm.extract_all(
                    vol_int=vol_quant_re, 
                    mask_morph=roi_obj_morph.data)

                # NGTDM features extraction
                ngtdm = MEDimage.biomarkers.ngtdm.extract_all(
                    vol=vol_quant_re, 
                    distCorrection=MEDimg.params.radiomics.ngtdm.distance_norm)

                # NGLDM features extraction
                ngldm = MEDimage.biomarkers.ngldm.extract_all(
                    vol=vol_quant_re)
                
                # Update radiomics results class
                MEDimg.update_radiomics(
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
                logging.info("{}\n".format(time() - start))

                # Saving radiomics results
                if self._path_save:
                    MEDimg.save_radiomics(
                                    scan_file_name=name_patient,
                                    path_save=self._path_save,
                                    roi_type=roi_type,
                                    roi_type_label=roi_type_label,
                                )
                
                logging.info("TOTAL TIME: {} seconds\n\n".format(time() - t_start))

                return  log_file

    def compute_radiomics(self) -> None:
        """
        Compute Radiomics_batchAllPatients.
        """
        # Initialize ray
        ray.init(local_mode=True, include_dashboard=True)

        # Load and process computing parameters
        im_params = self.__load_and_process_params()

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
            name_bacth_log = 'batchLog_' + roi_type_label
            p = Path.cwd().glob('*')
            files = [x for x in p if x.is_dir()]
            nfiles = len(files)
            exist_file = name_bacth_log in [x.name for x in files]
            if exist_file and (nfiles > 0):
                for i in range(0, nfiles):
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
            n_batch = self.n_bacth
            if n_batch is None or n_batch < 0:
                n_batch = 1
            elif n_patients < n_batch:
                n_batch = n_patients

            # Produce a list log_file path.
            log_files = [path_batch / ('log_file_' + str(i) + '.log')
                        for i in range(n_batch)]

            # Distribute the first tasks to all workers
            ids = [self.__compute_radiomics_one_patient.remote(
                        self,
                        name_patient=name_patients[i],
                        roi_name=roi_names[i], 
                        im_params=im_params,
                        roi_type=roi_type,
                        roi_type_label=roi_type_label,
                        log_file=log_files[i])
            for i in range(n_batch)]

            # Distribute the remaining tasks
            nb_job_left = n_patients - n_batch
            for _ in trange(n_patients):
                ready, not_ready = ray.wait(ids, num_returns=1)
                ids = not_ready
                log_file = ray.get(ready)[0]
                if nb_job_left > 0:
                    idx = n_patients - nb_job_left
                    ids.extend([self.__compute_radiomics_one_patient.remote(
                                    self,
                                    name_patients[idx],
                                    roi_names[idx], 
                                    im_params,
                                    roi_type,
                                    roi_type_label,
                                    log_file)
                                ])
                    nb_job_left -= 1

            print('DONE')
    
    def batch_all_tables():
        raise NotImplementedError