import logging
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numpyencoder import NumpyEncoder
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from MEDimage.learning.DataCleaner import DataCleaner
from MEDimage.learning.DesignExperiment import DesignExperiment
from MEDimage.learning.FSR import FSR
from MEDimage.learning.ml_utils import (average_results, combine_rad_tables,
                                        finalize_rad_table, find_best_model,
                                        get_ml_test_table, get_radiomics_table,
                                        intersect, intersect_var_tables,
                                        save_model)
from MEDimage.learning.Normalization import Normalization
from MEDimage.learning.Results import Results

from ..utils.json_utils import load_json, save_json


class RadiomicsLearner:
    def __init__(self, path_study: Path, path_settings: Path, experiment_label: str) -> None:
        """
        Constructor of the class DesignExperiment.

        Args:
            path_study (Path): Path to the main study folder where the outcomes, 
                learning patients and holdout patients dictionaries are found.
            path_settings (Path): Path to the settings folder.
            experiment_label (str): String specifying the label to attach to a given learning experiment in 
                "path_experiments". This label will be attached to the ml__$experiments_label$.json file as well
                as the learn__$experiment_label$ folder. This label is used to keep track of different experiments 
                with different settings (e.g. radiomics, scans, machine learning algorithms, etc.).
        
        Returns:
            None
        """
        self.path_study = Path(path_study)
        self.path_settings = Path(path_settings)
        self.experiment_label = experiment_label
    
    def __load_ml_info(self, ml_dict_paths: Dict) -> Dict:
        """
        Initializes the test dictionary information (training patients, test patients, ML dict, etc).

        Args:
            ml_dict_paths (Dict): Dictionary containing the paths to the different files needed 
                to run the machine learning experiment.
        
        Returns:
            dict: Dictionary containing the information of the machine learning test.
        """
        ml_dict = dict()

        # Training and test patients
        ml_dict['patientsTrain'] = load_json(ml_dict_paths['patientsTrain'])
        ml_dict['patientsTest'] = load_json(ml_dict_paths['patientsTest'])

        # Outcome table for training and test patients
        outcome_table = pd.read_csv(ml_dict_paths['outcomes'], index_col=0)
        ml_dict['outcome_table_binary'] = outcome_table.iloc[:, [0]]
        if outcome_table.shape[1] == 2:
            ml_dict['outcome_table_time'] = outcome_table.iloc[:, [1]]
        
        # Machine learning dictionary
        ml_dict['ml'] = load_json(ml_dict_paths['ml'])
        ml_dict['path_results'] = ml_dict_paths['results']

        return ml_dict

    def __find_balanced_threshold(
            self, 
            model: XGBClassifier, 
            variable_table: pd.DataFrame, 
            outcome_table_binary: pd.DataFrame
        ) -> float:
        """
        Finds the balanced threshold for the given machine learning test.

        Args:
            model (XGBClassifier): Trained XGBoost classifier for the given machine learning run.
            variable_table (pd.DataFrame): Radiomics table.
            outcome_table_binary (pd.DataFrame): Outcome table with binary labels.
        
        Returns:
            float: Balanced threshold for the given machine learning test.
        """

        # Getting the probability responses
        prob_xgb = self.predict_xgb(model, variable_table)

        # Calculating the ROC curve
        fpr, tpr, thresholds = metrics.roc_curve(outcome_table_binary.iloc[:, 0], prob_xgb)

        # Calculating the optimal threshold by minizing fpr (false positive rate) and maximizing tpr (true positive rate)
        minimum = np.argmin(np.power(fpr, 2) + np.power(1-tpr, 2))
        
        return thresholds[minimum]
    
    def get_hold_out_set_table(self, ml: Dict, var_id: str, patients_id: List):
        """
        Loads and pre-processes different radiomics tables then combines them to be used for hold-out testing.

        Args:
            ml (Dict): The machine learning dictionary containing the information of the machine learning test.
            var_id (str): String specifying the ID of the radiomics variable in ml.
                --> Ex: var1
            patients_id (List): List of patients of the hold-out set.

        Returns:
            pd.DataFrame: Radiomics table for the hold-out set.
        """
        # Loading and pre-processing
        rad_var_struct = ml['variables'][var_id]
        rad_tables_holdout = list()
        for item in rad_var_struct['path'].values():
            # Reading the table
            path_radiomics_csv = item['csv']
            path_radiomics_txt = item['txt']
            image_type = item['type']
            rad_table_holdout = get_radiomics_table(path_radiomics_csv, path_radiomics_txt, image_type, patients_id)
            rad_tables_holdout.append(rad_table_holdout)
        
        # Combine the tables
        rad_tables_holdout = combine_rad_tables(rad_tables_holdout)
        rad_tables_holdout.Properties['userData']['flags_processing'] = {}

        return rad_tables_holdout
    
    def pre_process_variables(self, ml: Dict, outcome_table_binary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads and pre-processes different radiomics tables from different variable types
        found in the ml dict.
        
        Note: 
            only patients of the training/learning set should be found in this outcome table.

        Args:
            ml (Dict): The machine learning dictionary containing the information of the machine learning test.
            outcome_table_binary (pd.DataFrame): outcome table with binary labels. This table may be used to
                pre-process some variables with the "FDA" feature set reduction algorithm.

        Returns:
            Tuple: Two dict of processed radiomics tables, one dict for training and one for 
                testing (no feature set reduction). 
        """
        # Get a list of unique varaibles found in the ml variables combinations dict
        variables_id = [s.split('_') for s in ml['variables']['combinations']]
        variables_id = list(set([x for sublist in variables_id for x in sublist]))

        # For each variable, load the corresponding radiomics table and pre-process it
        processed_var_tables, processed_var_tables_test =  {var_id : self.pre_process_radiomics_table(
            ml, 
            var_id, 
            outcome_table_binary
        ) for var_id in variables_id}
        
        return processed_var_tables, processed_var_tables_test

    def pre_process_radiomics_table(
            self, 
            ml: Dict, 
            var_id: str, 
            outcome_table_binary: pd.DataFrame,
            patients_train: list
        ) -> Tuple[Dict, Dict]:
        """
        For the given varaible, this function loads the corresponding radiomics tables and pre-processes them
        (cleaning, normalization and feature set reduction).

        Note: 
            Only patients of the training/learning set should be found in the given outcome table.
        
        Args:
            ml (Dict): The machine learning dictionary containing the information of the machine learning test 
                (parameters, options, etc.).
            var_id (str): String specifying the ID of the radiomics variable in ml. For example: 'var1'.
            outcome_table_binary (pd.DataFrame): outcome table with binary labels. This table may
                be used to pre-process some variables with the "FDA" feature set reduction algorithm.
            
            patients_train (list): List of patients to use for training.
        
        Returns:
            Tuple: Two dict of processed radiomics tables, one dict for training and one for 
                testing (no feature set reduction).
        """
        # Initialization
        patient_ids = list(outcome_table_binary.index)
        outcome_table_binary_training = outcome_table_binary.loc[patients_train]
        var_names = ['var_datacleaning', 'var_normalization', 'var_fSetReduction']
        flags_preprocessing =  {key: key in ml['variables'][var_id].keys() for key in var_names}

        # Pre-processing
        rad_var_struct = ml['variables'][var_id]
        rad_tables_training = list()
        rad_tables_testing = list()
        for item in rad_var_struct['path'].values():
            # Loading the table
            path_radiomics_csv = item['csv']
            path_radiomics_txt = item['txt']
            image_type = item['type']
            rad_table_training = get_radiomics_table(path_radiomics_csv, path_radiomics_txt, image_type, patients_train)
            rad_table_testing = get_radiomics_table(path_radiomics_csv, path_radiomics_txt, image_type, patient_ids)

            # Data cleaning
            if flags_preprocessing['var_datacleaning']:
                cleaning_dict = ml['datacleaning'][ml['variables'][var_id]['var_datacleaning']]['feature']['continuous']
                # Training data
                data_cleaner = DataCleaner(rad_table_training)
                rad_table_training = data_cleaner.apply_data_cleaning(cleaning_dict)
                # Testing data
                data_cleaner = DataCleaner(rad_table_testing)
                rad_table_testing = data_cleaner.apply_data_cleaning(cleaning_dict)

            # Normalization (ComBat)
            if flags_preprocessing['var_normalization']:
                normalization_method = ml['variables'][var_id]['var_normalization']
                # Some information must be stored to re-apply combat for testing data
                if 'combat' in normalization_method.lower():
                    # Training data
                    rad_table_training.Properties['userData']['normalization'] = dict()
                    rad_table_training.Properties['userData']['normalization']['original_data'] = dict()
                    rad_table_training.Properties['userData']['normalization']['original_data']['path_radiomics_csv'] = path_radiomics_csv
                    rad_table_training.Properties['userData']['normalization']['original_data']['path_radiomics_txt'] = path_radiomics_txt
                    rad_table_training.Properties['userData']['normalization']['original_data']['image_type'] = image_type
                    rad_table_training.Properties['userData']['normalization']['original_data']['patient_ids'] = patient_ids
                    if flags_preprocessing['var_datacleaning']:
                        data_cln_method = ml['variables'][var_id]['var_datacleaning']
                        rad_table_training.Properties['userData']['normalization']['original_data']['datacleaning_method'] = data_cln_method
                    
                    # Apply ComBat
                    normalization = Normalization('combat')
                    rad_table_training = normalization.apply_combat(variable_table=rad_table_training)  # Training data

                    # Testing data
                    rad_table_testing.Properties['userData']['normalization'] = dict()
                    rad_table_testing.Properties['userData']['normalization']['original_data'] = dict()
                    rad_table_testing.Properties['userData']['normalization']['original_data']['path_radiomics_csv'] = path_radiomics_csv
                    rad_table_testing.Properties['userData']['normalization']['original_data']['path_radiomics_txt'] = path_radiomics_txt
                    rad_table_testing.Properties['userData']['normalization']['original_data']['image_type'] = image_type
                    rad_table_testing.Properties['userData']['normalization']['original_data']['patient_ids'] = patient_ids
                    if flags_preprocessing['var_datacleaning']:
                        data_cln_method = ml['variables'][var_id]['var_datacleaning']
                        rad_table_testing.Properties['userData']['normalization']['original_data']['datacleaning_method'] = data_cln_method
                    
                    # Apply ComBat
                    normalization = Normalization('combat')
                    rad_table_testing = normalization.apply_combat(variable_table=rad_table_testing)  # Testing data
                    
                else:
                    raise NotImplementedError(f'Normalization method: {normalization_method} not recognized.')

            # Save the table
            rad_tables_training.append(rad_table_training)
            rad_tables_testing.append(rad_table_testing)

        # Feature set reduction (for training data only)
        if flags_preprocessing['var_fSetReduction']:
            f_set_reduction_method = ml['variables'][var_id]['var_fSetReduction']['method']
            if f_set_reduction_method.lower() == 'fda':
                fsr = FSR(f_set_reduction_method)
            else:
                raise NotImplementedError(f'Feature set reduction method: {f_set_reduction_method} not recognized.')
            
            # Apply FDA
            rad_tables_training = fsr.apply_fsr(ml, rad_tables_training, outcome_table_binary_training)

        # Finalization steps
        rad_tables_training.Properties['userData']['flags_preprocessing'] = flags_preprocessing

        return rad_tables_training, rad_tables_testing

    def train_xgboost_model(
            self, 
            var_table_train: pd.DataFrame,
            outcome_table_binary_train: pd.DataFrame,
            var_importance_threshold: float,
            optimal_threshold: float = None
        ) -> Dict:
        """
        Trains an XGBoost model for the given machine learning test.

        Args:
            var_table_train (pd.DataFrame): Radiomics table for the training/learning set.
            outcome_table_binary_train (pd.DataFrame): Outcome table with binary labels for the training/learning set.
            var_importance_threshold (float): Threshold for the variable importance. Variables with importance below
                this threshold will be removed from the model.
        
        Returns:
            Dict: Dictionary containing info about the trained XGBoost model.
        """
        
        # Safety check (make sure that the outcome table and the variable table have the same patients)
        var_table_train, outcome_table_binary_train = intersect_var_tables(var_table_train, outcome_table_binary_train)

        # Initial training to filter features using variable importance
        # XGB Classifier
        classifier = XGBClassifier()
        classifier.fit(var_table_train, outcome_table_binary_train)
        var_importance = classifier.feature_importances_
        var_table_train = var_table_train.iloc[:, var_importance >= var_importance_threshold]

        # Finalize the new radiomics table with the remaining variables
        var_table_train = finalize_rad_table(var_table_train)

        # Suggested scale_pos_weight
        scale_pos_weight = 1 - (outcome_table_binary_train == 0).sum().values[0] \
            / (outcome_table_binary_train == 1).sum().values[0]

        # XGB Classifier
        classifier = XGBClassifier(scale_pos_weight=scale_pos_weight)

        # Tune XGBoost parameters
        params = {
            'max_depth': [3, 4, 5], 
            'learning_rate': [0.1 , 0.01, 0.001], 
            'n_estimators': [50, 100, 200],
            'scale_pos_weight': [1, 25, 50, 75, 99, scale_pos_weight]
        }

        # Set up grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=classifier, 
            param_grid=params, 
            cv=5, 
            n_jobs=-1, 
            verbose=3, 
            scoring='matthews_corrcoef'
        )
        grid_search.fit(var_table_train, outcome_table_binary_train)

        # Get the best parameters
        best_params = grid_search.best_params_

        # Update scale_pos_weight
        best_params['scale_pos_weight'] = (outcome_table_binary_train == 0).sum().values[0] / (outcome_table_binary_train == 1).sum().values[0]

        # Fit the XGB Classifier with the best parameters
        classifier = XGBClassifier(**best_params)
        classifier.fit(var_table_train, outcome_table_binary_train)

        # Another work around
        #selection = SelectFromModel(classifier, threshold=var_importance_threshold, prefit=True)
        #var_table_train = selection.transform(var_table_train)
        
        # Saving the information of the model in a dictionary
        model_xgb = dict()
        model_xgb['algo'] = 'xgb'
        model_xgb['type'] = 'binary'
        if optimal_threshold:
            model_xgb['threshold'] = optimal_threshold
        else:
            try:
                model_xgb['threshold'] = self.__find_balanced_threshold(classifier, var_table_train, outcome_table_binary_train)
            except Exception as e:
                print('Error in finding optimal threshold, it will be set to 0.5:' + str(e))
                model_xgb['threshold'] = 0.5
        model_xgb['model'] = classifier
        model_xgb['var_names'] = list(var_table_train.columns.values)
        model_xgb['var_info'] = deepcopy(var_table_train.Properties['userData'])
        model_xgb['optimization'] = best_params

        return model_xgb
        
    def test_xgb_model(self, model_dict: Dict, variable_table: pd.DataFrame, patient_list: List) -> List:
        """
        Tests the XGBoost model for the given dataset patients.

        Args:
            model_dict (Dict): Dictionary containing info about the trained XGBoost model.
            variable_table (pd.DataFrame): Radiomics table for the test set (should not be normalized).
            patient_list (List): List of patients to test.
        
        Returns:
            List: List the model response for the training and test sets.
        """
        # Initialization
        n_test = len(patient_list)
        var_names = model_dict['var_names']
        var_def = model_dict['var_info']['variables']['var_def']
        model_response = list()

        # Preparing the variable table
        variable_table = get_ml_test_table(variable_table, var_names, var_def)

        # Test the model
        for i in range(n_test):
            # Get the patient IDs
            patient_ids = patient_list[i]

            # Getting predictions for each patient
            n_patients = len(patient_ids)
            varargout = np.zeros((n_patients, 1)) * np.nan  # NaN if the computation fails            
            for p in range(n_patients):
                try:
                    varargout[p] = self.predict_xgb(model_dict['model'], variable_table.loc[[patient_ids[p]], :])
                except Exception as e:
                    print('Error in computing prediction for patient ' + str(patient_ids[p]) + ': ' + str(e))
                    varargout[p] = np.nan

            # Save the predictions 
            model_response.append(varargout)

        return model_response
    
    def predict_xgb(self, xgb_model: XGBClassifier, variable_table: pd.DataFrame) -> float:
        """
        Computes the prediction of the XGBoost model for the given variable table.

        Args:
            xgb_model (XGBClassifier): XGBClassifier model.
            variable_table (pd.DataFrame): Variable table for the prediction.
        
        Returns:
            float: Prediction of the XGBoost model.
        """

        # Predictions
        predictions = xgb_model.predict_proba(variable_table)

        # Get the probability of the positive class
        predictions = predictions[:, 1][0]

        return predictions

    def ml_run(self, path_ml: Path, holdout_test: bool = True) -> None:
        """
        This function runs the machine learning test for the ceated experiment.

        Args:
            path_ml (Path): Path to the main dictionary containing info about the ml current experiment.
            holdout_test (bool, optional): Boolean specifying if the hold-out test should be performed.
        
        Returns:
            None.
        """
        # Set up logging file for the batch
        log_file = os.path.dirname(path_ml) + '/batch.log'
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s', filemode='w')

        # Start the timer
        batch_start = time.time()

        logging.info("\n\n********************MACHINE LEARNING RUN********************\n\n")

        # --> A. Initialization phase
        # Load the test dictionary and machine learning information
        ml_dict_paths = load_json(path_ml)      # Test information dictionary
        ml_info_dict = self.__load_ml_info(ml_dict_paths)       # Machine learning information dictionary

        # Machine learning assets
        patients_train = ml_info_dict['patientsTrain']
        patients_test = ml_info_dict['patientsTest']
        patients_holdout = load_json(self.path_study / 'patientsHoldOut.json') if holdout_test else None
        outcome_table_binary = ml_info_dict['outcome_table_binary']
        ml = ml_info_dict['ml']
        path_results = ml_info_dict['path_results']

        # Features processing flags
        var_names = ['datacleaning', 'fSetReduction', 'normalization', 'fSetSelection']
        flags_processing = {key: key in ml['settings'].keys() for key in var_names}

        # --> B. Machine Learning phase 
        # B.1. Pre-processing features
        start = time.time()
        logging.info("\n\n--> PRE-PROCESSING TRAINING VARIABLES")

        # Not all variables will be used to train the model, only the user-selected variable
        var_id = str(ml['variables']['varStudy'])

        # Pre-processing of the radiomics tables/variables
        processed_var_table, processed_var_tables_test = self.pre_process_radiomics_table(
            ml, 
            var_id, 
            outcome_table_binary.copy(),
            patients_train
        )
        logging.info(f"...Done in {time.time()-start} s")

        # B.2. Pre-learning initialization
        # Patient definitions (training and test sets)
        patient_ids = list(outcome_table_binary.index)
        patients_train = intersect(patient_ids, patients_train)
        patients_test = intersect(patient_ids, patients_test)
        patients_holdout = intersect(patient_ids, patients_holdout) if holdout_test else None

        # Initializing outcome tables for training and test sets
        outcome_table_binary_train = outcome_table_binary.loc[patients_train, :]
        outcome_table_binary_test = outcome_table_binary.loc[patients_test, :]
        outcome_table_binary_holdout = outcome_table_binary.loc[patients_holdout, :] if holdout_test else None

        # Serperate variable table for training sets (repetetive but double-checking)
        var_table_train = processed_var_table.loc[patients_train, :]

        # Initializing XGBoost model settings
        var_importance_threshold = ml['algorithms']['XGBoost']['varImportanceThreshold']
        optimal_threshold = ml['algorithms']['XGBoost']['optimalThreshold']

        # B.2. Training the XGBoost model
        tstart = time.time()
        logging.info(f"\n\n--> TRAINING XGBOOST MODEL FOR VARIABLE {var_id}")

        # Training the model
        model = self.train_xgboost_model(
            var_table_train, 
            outcome_table_binary_train, 
            var_importance_threshold, 
            optimal_threshold
        )

        # Saving the trained model using pickle
        name_save_model = ml['algorithms']['XGBoost']['nameSave']
        model_id = name_save_model + '_' + str(ml['variables']['varStudy'])
        path_model = os.path.dirname(path_results) + '/' + (model_id + '.pickle')
        model_dict = save_model(ml, model, str(ml['variables']['varStudy']), path_model)

        logging.info("{}--> DONE. TOTAL TIME OF LEARNING PROCESS: {:.2f} min".format(" " * 4, (time.time()-tstart) / 60))

        # --> C. Testing phase
        # C.1. Preparing test data
        var_table_all_test = combine_rad_tables(processed_var_tables_test)
        var_table_all_test.Properties['userData']['flags_processing'] = flags_processing

        # C.2. Testing the XGBoost model and computing model response
        tstart = time.time()
        logging.info(f"\n\n--> TESTING XGBOOST MODEL FOR VARIABLE {var_id}")

        response_train, response_test = self.test_xgb_model(
            model, 
            var_table_all_test, 
            [patients_train, patients_test]
        )
        
        logging.info('{}--> DONE. TOTAL TIME OF LEARNING PROCESS: {:.2f}'.format(" " * 4, (time.time() - tstart)/60))
        
        if holdout_test:
            # --> D. Holdoutset testing phase
            # D.1. Prepare holdout test data
            var_table_all_holdout = self.get_hold_out_set_table(ml, var_id, patients_holdout)

            # D.2. Testing the XGBoost model and computing model response on the holdout set
            tstart = time.time()
            logging.info(f"\n\n--> TESTING XGBOOST MODEL FOR VARIABLE {var_id} ON THE HOLDOUT SET")

            response_holdout = self.test_xgb_model(model, var_table_all_holdout, [patients_holdout])[0]
        
        logging.info('{}--> DONE. TOTAL TIME OF LEARNING PROCESS: {:.2f}'.format(" " * 4, (time.time() - tstart)/60))
        
        # E. Computing performance metrics
        tstart = time.time()

        # Initialize the Results class
        result = Results(model_dict, model_id)
        if holdout_test:
            run_results = result.to_json(
                response_train=response_train, 
                response_test=response_test,
                response_holdout=response_holdout, 
                patients_train=patients_train, 
                patients_test=patients_test, 
                patients_holdout=patients_holdout
            )
        else:
            run_results = result.to_json(
                response_train=response_train, 
                response_test=response_test,
                response_holdout=None, 
                patients_train=patients_train, 
                patients_test=patients_test, 
                patients_holdout=None
            )
        
        # Calculating performance metrics for training phase and saving the ROC curve
        run_results[model_id]['train']['metrics'] = result.get_model_performance(
            response_train, 
            outcome_table_binary_train,
            label='training'
        )
        
        # Calculating performance metrics for testing phase and saving the ROC curve
        run_results[model_id]['test']['metrics'] = result.get_model_performance(
            response_test, 
            outcome_table_binary_test,
            label='testing'
        )

        if holdout_test:
            # Calculating performance metrics for holdout phase and saving the ROC curve
            run_results[model_id]['holdout']['metrics'] = result.get_model_performance(
                response_holdout, 
                outcome_table_binary_holdout,
                label='holdout'
            )

        logging.info('\n\n--> COMPUTING PERFORMANCE METRICS ... Done in {:.2f} sec'.format(time.time()-tstart))
        
        # F. Saving the results dictionary
        save_json(path_results, run_results, cls=NumpyEncoder)

        # Total computing time
        logging.info("\n\n*********************************************************************")
        logging.info('{} TOTAL COMPUTATION TIME: {:.2f} hours'.format(" " * 13, (time.time()-batch_start)/3600))
        logging.info("*********************************************************************")
        
    def run_experiment(self, holdout_test: bool = True) -> None:
        """
        Run the machine learning experiment for each split/run
        """        
        # Initialize the DesignExperiment class
        experiment = DesignExperiment(self.path_study, self.path_settings, self.experiment_label)

        # Generate the machine learning experiment
        path_file_ml_paths = experiment.generate_experiment()

        # Run the different machine learning tests for the experiment
        tests_dict = load_json(path_file_ml_paths) # Tests dictionary
        for run in tests_dict.keys():
            self.ml_run(tests_dict[run], holdout_test)
        
        # Average results of the different splits/runs
        average_results(self.path_study / f'learn__{self.experiment_label}')
