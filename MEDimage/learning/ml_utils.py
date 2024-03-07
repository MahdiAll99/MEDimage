import csv
import json
import os
import pickle
import re
import string
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import seaborn as sns
from numpyencoder import NumpyEncoder
from sklearn.model_selection import StratifiedKFold

from MEDimage.utils import get_institutions_from_ids
from MEDimage.utils.get_full_rad_names import get_full_rad_names
from MEDimage.utils.json_utils import load_json, save_json


# Define useful constants
# Metrics to process
list_metrics = [
    'AUC', 'AUPRC', 'BAC', 'Sensitivity', 'Specificity',
    'Precision', 'NPV', 'F1_score', 'Accuracy', 'MCC',
    'TN', 'FP', 'FN', 'TP'
]

def average_results(path_results: Path, save: bool = False) -> None:
    """
    Averages the results (AUC, BAC, Sensitivity and Specifity) of all the runs of the same experiment,
    for training, testing and holdout sets.

    Args:
        path_results(Path): path to the folder containing the results of the experiment.
        save (bool, optional): If True, saves the results in the same folder as the model.
    
    Returns:
        None.
    """
    # Get all tests paths
    list_path_tests =  [path for path in path_results.iterdir() if path.is_dir()]

    # Initialize dictionaries
    results_avg = {
        'train': {},
        'test': {},
        'holdout': {}
    }

    # Metrics to process
    metrics = ['AUC', 'AUPRC', 'BAC', 'Sensitivity', 'Specificity',
            'Precision', 'NPV', 'F1_score', 'Accuracy', 'MCC',
            'TN', 'FP', 'FN', 'TP']

    # Process metrics
    for dataset in ['train', 'test', 'holdout']:
        dataset_dict = results_avg[dataset]
        for metric in metrics:
            metric_values = []
            for path_test in list_path_tests:
                results_dict = load_json(path_test / 'run_results.json')
                if dataset in results_dict[list(results_dict.keys())[0]].keys():
                    if 'metrics' in results_dict[list(results_dict.keys())[0]][dataset].keys():
                        metric_values.append(results_dict[list(results_dict.keys())[0]][dataset]['metrics'][metric])
                    else:
                        continue
                else:
                    continue

            # Fill the dictionary
            if metric_values:
                dataset_dict[f'{metric}_mean'] = np.nanmean(metric_values)
                dataset_dict[f'{metric}_std'] = np.nanstd(metric_values)
                dataset_dict[f'{metric}_max'] = np.nanmax(metric_values)
                dataset_dict[f'{metric}_min'] = np.nanmin(metric_values)
                dataset_dict[f'{metric}_2.5%'] = np.nanpercentile(metric_values, 2.5)
                dataset_dict[f'{metric}_97.5%'] = np.nanpercentile(metric_values, 97.5)

    # Save the results
    if save:
        save_json(path_results / 'results_avg.json', results_avg, cls=NumpyEncoder)
        return path_results / 'results_avg.json'

    return results_avg

def combine_rad_tables(rad_tables: List) -> pd.DataFrame:
    """
    Combines a list of radiomics tables into one single table.
    
    Args:
        rad_tables (List): List of radiomics tables.

    Returns:
        pd.DataFrame: Single combined radiomics table.
    """
    # Initialization
    n_tables = len(rad_tables)

    base_idx = 0
    for idx, table in enumerate(rad_tables):
        if not table.empty:
            base_idx = idx
            break
    # Finding patient intersection
    for t in range(n_tables):
        if rad_tables[t].shape[1] > 0 and t != base_idx:
           rad_tables[base_idx], rad_tables[t] = intersect_var_tables(rad_tables[base_idx], rad_tables[t])
    
    # Check for NaNs
    '''for table in rad_tables:
        assert(table.isna().sum().sum() == 0)'''

    # Initializing the radiomics table template
    radiomics_table = pd.DataFrame()
    radiomics_table.Properties = {}
    radiomics_table._metadata += ['Properties']
    radiomics_table.Properties['userData'] = {}
    radiomics_table.Properties['VariableNames'] = []
    radiomics_table.Properties['userData']['normalization'] = {}

    # Combining radiomics table one by one
    count = 0
    continuous = []
    str_names = '||'
    for t in range(n_tables):
        rad_table_id = 'radTab' + str(t+1)
        if rad_tables[t].shape[1] > 0 and rad_tables[t].shape[0] > 0:
            features = rad_tables[t].columns.values
            description = rad_tables[t].Properties['Description']
            full_rad_names = get_full_rad_names(rad_tables[t].Properties['userData']['variables']['var_def'], 
                                                features)
            if 'normalization' in rad_tables[t].Properties['userData']:
                radiomics_table.Properties['userData']['normalization'][rad_table_id] = rad_tables[t].Properties[
                                                                                    'userData']['normalization']
            for f, feature in enumerate(features):
                count += 1
                var_name = 'radVar' + str(count)
                radiomics_table[var_name] = rad_tables[t][feature]
                radiomics_table.Properties['VariableNames'].append(var_name)
                continuous.append(var_name)
                if description:
                    str_names += 'radVar' + str(count) + ':' + description + '___' + full_rad_names[f] + '||'
                else:
                    str_names += 'radVar' + str(count) + ':' + full_rad_names[f] + '||'

    # Updating the radiomics table properties
    radiomics_table.Properties['Description'] = ''
    radiomics_table.Properties['DimensionNames'] = ['PatientID']
    radiomics_table.Properties['userData']['variables'] = {}
    radiomics_table.Properties['userData']['variables']['var_def'] = str_names
    radiomics_table.Properties['userData']['variables']['continuous'] = continuous

    return radiomics_table

def combine_tables_from_list(var_list: List, combination: List) -> pd.DataFrame:
    """
    Concatenates all variable tables in ``var_list`` according to ``var_ids``.
    
    Unlike ``combine_rad_tables`` This method concatenates variable tables instead of creating a new table from
    the intersection of the tables.

    Args:
        var_list (List): List of tables. Each key is a given var_id and holds a radiomic table.
            --> Ex: .var1: variable table 1
                    .var2: variable table 2
                    .var3: variable table 3
        combination (list): List of strings to identify the table to combine in var_list.
            --> Ex: {'var1','var3'}

    Returns:
        pd.DataFrame: variable_table: Combined radiomics table.
    """
    def concatenate_varid(var_names, var_id):
        return np.asarray([var_id + "__" + var_name for var_name in var_names.tolist()])
    
    # Initialization
    variables = dict()
    variables['continuous'] = np.array([])
    variable_tables = list()

    # Using the first table as template
    var_id = combination[0]
    variable_table = deepcopy(var_list[var_id]) # first table from the list
    variable_table.Properties = deepcopy(var_list[var_id].Properties)
    new_columns = [var_id + '__' + col for col in variable_table.columns]
    variable_table.columns = new_columns
    variable_table.Properties['VariableNames'] = new_columns
    variable_table.Properties['userData'] = dict()  # Re-Initializing
    variable_table.Properties['userData'][var_id] = deepcopy(var_list[var_id].Properties['userData'])
    variables['continuous'] = np.concatenate((variables['continuous'], var_list[var_id].Properties[
                                                                        'userData']['variables']['continuous']))
    variable_tables.append(variable_table)

    # Concatenating all other tables
    for var_id in combination[1:]:
        variable_table.Properties['userData'][var_id] = var_list[var_id].Properties['userData']
        patient_ids = intersect(list(variable_table.index), (var_list[var_id].index))
        var_list[var_id] = var_list[var_id].loc[patient_ids]
        variable_table = variable_table.loc[patient_ids]
        old_columns = list(variable_table.columns)
        old_properties = deepcopy(variable_table.Properties)  # for unknown reason Properties are erased after concat
        variable_table = pd.concat([variable_table, var_list[var_id]], axis=1)
        variable_table.columns = old_columns + [var_id + "__" + col for col in var_list[var_id].columns]
        variable_table.Properties = old_properties
        variable_table.Properties['VariableNames'] = list(variable_table.columns)
        variables['continuous'] = np.concatenate((variables['continuous'], var_list[var_id].Properties['userData']['variables']['continuous']))

    # Updating the radiomics table properties
    variable_table.Properties['Description'] = "Data table"
    variables['continuous'] = concatenate_varid(variables['continuous'], var_id)
    variable_table.Properties['userData']['variables'] = variables

    return variable_table

def convert_comibnations_to_list(combinations_string: str) -> Tuple[List, List]:
    """
    Converts a cell of strings specifying variable ids combinations to
    a cell of cells of strings.
    
    Args:
        combinations_string (str): Cell of strings specifying var_ids combinations
            separated by underscores.
            --> Ex: {'var1_var2';'var2_var3';'var1_var2_var3'}

    Rerturs:
        - List: List of strings of the seperated var_ids.
            --> Ex: {{'var1','var2'};{'var2','var3'};{'var1','var2','var3'}}
        - List: List of strings specifying the "alphabetical" IDs of combined variables 
            in ``combinations``. var1 --> A, var2 -> B, etc.
            --> Ex: {'model_AB';'model_BC';'model_ABC'}
    """
    # Building combinations
    combinations = [s.split('_') for s in combinations_string]

    # Building model_ids
    alphabet = string.ascii_uppercase
    model_ids = list()
    for combination in combinations:
        model_ids.append('model_' + ''.join([alphabet[int(var[3:])-1] for var in combination]))
    
    return combinations, model_ids

def count_class_imbalance(path_csv_outcomes: Path) -> Dict:
    """
    Counts the class imbalance in a given outcome table.
    
    Args:
        path_csv_outcomes (Path): Path to the outcome table.

    Returns:
        Dict: Dictionary containing the count of each class.
    """
    # Initialization
    outcomes = pandas.read_csv(path_csv_outcomes, sep=',')
    outcomes.dropna(inplace=True)
    outcomes.reset_index(inplace=True, drop=True)
    name_outcome = outcomes.columns[-1]
    
    # Counting the percentage of each class
    class_0_perc = np.sum(outcomes[name_outcome] == 0) / len(outcomes)
    class_1_perc = np.sum(outcomes[name_outcome] == 1) / len(outcomes)

    return {'class_0': class_0_perc, 'class_1': class_1_perc}

def create_experiment_folder(path_outcome_folder: str, method: str = 'Random') -> str:
    """
    Creates the experiment folder where the hold-out splits will be saved and returns the path
    to the folder.
    
    Args:
        path_outcome_folder (str): Full path to the outcome folder (folder containing the outcome table etc).
        method (str): String specifying the split type. Default is 'Random'.
    
    Returns:
        str: Full path to the experiment folder.
    """

    # Creating the outcome folder if it does not exist
    if not os.path.isdir(path_outcome_folder):
        os.makedirs(path_outcome_folder)

    # Creating the experiment folder if it does not exist
    list_outcome = os.listdir(path_outcome_folder)
    if not list_outcome:
        flag_exist_split = False
    else:
        n_exist = 0
        flag_exist_split = False
        for i in range(len(list_outcome)):
            if 'holdOut__' + method + '__' in list_outcome[i]:
                n_exist = n_exist + 1
                flag_exist_split = True
    
    # If path experiment folder exists already, create a new one (sequentially)
    if not flag_exist_split:
        path_split = str(path_outcome_folder) + '/holdOut__' + method + '__001'
    else:
        path_split = str(path_outcome_folder) + '/holdOut__' + method + '__' + \
            str(n_exist+1).zfill(3)

    os.mkdir(path_split)
    return path_split

def create_holdout_set(
                path_outcome_file: Union[str, Path],
                outcome_name: str,
                path_save_experiments: Union[str, Path] = None,
                method: str = 'random',
                percentage: float = 0.2,
                n_split: int = 1,
                seed : int = 1) -> None:
    """
    Creates a hold-out patient set to be used for final independent testing after a final 
    model is chosen. All the information is saved in a JSON file.
    
    Args:
        path_outcome_file (str): Full path to where the outcome CSV file is stored.
        outcome_name (str): Name of the outcome. For example, 'OS' for overral survivor.
        path_save_experiments (str): Full path to the folder where the experiments
            will be saved.
        method (str): Method to use for creating the hold-out set. Options are:
            - 'random': Randomly selects patients for the hold-out set.
            - 'all_learn': No hold-out set is created. All patients are used for learning.
            - 'institution': TODO.
        percentage (float): Percentage of patients to use for the hold-out set. Default is 0.2.
        n_split (int): Number of splits to create. Default is 1.
        seed (int): Seed to use for the random split. Default is 1.
    
    Returns:
        None.
    """
    # Initilization
    outcome_name = outcome_name.upper()
    outcome_table = pandas.read_csv(path_outcome_file, sep=',')
    outcome_table.dropna(inplace=True)
    outcome_table.reset_index(inplace=True, drop=True)
    patient_ids = outcome_table['PatientID']

    # Creating experiment folders and patient test split(s)
    outcome_name = re.sub(r'\W', "", outcome_name)
    path_outcome = str(path_save_experiments) + '/' + outcome_name
    name_outcome_in_table_binary = outcome_name + '_binary'
    
    # Column names in the outcome table
    with open(path_outcome_file, 'r') as infile:
        reader = csv.DictReader(infile, delimiter=',')
        var_names = reader.fieldnames

    # Include time to event if it exists
    flag_time = False
    if(outcome_name + '_eventFreeTime' in str(var_names)):
        name_outcome_in_table_time = outcome_name + '_eventFreeTime'
        flag_time = True

    # Check if the outcome name for binary is correct
    if name_outcome_in_table_binary not in outcome_table.columns:
            name_outcome_in_table_binary = var_names[-1]
    
    # Run the split
    # Random
    if 'random' in method.lower():
        # Creating the experiment folder
        path_split = create_experiment_folder(path_outcome, 'random')

        # Getting the random split
        patients_learn_temp, patients_hold_out_temp = get_stratified_splits(
            outcome_table[['PatientID', name_outcome_in_table_binary]],
            n_split, percentage, seed, False)
        
        # Getting the patient IDs in the learning and hold-out sets
        if n_split > 1:
            patients_learn = np.empty((n_split, len(patients_learn_temp[0])), dtype=object)
            patients_hold_out = np.empty((n_split, len(patients_hold_out_temp[0])), dtype=object)
            for s in range(n_split):
                patients_learn[s] = patient_ids[patients_learn_temp[s]]
                patients_hold_out[s] = patient_ids[patients_hold_out_temp[s]]
        else:
            patients_learn = patient_ids[patients_learn_temp.values.tolist()]
            patients_learn.reset_index(inplace=True, drop=True)
            patients_hold_out = patient_ids[patients_hold_out_temp.values.tolist()]
            patients_hold_out.reset_index(inplace=True, drop=True)
    
    # All Learn
    elif 'all_learn' in method.lower():
        # Creating the experiment folder
        path_split = create_experiment_folder(path_outcome, 'all_learn')

        # Getting the split (all Learn so no hold out)
        patients_learn = patient_ids
        patients_hold_out = []
    else:
        raise ValueError('Method not recognized. Use "random" or "all_learn".')
    
    # Creating final outcome table and saving it
    if flag_time:
        outcomes = outcome_table[
            ['PatientID', name_outcome_in_table_binary, name_outcome_in_table_time]]
    else:
        outcomes = outcome_table[['PatientID', name_outcome_in_table_binary]]

    # Finalize the outcome table
    outcomes = outcomes.dropna(inplace=False)   # Drop NaNs
    outcomes.reset_index(inplace=True, drop=True)   # Reset index        

    # Save the outcome table
    paths_exp_outcomes = str(path_split + '/outcomes.csv')
    outcomes.to_csv(paths_exp_outcomes, index=False)

    # Save dict of patientsLearn
    paths_exp_patientsLearn = str(path_split) + '/patientsLearn.json'
    patients_learn.to_json(paths_exp_patientsLearn, orient='values', indent=4)

    # Save dict of patientsHoldOut
    if method == 'random':
        paths_exp_patients_hold_out = str(path_split) + '/patientsHoldOut.json'
        patients_hold_out.to_json(paths_exp_patients_hold_out, orient='values', indent=4)

        # Save dict of all the paths
        data={
            "outcomes" : paths_exp_outcomes,
            "patientsLearn": paths_exp_patientsLearn,
            "patientsHoldOut": paths_exp_patients_hold_out,
            "pathWORK": path_split
        }
    else:
        data={
            "outcomes" : paths_exp_outcomes,
            "patientsLearn": paths_exp_patientsLearn,
            "pathWORK": path_split
        }
    paths_exp = str(path_split + '/paths_exp.json')
    with open(paths_exp, 'w') as f:
        json.dump(data, f, indent=4)
    
    # Return the path to the experiment and path to split
    return path_split, paths_exp

def cross_validation_split(
        outcome: List[Union[int, float]], 
        n_splits: int = 5, 
        seed: int = None
    ) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Perform stratified cross-validation split.

    Args:
        outcome (list): Outcome variable (binary).
        n_splits (int, optional): Number of folds. Default is 5.
        seed (int or None, optional): Random seed for reproducibility. Default is None.

    Returns:
        train_indices_list (list of lists): List of training indices for each fold.
        test_indices_list (list of lists): List of testing indices for each fold.
    """

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_data_list = []
    test_data_list = []
    patient_ids = pd.Series(outcome.index)

    for train_indices, test_indices in skf.split(X=outcome, y=outcome):
        train_data_list.append(patient_ids[train_indices])
        test_data_list.append(patient_ids[test_indices])
    
    train_data_array = np.array(train_data_list)
    test_data_array = np.array(test_data_list)

    return train_data_array, test_data_array

def find_best_model(path_results: Path, metric: str = 'AUC', second_metric: str = 'AUC') -> Tuple[Dict, Path]:
    """
    Find the best model with the highest performance on the test set
    in a given path based on a given metric.

    Args:
        path_results (Path): Path to the results folder.
        metric (str): Metric to use to find the best model in case of a tie. Default is 'AUC'.

    Returns:
        Tuple[Dict, Path]: Tuple containing the best model result dict and the path to the best model.
    """
    list_metrics = [
        'AUC', 'Sensitivity', 'Specificity', 
        'BAC', 'AUPRC', 'Precision', 
        'NPV', 'Accuracy', 'F1_score', 'MCC',
        'TP', 'TN', 'FP', 'FN'
    ]
    assert metric in list_metrics, f'Given metric {metric} is not in the list of metrics. Please choose from {list_metrics}'

    # Get all tests paths
    list_path_tests =  [path for path in path_results.iterdir() if path.is_dir()]

    # Initialization
    metric_best = -1
    second_metric_best = -1
    path_result_best = None

    # Get all models and their metrics (AUC especially)
    for path_test in list_path_tests:
        if not (path_test / 'run_results.json').exists():
            continue
        results_dict = load_json(path_test / 'run_results.json')
        metric_test = results_dict[list(results_dict.keys())[0]]['test']['metrics'][metric]
        if metric_test > metric_best:
            metric_best = metric_test
            path_result_best = path_test
        elif metric_test == metric_best:
            second_metric_test = results_dict[list(results_dict.keys())[0]]['test']['metrics'][second_metric]
            if second_metric_test > second_metric_best:
                second_metric_best = second_metric_test
                path_result_best = path_test
    
    # Load best model result dict
    results_dict_best = load_json(path_result_best / 'run_results.json')

    # Load model
    model_name = list(results_dict_best.keys())[0]
    with open(path_result_best / f'{model_name}.pickle', 'rb') as file:
        model = pickle.load(file)
    
    return model, results_dict_best

def feature_imporance_analysis(path_results: Path):
    """
    Averages the results (AUC, BAC, Sensitivity and Specifity) of all the runs of the same experiment,
    for training, testing and holdout sets.

    Args:
        path_results(Path): path to the folder containing the results of the experiment.
        save (bool, optional): If True, saves the results in the same folder as the model.
    
    Returns:
        None.
    """
    # Get all tests paths
    list_path_tests =  [path for path in path_results.iterdir() if path.is_dir()]

    # Initialization
    results_avg_temp = {}
    results_avg = {}

    # Process metrics
    for path_test in list_path_tests:
        variables = []
        list_models = list(path_test.glob('*.pickle'))
        if len(list_models) == 0 or len(list_models) > 1:
            raise ValueError(f'Path {path_test} does not contain a single model.')
        model_obj = list_models[0]
        with open(model_obj, "rb") as f:
            model_dict = pickle.load(f)
        if model_dict["var_names"]:
            variables = get_full_rad_names(model_dict['var_info']['variables']['var_def'], model_dict["var_names"])
        for index, var in enumerate(variables):
            var = var.split("\\")[-1]   # Remove the path for windows
            var = var.split("/")[-1]    # Remove the path for linux
            if var not in results_avg_temp:
                results_avg_temp[var] = {
                    'importance_mean': [],
                    'times_selected': 0
                }
            
            results_avg_temp[var]['importance_mean'].append(model_dict['model'].feature_importances_[index])
            results_avg_temp[var]['times_selected'] += 1
    for var in results_avg_temp:
        results_avg[var] = {
            'importance_mean': np.sum(results_avg_temp[var]['importance_mean']) / len(list_path_tests),
            'times_selected': results_avg_temp[var]['times_selected']
        }
    
    del results_avg_temp
            
    save_json(path_results / 'feature_importance_analysis.json', results_avg, cls=NumpyEncoder)

def get_ml_test_table(variable_table: pd.DataFrame, var_names: List, var_def: str) -> pd.DataFrame:
    """
    Gets the test table with the variables that are present in the training table.

    Args:
        variable_table (pd.DataFrame): Table with the variables to use for the ML model that 
            will be matched with the training table.
        var_names (List): List of variable names used for the ML model .
        var_def (str): String of the full variables names used for the ML model.
    
    Returns:
        pd.DataFrame: Table with the variables that are present in the training table.
    """

    # Get the full variable names for training
    full_radvar_names_trained = get_full_rad_names(var_def, var_names).tolist()

    # Get the full variable names for testing
    full_rad_var_names_test = get_full_rad_names(
        variable_table.Properties['userData']['variables']['var_def'], 
        variable_table.columns.values
    ).tolist()

    # Get the indexes of the variables that are present in the training table
    indexes = []
    for radvar in full_radvar_names_trained:
        try:
            indexes.append(full_rad_var_names_test.index(radvar))
        except ValueError as e:
            print(e)
            raise ValueError('The variable ' + radvar + ' is not present in the test table.')

    # Get the test table with the variables that are present in the training table
    variable_table = variable_table.iloc[:, indexes]

    # User data - var_def
    str_names = '||'
    for v in range(len(var_names)):
        str_names += var_names[v] + ':' + full_radvar_names_trained[v] + '||'

    # Update metadata and variable names
    variable_table.columns = var_names
    variable_table.Properties['VariableNames'] = var_names
    variable_table.Properties['userData']['variables']['var_def'] = str_names
    variable_table.Properties['userData']['variables']['continuous'] = var_names
    
    # Rename columns to s sequential names again
    return variable_table

def finalize_rad_table(rad_table: pd.DataFrame) -> pd.DataFrame:
    """
    Finalizes the variable names and the associated metadata. Used to have sequential variable 
    names and UserData with only variable names present in the table.
    
    Args:
        rad_table (pd.DataFrame): radiomics table to be finalized.
    
    Returns:
        pd.DataFrame: Finalized radiomics table.
    """

    # Initialization
    var_names = rad_table.columns.values
    full_rad_names = get_full_rad_names(rad_table.Properties['userData']['variables']['var_def'], var_names)

    # User data - var_def
    str_names = '||'
    for v in range(var_names.size):
        var_names[v] = 'radVar' + str(v+1)
        str_names = str_names + var_names[v] + ':' + full_rad_names[v] + '||'

    # Update metadata and variable names
    rad_table.columns = var_names
    rad_table.Properties['VariableNames'] = var_names
    rad_table.Properties['userData']['variables']['var_def'] = str_names
    rad_table.Properties['userData']['variables']['continuous'] = var_names

    return rad_table

def get_radiomics_table(
        path_radiomics_csv: Path, 
        path_radiomics_txt: Path, 
        image_type: str, 
        patients_ids: List = None
    ) -> pd.DataFrame:
    """
    Loads the radiomics table from the .csv file and the associated metadata.
    
    Args:
        path_radiomics_csv (Path): full path to the csv file of radiomics table.
            --> Ex: /home/myStudy/FEATURES/radiomics__PET(GTV)__image.csv
        path_radiomics_txt: full path to the radiomics variable definitions in text format (associated
            to path_radiomics_csv).
            -> Ex: /home/myStudy/FEATURES/radiomics__PET(GTV)__image.txt
        image_type (str): String specifying the type of image on which the radiomics
            features were computed.
            --> Format: $scan$($roiType$)__$imSpace$
            --> Ex: PET(tumor)__HHH_coif1
        patients_ids (list, optional): List of strings specifying the patientIDs of
            patients to fetch from the radiomics table. If this
            argument is not present, all patients are fetched.
            --> Ex: {'Cervix-UCSF-001';Cervix-McGill-004}

    Returns:
        pd.DataFrame: radiomics table
    """
    # Read CSV table
    radiomics_table = pd.read_csv(path_radiomics_csv, index_col=0)
    if patients_ids is not None:
        patients_ids = intersect(patients_ids, list(radiomics_table.index))
        radiomics_table = radiomics_table.loc[patients_ids]

    # Read the associated TXT file
    with open(path_radiomics_txt, 'r') as f:
        user_data = f.read()

    # Grouping the information
    radiomics_table._metadata += ["Properties"]
    radiomics_table.Properties = dict()
    radiomics_table.Properties['userData'] = dict()
    radiomics_table.Properties['userData']['variables'] = dict()
    radiomics_table.Properties['userData']['variables']['var_def'] = user_data
    radiomics_table.Properties['Description'] = image_type

    # Only continuous will be used for now but this design will facilitate the use of 
    # other categories in the future.
    # radiomics = continous.
    radiomics_table.Properties['userData']['variables']['continuous'] = np.asarray(list(radiomics_table.columns.values))

    return radiomics_table

def get_splits(outcome: pd.DataFrame, n_split: int, test_split_proportion: float) -> Tuple[List, List]:
    """
    Splits the given outcome table in two sets.

    Args:
        outcome (pd.DataFrame): Table with a single outcome column of 0's and 1's.
        n_splits (int): Integer specifying the number of splits to create.
        test_split_proportion (float): Float between 0 and 1 specifying the proportion
                of patients to include in the test set.

    Returns:
        train_sets List of indexes for the train_sets.
        test_sets: List of indexes for the test_sets.

    """

    ind_neg = np.where(outcome == 0)
    n_neg = len(ind_neg[0])
    ind_pos = np.where(outcome == 1)
    n_pos = len(ind_pos[0])
    n_neg_test = round(test_split_proportion * n_neg)
    n_pos_test = round(test_split_proportion * n_pos)

    n_inst = len(outcome)
    n_test = n_pos_test + n_neg_test
    n_train = n_inst - n_test

    if(n_split==1):
        train_sets = np.zeros(n_train)
        test_sets = np.zeros(n_test)
    else:
        train_sets = np.zeros((n_split, n_train))
        test_sets = np.zeros((n_split, n_test))

    for s in range(n_split):
        ind_pos_test = np.random.choice(ind_pos[0], n_pos_test, replace=False)
        ind_neg_test = np.random.choice(ind_neg[0], n_neg_test, replace=False)

        ind_test = np.concatenate((ind_pos_test,ind_neg_test))
        ind_test.sort()

        ind_train = np.arange(n_inst)
        ind_train = np.delete(ind_train, ind_test)
        ind_train.sort()

        if(n_split>1):
            train_sets[s] = ind_train
            test_sets[s] = ind_test
        else:
            train_sets = ind_train
            test_sets = ind_test

    return train_sets, test_sets

def get_stratified_splits(
        outcome_table: pd.DataFrame,
        n_splits: int,
        test_split_proportion: float,
        seed: int,
        flag_by_cat: bool=False
    ) -> Tuple[List, List]:
    """
    Sub-divides a given outcome dataset into multiple stratified patient splits. 
    The stratification is performed per class proportion (or by institution).

    Args:
        outcome_table: Table with a single outcome column of 0's and 1's.
                    The rows of the table must define the patient IDs: $Cancer-$Institution-$Number.
        n_splits: Integer specifying the number of splits to create.
        test_split_proportion: Float between 0 and 1 specifying the proportion 
            of patients to include in the test set.
        seed: Integer specifying the random generator seed to use for random splitting.
        flag_by_cat (optional): Logical flag specifying if we are to produce 
            the split by taking into account the institutions in the outcome table. 
            If true, patients in Training and testing splits have the same prortion 
            of events per instiution as originally found in the initial data. Default: False.

    Returns:
        List: patients_train_splits, list of size nTrainXnSplit, where each entry
            is a string specifying a "Training" patient.
        List: patients_test_splits, list of size nTestXnSplit, where each entry
            is a string specifying a "testing" patient
    """
    patient_ids = pd.Series(outcome_table.index)
    patients_train_splits = []
    patients_test_splits = []

    # Take into account the institutions in the outcome table
    if flag_by_cat:
        institution_cat_vector = get_institutions_from_ids(patient_ids)
        all_categories = np.unique(institution_cat_vector)
        n_cat = len(all_categories)
        # Split for each institution
        for i in range(n_cat):
            np.random.seed(seed)
            cat = all_categories[i]
            flag_cat = institution_cat_vector == cat
            patient_ids_cat = patient_ids[flag_cat]
            patient_ids_cat.reset_index(inplace=True, drop=True)

            # Split train and test sets
            train_sets, test_sets = get_splits(outcome_table[flag_cat.values], n_splits, test_split_proportion)

            if n_splits > 1:
                temp_patients_train = np.empty((n_splits, len(train_sets[0])), dtype=object)
                temp_patientsTest = np.empty((n_splits, len(test_sets[0])), dtype=object)
                for s in range(n_splits):
                    temp_patients_train[s] = patient_ids_cat[train_sets[s]]
                    temp_patientsTest[s] = patient_ids_cat[test_sets[s]]
            else:
                temp_patients_train = patient_ids_cat[train_sets]
                temp_patients_train.reset_index(inplace=True, drop=True)
                temp_patientsTest = patient_ids_cat[test_sets]
                temp_patientsTest.reset_index(inplace=True, drop=True)
            
            # Initialize the train and test patients list (1st iteration)
            if i==0:
                patients_train_splits=temp_patients_train
                patients_test_splits=temp_patientsTest

            # Add new patients to the train and test patients list (other iterations)
            if i>0:
                if n_splits>1:
                    patients_train_splits = np.append(patients_train_splits, temp_patients_train, axis=1)
                    patients_test_splits = np.append(patients_test_splits, temp_patientsTest, axis=1)

                else: 
                    patients_train_splits = np.append(patients_train_splits, temp_patients_train)
                    patients_test_splits = np.append(patients_test_splits, temp_patientsTest)
    
    # Do not take into account the institutions in the outcome table
    else:
        # Split train and test sets
        train_sets, test_sets = get_splits(outcome_table, n_splits, test_split_proportion)
        if n_splits > 1:
            patients_train_splits = np.empty((n_splits, len(train_sets[0])), dtype=object)
            patients_test_splits = np.empty((n_splits, len(test_sets[0])), dtype=object)
            for s in range(n_splits):
                patients_train_splits[s] = patient_ids[train_sets[s]]
                patients_test_splits[s] = patient_ids[test_sets[s]]
        else:
            patients_train_splits = patient_ids[train_sets]
            patients_train_splits.reset_index(inplace=True, drop=True)
            patients_test_splits = patient_ids[test_sets]
            patients_test_splits.reset_index(inplace=True, drop=True)

    return patients_train_splits, patients_test_splits

def get_patient_id_classes(outcome_table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Yields the patients from the majority class and the minority class in the given outcome table.
    Only supports binary classes.
    
    Args:
        outcome_table(pd.DataFrame): outcome table with binary labels.
    
    Returns:
        pd.DataFrame: Majority class patientIDs.
        pd.DataFrame: Minority class patientIDs.
    """
    ones = outcome_table.loc[outcome_table.iloc[0:].values == 1].index
    zeros = outcome_table.loc[outcome_table.iloc[0:].values == 0].index
    if ones.size > zeros.size:
        return ones, zeros
    
    return zeros, ones

def intersect(list1: List, list2: List, sort: bool = False) -> List:
    """
    Returns the intersection of two list.

    Args:
        list1 (List): the first list.
        list2 (List): the second list.
        order (bool): if True, the intersection is sorted.
    
    Returns:
        List: the intersection of the two lists.
    """

    intersection = list(filter(lambda x: x in list1, list2))
    if sort:
        return sorted(intersection)
    return intersection

def intersect_var_tables(var_table1: pd.DataFrame, var_table2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function takes 2 variable table, compares the indexes and drops the
    ones that are not in both, then returns the 2 table.

    Args:
        var_table1 (pd.DataFrame): first variable table.
        var_table2 (pd.DataFrame): second variable table.
    
    Returns:
        pd.DataFrame: first variable table with the same indexes as the second.
        pd.DataFrame: second variable table with the same indexes as the first.
    """
    # Find the unique values in var_table1 that are not in var_table2
    missing = np.setdiff1d(var_table1.index.to_numpy(), var_table2.index.to_numpy())
    if missing.size > 0:
        var_table1 = var_table1.drop(missing)

    # Find the unique values in var_table2 that are not in var_table1
    missing = np.setdiff1d(var_table2.index.to_numpy(), var_table1.index.to_numpy())
    if missing.size > 0:
        var_table2 = var_table2.drop(missing)

    return var_table1, var_table2

def under_sample(outcome_table_binary: pd.DataFrame) -> pd.DataFrame:
    """
    Performs under-sampling to obtain an equal number of outcomes in the binary outcome table.
    
    Args:
        outcome_table_binary (pd.DataFrame): outcome table with binary labels.
    
    Returns:
        pd.DataFrame: outcome table with balanced binary labels.
    """

    # We place them prematurely in maj and min and correct it afterwards
    n_maj = (outcome_table_binary == 0).sum().values[0]
    n_min = (outcome_table_binary == 1).sum().values[0]
    if n_maj == n_min:
        return outcome_table_binary
    elif n_min > n_maj:
        n_min, n_maj = n_maj, n_min

    # Sample the patients from the majority class
    patient_ids_maj, patient_ids_min = get_patient_id_classes(outcome_table_binary)
    patient_ids_min = list(patient_ids_min)
    patient_ids_numpy = patient_ids_maj.to_numpy()
    np.random.shuffle(patient_ids_numpy)
    patient_ids_sample = list(patient_ids_numpy[0:n_min])
    new_ids = patient_ids_min + patient_ids_sample

    return outcome_table_binary.loc[new_ids, :]

def save_model(model: Dict, var_id: str, path_model: Path, ml: Dict = None, name_type: str = "") -> Dict:
    """
    Saves a given model locally as a pickle object and outputs a dictionary
    containing the model's information.

    Args:
        model (Dict): The model dict to save.
        var_id (str): The stduied variable. For ex: 'var3'.
        path_model (str): The path to save the model.
        ml (Dict, optional): Dicionary containing the settings of the machine learning experiment.
        name_type (str, optional): String specifying the type of the variable. For examlpe: "RadiomicsIntensity". Default is "".
    
    Returns:
        Dict: A dictionary containing the model's information.
    """
    # Saving model
    with open(path_model, "wb") as f:
        pickle.dump(model, f)

    # Getting the "var_names" string
    if ml is not None:
        var_names = ml['variables'][var_id]['nameType']
    elif name_type != "":
        var_names = name_type
    else:
        var_names = [var_id]

    # Recording model info
    model_info = dict()
    model_info['path'] = path_model
    model_info['var_ids'] = var_id
    model_info['var_type'] = var_names

    try: # This part may fail if model training failed.
        model_info['var_names'] = model['var_names']
        model_info['var_info'] = model['var_info']
        if 'normalization' in model_info['var_info'].keys():
            if 'normalization_table' in model_info['var_info']['normalization'].keys():
                normalization_struct = write_table_structure(model_info['var_info']['normalization']['normalization_table'])
                model_info['var_info']['normalization']['normalization_table'] = normalization_struct
        model_info['threshold'] = model['threshold']
    except Exception as e:
        print("Failed to create a fully model info")
        print(e)

    return model_info

def write_table_structure(data_table: pd.DataFrame) -> Dict:
    """
    Writes the structure of a table in a dictionary.

    Args:
        data_table (pd.DataFrame): a table.
    
    Returns:
        Dict: a dictionary containing the table's structure.
    """
    # Initialization
    data_struct = dict()

    if len(data_table.index) != 0:
        data_struct['index'] = list(data_table.index)

    # Creating the structure
    for column in data_table.columns:
        data_struct[column] = data_table[column]

    return data_struct
