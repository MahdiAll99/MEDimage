import os
from pathlib import Path
from typing import List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from networkx.drawing.nx_pydot import graphviz_layout
from numpyencoder import NumpyEncoder
from sklearn import metrics

from MEDimage.learning.ml_utils import feature_imporance_analysis
from MEDimage.utils.json_utils import load_json, save_json


class Results:
    def __init__(self, model_dict: dict = {}, model_id: str = "") -> None:
        """
        Constructor of the class Results
        """
        self.model_dict = model_dict
        self.model_id = model_id
        self.results_dict = {}

    def __calculate_performance(
            self, 
            response: list, 
            labels: pd.DataFrame, 
            thresh: float
        ) -> dict:
        """
        Computes performance metrics of given a model's response, outcome and threshold.

        Args:
            response (np.array): Column vector specifying the probability of class "1" for all instances (prediction)
            labels (np.array): Column vector specifying the outcome status (1 or 0) for all instances.
            thresh (float): Optimal threshold selected from the ROC curve.

        Returns:
            Dict: Dictionary containing the performance metrics.
        """
        # Removing Nans
        df = labels.copy()
        outcome_name = labels.columns.values[0]
        df['response'] = response
        df.dropna(axis=0, how='any', inplace=True)

        # Confusion matrix elements:
        TP = ((df['response'] >= thresh) & (df[outcome_name] == 1)).sum()
        TN = ((df['response'] < thresh) & (df[outcome_name] == 0)).sum()
        FP = ((df['response'] >= thresh) & (df[outcome_name] == 0)).sum()
        FN = ((df['response'] < thresh) & (df[outcome_name] == 1)).sum()

        # AUC
        auc = metrics.roc_auc_score(df[outcome_name], df['response'])

        # AUPRC
        auprc = metrics.average_precision_score(df[outcome_name], df['response'])

        # Sensitivity
        try:
            sensitivity = TP / (TP + FN)
        except(ZeroDivisionError):
            print('TP + FN = 0, Division by 0, replacing sensitivity by 0.5')
            sensitivity = 0.5

        # Specificity
        try:
            specificity = TN / (TN + FP)
        except(ZeroDivisionError):
            print('TN + FP= 0, Division by 0, replacing specificity by 0.5')
            specificity = 0.5

        # Balanced accuracy
        bac = (sensitivity + specificity) / 2

        # Precision
        precision = TP / (TP + FP)

        # NPV (Negative Predictive Value)
        npv = TN / (TN + FN)

        # Accuracy
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        # F1 score
        f1_score = 2 * TP / (2 * TP + FP + FN)

        # mcc (mathews correlation coefficient)
        mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

        # Recording results
        results_dict = dict()
        results_dict['TP'] = TP
        results_dict['TN'] = TN
        results_dict['FP'] = FP
        results_dict['FN'] = FN
        results_dict['AUC'] = auc
        results_dict['AUPRC'] = auprc
        results_dict['Sensitivity'] = sensitivity
        results_dict['Specificity'] = specificity
        results_dict['BAC'] = bac
        results_dict['Precision'] = precision
        results_dict['NPV'] = npv
        results_dict['Accuracy'] = accuracy
        results_dict['F1_score'] = f1_score
        results_dict['MCC'] = mcc

        return results_dict

    def __compute_midrank(self, x: np.array) -> np.array:
        """
        Computes midranks.
        Args:
            x(np.array): 1D array of probabilities.

        Returns:
            np.array: Midranks.
        """
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=np.float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5*(i + j - 1)
            i = j
        T2 = np.empty(N, dtype=np.float)
        # Note(kazeevn) +1 is due to Python using 0-based indexing
        # instead of 1-based in the AUC formula in the paper
        T2[J] = T + 1
        return T2

    def __fast_delong(self, predictions_sorted_transposed: np.array, label_1_count: int) -> tuple[float, float]:
        """
        Computes the empricial AUC and its covariance using the fast version of DeLong's method.

        Args:
            predictions_sorted_transposed (np.array): a 2D numpy.array[n_classifiers, n_examples]
                sorted such as the examples with label "1" are first.
            label_1_count (int): number of examples with label "1".
        
        Returns:
            Tuple(float, float): (AUC value, DeLong covariance)
        
        Reference:
            @article{sun2014fast,
            title={Fast Implementation of DeLong's Algorithm for
                    Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
            author={Xu Sun and Weichao Xu},
            journal={IEEE Signal Processing Letters},
            volume={21},
            number={11},
            pages={1389--1393},
            year={2014},
            publisher={IEEE}
        }
        """
        # Short variables are named as they are in the paper
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]

        tx = np.empty([k, m], dtype=np.float)
        ty = np.empty([k, n], dtype=np.float)
        tz = np.empty([k, m + n], dtype=np.float)
        for r in range(k):
            tx[r, :] = self.__compute_midrank(positive_examples[r, :])
            ty[r, :] = self.__compute_midrank(negative_examples[r, :])
            tz[r, :] = self.__compute_midrank(predictions_sorted_transposed[r, :])
        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n

        return aucs, delongcov

    def __compute_ground_truth_statistics(self, ground_truth: np.array) -> tuple[np.array, int]:
        """
        Computes the order of the ground truth and the number of positive examples.

        Args:
            ground_truth(np.array): np.array of 0 and 1.
        
        Returns:
            Tuple[np.array, int]: ground truth ordered and the number of positive examples.
        """
        assert np.array_equal(np.unique(ground_truth), [0, 1])
        order = (-ground_truth).argsort()
        label_1_count = int(ground_truth.sum())
        return order, label_1_count

    def __calc_pvalue(self, aucs: np.array, sigma: float) -> float:
        """
        Computes p-values of the AUCs distribution.
        Args:
            aucs(np.array): 1D array of AUCs.
            sigma (flaot): AUC DeLong covariances
        Returns:
            flaot: p-value of the AUCs.
        """
        l = np.array([[1, -1]])
        z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
        p_value = 2 * scipy.stats.norm.sf(z, loc=0, scale=1)
        return p_value

    def __delong_roc_test(self, ground_truth: np.array, predictions_one: list, predictions_two: list) -> float:        
        """
        Computes log(p-value) for hypothesis that two ROC AUCs are different
        
        Args:
            ground_truth(np.array): np.array of 0 and 1
            predictions_one(np.array): np.array of floats of the probability of being class 1 for the first model.
            predictions_two(np.array): np.array of floats of the probability of being class 1 for the second model.
        
        Returns:
            flaot: p-value of the AUCs.
        """
        order, label_1_count = self.__compute_ground_truth_statistics(ground_truth)
        predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
        aucs, delongcov = self.__fast_delong(predictions_sorted_transposed, label_1_count)
        return self.__calc_pvalue(aucs, delongcov)
    
    def __get_metrics_failure_dict(
            self, 
            metrics: list = [
                'AUC', 'Sensitivity', 'Specificity', 
                'BAC', 'AUPRC', 'Precision', 
                'NPV', 'Accuracy', 'F1_score', 'MCC',
                'TP', 'TN', 'FP', 'FN'
            ]
        ) -> dict:
        """
        This function fills the metrics with NaNs in case of failure.

        Args:
            metrics (list, optional): List of metrics to be filled with NaNs. 
                Defaults to ['AUC', 'Sensitivity', 'Specificity', 'BAC', 
                'AUPRC', 'Precision', 'NPV', 'Accuracy', 'F1_score', 'MCC'
                'TP', 'TN', 'FP', 'FN'].
        
        Returns:
            Dict: Dictionary with the metrics filled with NaNs.
        """
        failure_struct = dict()
        failure_struct = dict()
        for metric in metrics:
            failure_struct[metric] = np.nan

        return failure_struct
    
    def __save_roc_curve(self, response: list, outcome_binary: pd.DataFrame, label: str) -> None:
        """
        This function saves the ROC curve of the model.

        Args:
            response (List): List of machine learning model predictions of the class 1.
            outcome_binary (pd.DataFrame): Outcome table with binary labels.
            label (str): Label that gives context to the ROC curve. For example, "Test" or "Traing".
        
        Returns:
            None: Saves the ROC curve.
        """
        # Removing Nans
        df = outcome_binary.copy()
        outcome_name = outcome_binary.columns.values[0]
        df['response'] = response
        df.dropna(axis=0, how='any', inplace=True)

        # Get ROC curve
        fpr, tpr, _ = metrics.roc_curve(df[outcome_name], df['response'])

        # Create ROC curve plot and save it
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve - ' + label)
        plt.legend(loc="lower right")
        plt.savefig(os.path.dirname(self.model_dict['path']) + "/ROC_curve_" + label + ".png")
    
    def __count_percentage_levels(self, features_dict: dict, fda: bool = False) -> list:
        """
        Counts the percentage of each radiomics level in a given features dictionary.

        Args:
            features_dict (dict): Dictionary of features.
            fda (bool, optional): If True, the features are from the FDA logging dict and will be
                treated differently. Defaults to False.
        
        Returns:
            list: List of percentages of features in each complexity levels.
        """
        # Intialization
        perc_levels = [0] * 7   # 4 levels and two variants for the filters

        # List all features in dict
        if fda:
            list_features = [feature.split('/')[-1] for feature in features_dict['final']]
        else:
            list_features = list(features_dict.keys())

        # Count the percentage of levels
        for feature in list_features:
            level_name = feature.split('__')[1].lower()
            feature_name = feature.split('__')[2].lower()
            # Morph
            if level_name.startswith('morph'):
                perc_levels[0] += 1
            # Intensity
            elif level_name.startswith('intensity'):
                perc_levels[1] += 1
            # Texture
            elif level_name.startswith('texture'):
                perc_levels[2] += 1
            # Linear filters
            elif level_name.startswith('mean') \
                or level_name.startswith('log') \
                or level_name.startswith('laws') \
                or level_name.startswith('gabor') \
                or level_name.startswith('wavelet') \
                or level_name.startswith('coif'):
                # seperate intensity and texture
                if feature_name.startswith('_int'):
                    perc_levels[3] += 1
                elif feature_name.startswith(tuple(['_glcm', '_gldzm', '_glrlm', '_glszm', '_ngtdm', '_ngldm'])):
                    perc_levels[4] += 1
            # Textural filters
            elif level_name.startswith('glcm'):
                # seperate intensity and texture
                if feature_name.startswith('_int'):
                    perc_levels[5] += 1
                elif feature_name.startswith(tuple(['_glcm', '_gldzm', '_glrlm', '_glszm', '_ngtdm', '_ngldm'])):
                    perc_levels[6] += 1
                
        return perc_levels / np.sum(perc_levels, axis=0) * 100

    def __count_percentage_radiomics(self, results_dict: dict) -> list:
        """
        Counts the percentage of radiomics levels for all features used for the experiment.

        Args:
            results_dict (dict): Dictionary of final run results.
        
        Returns:
            list: List of percentages of features used for the model sorted by complexity levels.
        """
        # Intialization
        perc_levels = [0] * 5   # 5 levels: morph, intensity, texture, linear filters, textural filters
        model_name = list(results_dict.keys())[0]
        radiomics_tables_dict = results_dict[model_name]['var_info']['normalization']

        # Count the percentage of levels
        for key in list(radiomics_tables_dict.keys()):
            if key.lower().startswith('radtab'):
                table_path = radiomics_tables_dict[key]['original_data']['path_radiomics_csv']
                table_name = table_path.split('/')[-1]
                table = pd.read_csv(table_path, index_col=0)
                # Morph
                if 'morph' in table_name.lower():
                    perc_levels[0] += table.columns.shape[0]
                # Intensity
                elif 'intensity' in table_name.lower():
                    perc_levels[1] += table.columns.shape[0]
                # Texture
                elif 'texture' in table_name.lower():
                    perc_levels[2] += table.columns.shape[0]
                # Linear filters
                elif 'mean' in table_name.lower() \
                    or 'log' in table_name.lower() \
                    or 'laws' in table_name.lower() \
                    or 'gabor' in table_name.lower() \
                    or 'wavelet' in table_name.lower() \
                    or 'coif' in table_name.lower():
                    perc_levels[3] += table.columns.shape[0]
                # Textural filters
                elif 'glcm' in table_name.lower():
                    perc_levels[4] += table.columns.shape[0]
                
        return perc_levels / np.sum(perc_levels, axis=0) * 100

    def __count_stable_fda(self, features_dict: dict) -> list:
        """
        Counts the percentage of levels in the features dictionary.

        Args:
            features_dict (dict): Dictionary of features.
        
        Returns:
            list: List of percentages of features in each complexity levels.
        """
        # Intialization
        count_levels = [0] * 5   # 5 levels and two variants for the filters

        # List all features in dict
        features_dict = features_dict["one_space"]["unstable"]
        list_features = list(features_dict.keys())

        # Count the percentage of levels
        for feature_name in list_features:
            # Morph
            if feature_name.lower().startswith('morph'):
                count_levels[0] += features_dict[feature_name]
            # Intensity
            elif feature_name.lower().startswith('intensity'):
                count_levels[1] += features_dict[feature_name]
            # Texture
            elif feature_name.lower().startswith('texture'):
                count_levels[2] += features_dict[feature_name]
            # Linear filters
            elif feature_name.lower().startswith('mean') \
                or feature_name.lower().startswith('log') \
                or feature_name.lower().startswith('laws') \
                or feature_name.lower().startswith('gabor') \
                or feature_name.lower().startswith('wavelet') \
                or feature_name.lower().startswith('coif'):
                    count_levels[3] += features_dict[feature_name]
            # Textural filters
            elif feature_name.lower().startswith('glcm'):
                count_levels[4] += features_dict[feature_name]
                
        return count_levels
    
    def average_results(self, path_results: Path, save: bool = False) -> None:
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

        # Retrieve metrics
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

    def get_delong_p_value(
            self, 
            path_experiment: Path, 
            experiment: str, 
            levels: List, 
            modalities: List, 
            nb_split: int = 10
        ) -> float:
        """
        Calculates the p-value of the Delong test for the given experiment.

        Args:
            path_experiment (Path): Path to the folder containing the experiment.
            experiment (str): Name of the experiment.
            levels (List): List of levels to analyze.
            modalities (List): List of modalities to analyze.
        
        Returns:
            float: p-value of the Delong test.
        """
        # Assertions
        if len(modalities) == 1:
            assert len(levels) == 2, "The number of levels must be 2 for a single modality"
        elif len(modalities) == 2:
            assert len(levels) == 1, "The number of levels must be 1 for two modalities"
        else:
            raise ValueError("The number of modalities must be 1 or 2")
        
        # Load outcomes dataframe
        outcomes = pd.read_csv(path_experiment / "outcomes.csv", sep=',')

        # Initialization
        list_p_values_temp = list()
        patients_ids_one_all = list()
        patients_ids_two_all = list()
        predictions_one_all = list()
        predictions_two_all = list()

        # For each split
        for i in range(1, nb_split + 1):
            # Get level and modality
            if len(modalities) == 1:
                # Load ground truths and predictions
                if i < 10:
                    path_json_1 = path_experiment / f'learn__{experiment}_{levels[0]}_{modalities[0]}' / f'test__00{i}' / 'run_results.json'
                    path_json_2 = path_experiment / f'learn__{experiment}_{levels[1]}_{modalities[0]}' / f'test__00{i}' / 'run_results.json'
                else:
                    path_json_1 = path_experiment / f'learn__{experiment}_{levels[0]}_{modalities[0]}' / f'test__0{i}' / 'run_results.json'
                    path_json_2 = path_experiment / f'learn__{experiment}_{levels[1]}_{modalities[0]}' / f'test__0{i}' / 'run_results.json'
            else:
                # Load ground truths and predictions
                if i < 10:
                    path_json_1 = path_experiment / f'learn__{experiment}_{levels[0]}_{modalities[0]}' / f'test__00{i}' / 'run_results.json'
                    path_json_2 = path_experiment / f'learn__{experiment}_{levels[0]}_{modalities[1]}' / f'test__00{i}' / 'run_results.json'
                else:
                    path_json_1 = path_experiment / f'learn__{experiment}_{levels[0]}_{modalities[0]}' / f'test__0{i}' / 'run_results.json'
                    path_json_2 = path_experiment / f'learn__{experiment}_{levels[0]}_{modalities[1]}' / f'test__0{i}' / 'run_results.json'

            # Load models dicts
            model_one = load_json(path_json_1)
            model_two = load_json(path_json_2)

            # Get name models
            name_model_one = list(model_one.keys())[0]
            name_model_two = list(model_two.keys())[0]

            # Get predictions
            predictions_one = np.array(model_one[name_model_one]['test']['response'])
            predictions_one = np.reshape(predictions_one, (predictions_one.shape[0])).tolist()
            predictions_two = np.array(model_two[name_model_two]['test']['response'])
            predictions_two = np.reshape(predictions_two, (predictions_two.shape[0])).tolist()

            # Get patients ids
            patients_ids_one = model_one[name_model_one]['test']['patients'] 
            patients_ids_two = model_two[name_model_two]['test']['patients']

            # Check if the number of patients is the same
            patients_delete = []
            if len(patients_ids_one) > len(patients_ids_two):
                for patient_id in patients_ids_one:
                    if patient_id not in patients_ids_two:
                        patients_delete.append(patient_id)
                        predictions_one.pop(patients_ids_one.index(patient_id))
                for patient in patients_delete:
                    patients_ids_one.remove(patient)
            elif len(patients_ids_one) < len(patients_ids_two):
                for patient_id in patients_ids_two:
                    if patient_id not in patients_ids_one:
                        patients_delete.append(patient_id)
                        predictions_two.pop(patients_ids_two.index(patient_id))
                for patient in patients_delete:
                    patients_ids_two.remove(patient)

            # Check if the patient IDs are the same
            if patients_ids_one != patients_ids_two:
                raise ValueError("The patient IDs must be the same for both models")
        
            # Check if the number of predictions is the same
            if len(predictions_one) != len(predictions_two):
                raise ValueError("The number of predictions must be the same for both models")
            
            # Add-up all information
            patients_ids_one_all += patients_ids_one
            patients_ids_two_all += patients_ids_two
            predictions_one_all += predictions_one
            predictions_two_all += predictions_two

        # Check if the patient IDs are the same
        if patients_ids_one_all != patients_ids_two_all:
            raise ValueError("The patient IDs must be the same for both models")
    
        # Check if the number of predictions is the same
        if len(predictions_one_all) != len(predictions_two_all):
            raise ValueError("The number of predictions must be the same for both models")
        
        seen = {}  # Create a dictionary to track seen elements
        unique_list = []  # Create a list to store unique elements
        removed_indices = []  # Create a list to store indices of removed elements

        for index, item in enumerate(patients_ids_one_all):
            if item not in seen:
                seen[item] = True  # Mark the element as seen in the dictionary
                unique_list.append(item)  # Append the unique element to the result list
            else:
                removed_indices.append(index)  # Record the index of the duplicate element

        # Remove the duplicate elements from the list of all predictions
        for index in sorted(removed_indices, reverse=True):
            del predictions_one_all[index]
            del predictions_two_all[index]
            del patients_ids_one_all[index]
            del patients_ids_two_all[index]
        
        # Get ground truth for selected patients
        ground_truth = outcomes[outcomes['PatientID'].isin(unique_list)][outcomes.columns[-1]].values

        # Compute p-value
        list_p_values_temp.append(self.__delong_roc_test(ground_truth, predictions_one_all, predictions_two_all).item())

        # Compute the median p-value of all splits
        return list_p_values_temp

    def get_ttest_p_value(
            self, 
            path_experiment: Path, 
            experiment: str, 
            levels: List, 
            modalities: List, 
            metric: str = 'AUC',
            nb_split: int = 10
        ) -> float:
        """
        Calculates the p-value using the t-test for two related samples of scores.

        Args:
            path_experiment (Path): Path to the folder containing the experiment.
            experiment (str): Name of the experiment.
            levels (List): List of levels to analyze.
            modalities (List): List of modalities to analyze.
            metric (str, optional): Metric to analyze. Defaults to 'AUC'.
            n_split (int, optional): Number of splits to analyze. Defaults to 10.
        
        Returns:
            float: p-value of the Delong test.
        """
        # Assertions
        if len(modalities) == 1:
            assert len(levels) == 2, "The number of levels must be 2 for a single modality"
        elif len(modalities) == 2:
            assert len(levels) == 1, "The number of levels must be 1 for two modalities"
        else:
            raise ValueError("The number of modalities must be 1 or 2")

        # Initialization
        metrics_one_all = list()
        metrics_two_all = list()

        # For each split
        for i in range(1, nb_split + 1):
            # Get level and modality
            if len(modalities) == 1:
                # Load ground truths and predictions
                if i < 10:
                    path_json_1 = path_experiment / f'learn__{experiment}_{levels[0]}_{modalities[0]}' / f'test__00{i}' / 'run_results.json'
                    path_json_2 = path_experiment / f'learn__{experiment}_{levels[1]}_{modalities[0]}' / f'test__00{i}' / 'run_results.json'
                else:
                    path_json_1 = path_experiment / f'learn__{experiment}_{levels[0]}_{modalities[0]}' / f'test__0{i}' / 'run_results.json'
                    path_json_2 = path_experiment / f'learn__{experiment}_{levels[1]}_{modalities[0]}' / f'test__0{i}' / 'run_results.json'
            else:
                # Load ground truths and predictions
                if i < 10:
                    path_json_1 = path_experiment / f'learn__{experiment}_{levels[0]}_{modalities[0]}' / f'test__00{i}' / 'run_results.json'
                    path_json_2 = path_experiment / f'learn__{experiment}_{levels[0]}_{modalities[1]}' / f'test__00{i}' / 'run_results.json'
                else:
                    path_json_1 = path_experiment / f'learn__{experiment}_{levels[0]}_{modalities[0]}' / f'test__0{i}' / 'run_results.json'
                    path_json_2 = path_experiment / f'learn__{experiment}_{levels[0]}_{modalities[1]}' / f'test__0{i}' / 'run_results.json'


            # Load models dicts
            model_one = load_json(path_json_1)
            model_two = load_json(path_json_2)

            # Get name models
            name_model_one = list(model_one.keys())[0]
            name_model_two = list(model_two.keys())[0]

            # Get predictions
            metric_one = model_one[name_model_one]['test']['metrics'][metric]
            metric_two = model_two[name_model_two]['test']['metrics'][metric]
            
            # Add-up all information
            metrics_one_all.append(metric_one)
            metrics_two_all.append(metric_two)
    
        # Check if the number of predictions is the same
        if len(metrics_one_all) != len(metrics_two_all):
            raise ValueError("The number of metrics must be the same for both models")
        
        # Compute p-value by performing paired t-test
        _, p_value = scipy.stats.ttest_rel(metrics_one_all, metrics_two_all)

        return p_value
     
    def count_patients(self, path_results: Path) -> dict:
        """
        Counts the number of patients used in learning, testing and holdout.

        Args:
            path_results(Path): path to the folder containing the results of the experiment.
        
        Returns:
            Dict: Dictionary with the number of patients used in learning, testing and holdout.
        """
        # Get all tests paths
        list_path_tests =  [path for path in path_results.iterdir() if path.is_dir()]

        # Initialize dictionaries
        patients_count = {
            'train': {},
            'test': {},
            'holdout': {}
        }

        # Process metrics
        for dataset in ['train', 'test', 'holdout']:
            for path_test in list_path_tests:
                results_dict = load_json(path_test / 'run_results.json')
                if dataset in results_dict[list(results_dict.keys())[0]].keys():
                    if 'patients' in results_dict[list(results_dict.keys())[0]][dataset].keys():
                        if results_dict[list(results_dict.keys())[0]][dataset]['patients']:
                            patients_count[dataset] = len(results_dict[list(results_dict.keys())[0]][dataset]['patients'])
                    else:
                        continue
                else:
                    continue
                break   # The number of patients is the same for all the runs

        return patients_count
    
    def get_model_performance_metrics(
            self, 
            response: list, 
            outcome_table: pd.DataFrame,
            threshold, 
            label: str
        ) -> dict:
        """
        This function calculates the performance metrics for the given machine learning model reponse.

        Args:
            response (List): List of machine learning model predictions.
            outcome_table (pd.DataFrame): Outcome table with binary labels.
            threshold (float): Optimal threshold selected from the ROC curve.
            label (str): Label that gives context to the ROC curve. For example, "Test" or "Traing".
        
        Returns:
            Dict: Dictionary with the performance metrics.
        """
        
        # Convert list of model response to a table to facilitate the process
        results_dict = dict()
        patient_ids = list(outcome_table.index)
        response_table = pd.DataFrame(response)
        response_table.index = patient_ids
        response_table._metadata += ['Properties']
        response_table.Properties = dict()
        response_table.Properties['RowNames'] = patient_ids

        # Make sure the outcome table and the response table have the same patients
        outcome_binary = outcome_table.loc[patient_ids, :]
        outcome_binary = outcome_binary.iloc[:, 0]
        response = response_table.loc[patient_ids, :]
        response = response.iloc[:, 0]

        # Create ROC curve and save it
        self.__save_roc_curve(response, outcome_binary.to_frame(), label)
        
        # Calculating performance
        results_dict = self.__calculate_performance(response, outcome_binary.to_frame(), threshold)

        return results_dict
    
    def get_model_performance(
            self, 
            response: list, 
            outcome_table: pd.DataFrame,
            label:str
        ) -> None:
        """
        Calculates the performance of the model
        Args:
            response (list): List of machine learning model predictions.
            outcome_table (pd.DataFrame): Outcome table with binary labels.
            label (str): Label that gives context to the ROC curve. For example, "Test" or "Traing".
        
        Returns:
            None: Updates the ``run_results`` attribute.
        """
        # Calculating performance metrics for the training set
        try:
            return self.get_model_performance_metrics(response, outcome_table, self.model_dict['threshold'], label)
        except Exception as e:
            print(f"Error in get_model_performance_metrics: ", e, "filling metrics with nan...")
            return self.__get_metrics_failure_dict()
    
    def plot_heatmap(
            self, 
            path_experiments: Path, 
            experiment: str,
            levels: List,
            stat_extra: list = [],
            modalities: List = [], 
            metric: str = 'AUC_mean',
            plot_p_values: bool = False,
            p_value_test: str = 'ttest',
            title: str = None,
            save: bool = False,
            figsize: tuple = (8, 8)
        ) -> None:
        """
        This function plots a heatmap of the metrics values for the performance of the models in the given experiment.

        Args:
            path_experiments (Path): Path to the folder containing the experiments.
            experiment (str): Name of the experiment to plot. Will be used to find the results.
            levels (List): List of radiomics levels to include in plot. For example: ['morph', 'intensity'].
            stat (str, optional): Statistic to plot. Defaults to 'mean'.
            stat_extra (list, optional): List of extra statistics to include in the plot. Defaults to [].
            modalities (List, optional): List of imaging modalities to include in the plot. Defaults to [].
            metric (str, optional): Metric to plot. Defaults to 'AUC'.
            plot_p_values (bool, optional): If True plots the p-value of the choosen test. Defaults to False.
            p_value_test (str, optional): Method to use to calculate the p-value. Defaults to 'ttest'.
            title (str, optional): Title of the plot. Defaults to None.
            save (bool, optional): Whether to save the plot. Defaults to False.
            figsize (tuple, optional): Size of the figure. Defaults to (8, 8).
        
        Returns:
            None.
        """
        # Make sure the metric is in the list of metrics
        list_metrics = [
            'AUC', 'Sensitivity', 'Specificity', 
            'BAC', 'AUPRC', 'Precision', 
            'NPV', 'Accuracy', 'F1_score', 'MCC',
            'TP', 'TN', 'FP', 'FN'
        ]
        assert metric.split('_')[0] in list_metrics, f'Given metric {metric} is not in the list of metrics. Please choose from {list_metrics}'

        # Prepare the data for the heatmap
        # Average of the results over all the runs
        results_dicts = []
        patients_count = dict.fromkeys(modalities)
        for modality in modalities:
            for level in levels:
                exp_full_name = 'learn__' + experiment + '_' + level + '_' + modality
                results_dict = self.average_results(path_experiments / exp_full_name)
                results_dicts.append(results_dict)

            # Patient count
            patients_count[modality] = self.count_patients(path_experiments / exp_full_name)

        # Create the heatmap data using the metric of interest
        if plot_p_values:
            heatmap_data = np.zeros((len(modalities)*2, len(levels)))
        else:
            heatmap_data = np.zeros((len(modalities), len(levels)))

        # Fill the heatmap data
        labels = heatmap_data.tolist()
        for i in range(len(modalities)):
            for j in range(len(levels)):
                # get metrics and p-values
                results_dict = results_dicts[i * len(levels) + j]
                metric_stat = round(results_dict['test'][metric], 2)
                if plot_p_values:
                    heatmap_data[i*2, j] = metric_stat
                    if p_value_test == 'ttest':
                        if j < len(levels) - 1:
                            metric_pvalue = metric.split('_')[0] if '_' in metric else metric
                            heatmap_data[i*2+1, j+1] = self.get_ttest_p_value(
                                path_experiments, 
                                experiment, 
                                [levels[j], levels[j+1]], 
                                [modalities[i]], 
                                metric_pvalue
                            )
                    elif p_value_test == 'delong':
                        heatmap_data[i, j] = self.get_delong_p_value(
                            path_experiments, 
                            experiment, 
                            [levels[j], levels[j+1]], 
                            [modalities[i]]
                        )
                else:
                    heatmap_data[i, j] = metric_stat
                
                # Extra statistics
                if stat_extra:
                    if plot_p_values:
                        labels[i*2][j] = f'{metric_stat}'
                        if j < len(levels) - 1:
                            labels[i*2+1][j+1] = f'{round(heatmap_data[i*2+1, j+1], 5)}'
                            labels[i*2+1][0] = '-'
                        for extra_stat in stat_extra:
                            extra_metric_stat = round(results_dict['test'][extra_stat], 2)
                            labels[i*2][j] += f'\n{extra_stat}: {extra_metric_stat}'
                    else:
                        labels[i][j] = f'{metric_stat}'
                        for extra_stat in stat_extra:
                            extra_metric_stat = round(results_dict['test'][extra_stat], 2)
                            labels[i][j] += f'\n{extra_stat}: {extra_metric_stat}'
                else:
                    labels = np.array(heatmap_data).round(4).tolist()
        
        # Update modality name to include the number of patients for training and testing
        modalities = [modality + f' ({patients_count[modality]["train"]} train, {patients_count[modality]["test"]} test)' for modality in modalities]

        # Set up the rows (modalities and p-values)
        if plot_p_values:
            modalities_temp = modalities.copy()
            modalities = ['p_value'] * len(modalities_temp) * 2
            for idx in range(len(modalities)):
                if idx % 2 == 0:
                    modalities[idx] = modalities_temp[idx // 2]

        # Convert the numpy array to a DataFrame for Seaborn
        df = pd.DataFrame(heatmap_data, columns=levels, index=modalities)
            
        # Count the patients used in learning
        patients_count = self.count_patients(path_experiments / exp_full_name)

        # Create the heatmap using seaborn
        plt.figure(figsize=figsize)
        sns.set(font_scale=1.2)
        if metric == 'MCC':
            sns.heatmap(
                df, 
                annot=labels, 
                fmt="", 
                cmap="Blues", 
                cbar=True, 
                linewidths=0.5, 
                vmin=-1, 
                vmax=1, 
                annot_kws={"weight": "bold", "fontsize": 12}
            )
        else:
            sns.heatmap(
                df, 
                annot=labels, 
                fmt="", 
                cmap="Blues", 
                cbar=True, 
                linewidths=0.5, 
                vmin=0, 
                vmax=1, 
                annot_kws={"weight": "bold", "fontsize": 12}
            )
        
        plt.title("Testing Results Heatmap")
        plt.xticks(rotation=45)
        plt.tight_layout()
        if title:
            plt.title(title)
        else:
            plt.title(f'{metric} heatmap')

        # Save the heatmap
        if save:
            if title:
                plt.savefig(path_experiments / f'{title}.png')
            else:
                plt.savefig(path_experiments / f'{metric}_heatmap.png')
        else:
            plt.show()
    
    def plot_radiomics_starting_percentage(
            self, 
            path_experiments: Path, 
            experiment: str,
            levels: List,
            modalities: List = [], 
            title: str = None,
            figsize: tuple = (15, 10),
            save: bool = False
        ) -> None:
        """
        This function plots a heatmap with the performance of the models in the given experiment.

        Args:
            path_experiments (Path): Path to the folder containing the experiments.
            experiment (str): Name of the experiment to plot. Will be used to find the results.
            levels (List): List of radiomics levels to include in the plot.
            modalities (List, optional): List of imaging modalities to include in the plot. Defaults to [].
            title(str, optional): Title and name used to save the plot. Defaults to None.
            figsize(tuple, optional): Size of the figure. Defaults to (15, 10).
            save (bool, optional): Whether to save the plot. Defaults to False.
        
        Returns:
            None.
        """
        # Levels names
        levels_names = [
            'Morphology', 
            'Intensity', 
            'Texture', 
            'Linear filters', 
            'Textural filters'
        ]

        # Initialization
        colors_sns = sns.color_palette("pastel", n_colors=5)

        # Create mutliple plots for the pie charts
        fig, axes = plt.subplots(len(modalities), len(levels), figsize=figsize)

        # Load the models resutls
        for i, modality in enumerate(modalities):
            for j, level in enumerate(levels):
                exp_full_name = 'learn__' + experiment + '_' + level + '_' + modality
                if 'test__001' in os.listdir(path_experiments / exp_full_name):
                    run_results_dict = load_json(path_experiments / exp_full_name / 'test__001' / 'run_results.json')
                else:
                    raise FileNotFoundError(f'no test file named test__001 in {path_experiments / exp_full_name}')

                # Extract percentage of features per level
                perc_levels = np.round(self.__count_percentage_radiomics(run_results_dict), 2)
                
                # Update heatmap data
                if len(modalities) > 1:
                    axes[i, j].pie(
                        perc_levels, 
                        autopct= lambda p: '{:.1f}%'.format(p) if p > 0 else '',
                        pctdistance=0.8,
                        startangle=120, 
                        rotatelabels=True, 
                        textprops={'fontsize': 14, 'weight': 'bold'},
                        colors=colors_sns)
                    axes[i, j].set_title(f'{level} - {modality}', fontsize=15)
                else:
                    axes[j].pie(
                        perc_levels, 
                        autopct= lambda p: '{:.1f}%'.format(p) if p > 0 else '',
                        pctdistance=0.8,
                        startangle=120, 
                        rotatelabels=True, 
                        textprops={'fontsize': 14, 'weight': 'bold'},
                        colors=colors_sns)
                    axes[j].set_title(f'{level} - {modality}', fontsize=15)
        
        # Add legend
        plt.legend(levels_names, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 15})
        fig.tight_layout()

        if title:
            fig.suptitle(title, fontsize=20)
        else:
            fig.suptitle(f'{experiment}: % of starting features per level', fontsize=20)

        # Save the heatmap
        if save:
            if title:
                plt.savefig(path_experiments / f'{title}.png')
            else:
                plt.savefig(path_experiments / f'{experiment}_percentage_starting_features.png')
        else:
            plt.show()
    
    def plot_fda_analysis_heatmap(
            self, 
            path_experiments: Path, 
            experiment: str,
            levels: List,
            modalities: List = [], 
            title: str = None,
            save: bool = False
        ) -> None:
        """
        This function plots a heatmap with the performance of the models in the given experiment.

        Args:
            path_experiments (Path): Path to the folder containing the experiments.
            experiment (str): Name of the experiment to plot. Will be used to find the results.
            levels (List): List of radiomics levels to include in plot. For example: ['morph', 'intensity'].
            modalities (List, optional): List of imaging modalities to include in the plot. Defaults to [].
            title(str, optional): Title and name used to save the plot. Defaults to None.
            save (bool, optional): Whether to save the plot. Defaults to False.
        
        Returns:
            None.
        """
        # Levels names
        levels_names = [
            'Morphology',
            'Intensity',
            'Texture',
            'LF - Intensity',
            'LF - Texture',
            'TF - Intensity',
            'TF - Texture'
        ]
        level_names_stable = [
            'Morphology',
            'Intensity',
            'Texture',
            'LF',
            'TF'
        ]

        # Initialization
        colors_sns = sns.color_palette("pastel", n_colors=5)
        colors_sns_stable = sns.color_palette("pastel", n_colors=5)
        colors_sns.insert(3, colors_sns[3])
        colors_sns.insert(5, colors_sns[-1])
        hatch = ['', '', '', '..', '//', '..', '//']

        # Set hatches color
        plt.rcParams['hatch.color'] = 'white'

        # Create mutliple plots for the pie charts
        fig, axes = plt.subplots(len(modalities) * 2, len(levels), figsize=(18, 10))

        # Load the models resutls
        for i, modality in enumerate(modalities):
            for j, level in enumerate(levels):
                perc_levels_stable = []
                perc_levels_final = []
                exp_full_name = 'learn__' + experiment + '_' + level + '_' + modality
                for folder in os.listdir(path_experiments / exp_full_name):
                    if folder.lower().startswith('test__'):
                        if 'fda_logging_dict.json' in os.listdir(path_experiments / exp_full_name / folder):
                                fda_dict = load_json(path_experiments / exp_full_name / folder / 'fda_logging_dict.json')
                                perc_levels_stable.append(self.__count_stable_fda(fda_dict))
                                perc_levels_final.append(self.__count_percentage_levels(fda_dict, fda=True))
                        else:
                            raise FileNotFoundError(f'no fda_logging_dict.json file in {path_experiments / exp_full_name / folder}')

                # Average the results
                perc_levels_stable = np.mean(perc_levels_stable, axis=0).astype(int)
                perc_levels_final = np.mean(perc_levels_final, axis=0).astype(int)
                
                # Update heatmap data
                # Plot stable features
                axes[i*2, j].pie(
                    perc_levels_stable,
                    pctdistance=0.6,
                    startangle=120, 
                    radius=1.1, 
                    rotatelabels=True, 
                    textprops={'fontsize': 14, 'weight': 'bold'},
                    colors=colors_sns_stable
                    )
                # Title
                axes[i*2, j].set_title(f'{level} - {modality} - Stable', fontsize=15)

                # Legend
                legends = [f'{level} - {perc_levels_stable[idx]}' for idx, level in enumerate(level_names_stable)]
                axes[i*2, j].legend(legends, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 13})
                
                # Plot final features
                axes[i*2+1, j].pie(
                    perc_levels_final, 
                    autopct= lambda p: '{:.1f}%'.format(p) if p > 0 else '',
                    pctdistance=0.6,
                    startangle=120, 
                    radius=1.1, 
                    rotatelabels=True, 
                    textprops={'fontsize': 14, 'weight': 'bold'},
                    colors=colors_sns,
                    hatch=hatch)
                # Title
                axes[i*2+1, j].set_title(f'{level} - {modality} - Fianl 10', fontsize=15)

                # Legend
                axes[i*2+1, j].legend(levels_names, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 13})
        
        # Add legend
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        if title:
            fig.suptitle(title, fontsize=20)
        else:
            fig.suptitle(f'{experiment}: FDA breakdown per level', fontsize=20)

        # Save the heatmap
        if save:
            if title:
                plt.savefig(path_experiments / f'{title}.png')
            else:
                plt.savefig(path_experiments / f'{experiment}_fda_features.png')
        else:
            plt.show()
    
    def plot_feature_analysis(
            self, 
            path_experiments: Path, 
            experiment: str,
            levels: List,
            modalities: List = [], 
            title: str = None,
            save: bool = False
        ) -> None:
        """
        This function plots a heatmap with the performance of the models in the given experiment.

        Args:
            path_experiments (Path): Path to the folder containing the experiments.
            experiment (str): Name of the experiment to plot. Will be used to find the results.
            levels (List): List of radiomics levels to include in plot. For example: ['morph', 'intensity'].
            modalities (List, optional): List of imaging modalities to include in the plot. Defaults to [].
            title(str, optional): Title and name used to save the plot. Defaults to None.
            save (bool, optional): Whether to save the plot. Defaults to False.
        
        Returns:
            None.
        """
        # Levels names
        levels_names = [
            'Morphology', 
            'Intensity', 
            'Texture', 
            'Linear filters - Intensity', 
            'Linear filters - Texture', 
            'Textural filters - Intensity',
            'Textural filters - Texture'
        ]

        # Initialization
        colors_sns = sns.color_palette("pastel", n_colors=5)
        colors_sns.insert(3, colors_sns[3])
        colors_sns.insert(5, colors_sns[-1])
        hatch = ['', '', '', '..', '//', '..', '//']

        # Set hatches color
        plt.rcParams['hatch.color'] = 'white'

        # Create mutliple plots for the pie charts
        fig, axes = plt.subplots(len(modalities), len(levels), figsize=(15, 10))

        # Load the models resutls
        for i, modality in enumerate(modalities):
            for j, level in enumerate(levels):
                exp_full_name = 'learn__' + experiment + '_' + level + '_' + modality
                if 'feature_importance_analysis.json' in os.listdir(path_experiments / exp_full_name):
                    fa_dict = load_json(path_experiments / exp_full_name / 'feature_importance_analysis.json')
                else:
                    fa_dict = feature_imporance_analysis(path_experiments / exp_full_name)

                # Extract percentage of features per level
                perc_levels = np.round(self.__count_percentage_levels(fa_dict), 2)
                
                # Update heatmap data
                if len(modalities) > 1:
                    axes[i, j].pie(
                        perc_levels, 
                        autopct= lambda p: '{:.1f}%'.format(p) if p > 0 else '',
                        pctdistance=0.8,
                        startangle=120, 
                        radius=1.3, 
                        rotatelabels=True, 
                        textprops={'fontsize': 14, 'weight': 'bold'},
                        colors=colors_sns,
                        hatch=hatch)
                    axes[i, j].set_title(f'{level} - {modality}', fontsize=15)
                else:
                    axes[j].pie(
                        perc_levels, 
                        autopct= lambda p: '{:.1f}%'.format(p) if p > 0 else '',
                        pctdistance=0.8,
                        startangle=120, 
                        radius=1.3, 
                        rotatelabels=True, 
                        textprops={'fontsize': 14, 'weight': 'bold'},
                        colors=colors_sns,
                        hatch=hatch)
                    axes[j].set_title(f'{level} - {modality}', fontsize=15)

        # Add legend
        plt.legend(levels_names, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 15})
        plt.tight_layout()

        # Add title
        if title:
            fig.suptitle(title, fontsize=20)
        else:
            fig.suptitle(f'{experiment}: % of selected features per level', fontsize=20)

        # Save the heatmap
        if save:
            if title:
                plt.savefig(path_experiments / f'{title}.png')
            else:
                plt.savefig(path_experiments / f'{experiment}_percentage_features.png')
        else:
            plt.show()

    def plot_models_performance(path_results: Path, dataset: str = 'test') -> None:
        """
        Plots the performance metrics of every model found in the given path.

        Args:
            path_results(Path): path to the folder containing the results of the experiment.

        Returns:
            None: Saves the plots.
        """
        # Get all tests paths
        list_path_tests =  [path for path in path_results.iterdir() if path.is_dir()]
        test_names = [path.name for path in list_path_tests]

        # Metrics to process
        metrics = ['AUC', 'AUPRC', 'BAC', 'Sensitivity', 'Specificity',
                'Precision', 'NPV', 'F1_score', 'Accuracy', 'MCC']
        
        # Organize metrics in a dataframe
        metrics_df = pd.DataFrame(columns=test_names, dtype=float)

        # Process metrics
        for metric in metrics:
            for path_test in list_path_tests:
                # Load the results
                results_dict = load_json(path_test / 'run_results.json')

                # Get the metric value
                metric_value = results_dict[list(results_dict.keys())[0]][dataset]['metrics'][metric]

                # Normalize the MCC
                if metric == 'MCC':
                    metric_value = (metric_value + 1) / 2

                # fill the dataframe
                metrics_df.loc[metric, path_test.name] = float(round(metric_value, 3))
        
        # Plot a heatmap of the metrics
        sns.heatmap(metrics_df.reindex(sorted(metrics_df.columns), axis=1), annot=True, cmap='coolwarm')
        plt.title(f"Models perfomance on the {dataset} set")
        plt.show()
    
    def plot_tree(
            self, 
            path_experiments: Path,
            path_levels_results: Path,
            experiment: str,
            experiment_levels: str,
            level: str,
            modalities: list,
            use_auc_pvalues: bool = True,
            accumulate_importance: bool = False,
            accumulation_levels: list = ["Morph", "MI", "MIT", "MITLF", "MITLFTF"],
            usage_accross_levels: list = [],
            use_times_selected: bool = True,
            levels_names : list =  ['Morph', 'Intensity', 'Texture', 'LF', 'TF'] ,
            weight_lines: float = 2,
            title: str = None,
            figsize: tuple = (20,10),
        ) -> None:
        """
        This function plots a heatmap with the performance of the models in the given experiment.

        Args:
            path_experiments (Path): Path to the folder containing the experiments.
            path_levels_results (Path): Path to the folder containing the results of the each radiomics complexity level.
            experiment (str): Name of the experiment to plot. Will be used to find the results.
            experiment_levels (str): Name of the experiment to retrieve the single radiomics levels perfomance.
            levels (List): List of radiomics levels to include in plot. For example: ['morph', 'intensity'].
            modalities (List, optional): List of imaging modalities to include in the plot. Defaults to [].
            use_auc_pvalues (bool, optional): Whether to use the AUC values or the p-values to sort the radiomics levels. 
                Defaults to True.
            accumulate_importance (bool, optional): Whether to accumulate the importance of the features accross
                the radiomics levels. Defaults to False.
            accumulation_levels (list, optional): List of the radiomics levels to accumulate the importance. Defaults to
                ["Morph", "MI", "MIT", "MITLF", "MITLFTF"].
            usage_accross_levels (list, optional): List of the number of times a complexity level was used in the
                given experiment. Defaults to [].
            use_times_selected (bool, optional): Whether to use the number of times a feature was selected across
                the different splits. Defaults to True.
            levels_names (list, optional): List of levels to  use for AUC values, basically the radiomics main levels.
                Defaults to ['Morph', 'Intensity', 'Texture', 'LF', 'TF'].
            weight_lines (float, optional): Weight applied to the lines of the tree. Defaults to 5.
            title(str, optional): Title and name used to save the plot. Defaults to None.
            figsize(tuple, optional): Size of the figure. Defaults to (20, 10).
        
        Returns:
            None.
        """
        # Checks
        if accumulate_importance:
            assert len(usage_accross_levels) == len(levels_names), "usage_accross_levels must have the same length as levels_names"

        # Fill tree data 
        for modality in modalities:
            # Initialization
            selected_feat_color = 'limegreen'
            important_lvl_color = 'r'

            # Initialization - outcome - levels
            styles_outcome_levels = ["dashed"] * 3
            colors_outcome_levels = ["black"] * 3
            width_outcome_levels = [1] * 3

            # Initialization - original - sublevels
            styles_original_levels = ["dashed"] * 3
            colors_original_levels = ["black"] * 3
            width_original_levels = [1] * 3

            # Initialization - texture-families
            styles_texture_families = ["dashed"] * 6
            colors_texture_families = ["black"] * 6
            width_texture_families = [1] * 6

            # Initialization - lf - sublevels
            styles_lf_levels = ["dashed"] * 2
            colors_lf_levels = ["black"] * 2
            width_lf_levels = [1] * 2

            # Initialization - lf-texture-families
            styles_lftexture_families = ["dashed"] * 6
            colors_lftexture_families = ["black"] * 6
            width_lftexture_families = [1] * 6

            # Initialization - tf - sublevels
            styles_tf_levels = ["dashed"] * 2
            colors_tf_levels = ["black"] * 2
            width_tf_levels = [1] * 2

            # Initialization - tf-texture-families
            styles_tftexture_families = ["dashed"] * 6
            colors_tftexture_families = ["black"] * 6
            width_tftexture_families = [1] * 6

            # Get all learning results
            if not accumulate_importance:
                levels = [level]
            else:
                levels = accumulation_levels
            
            # Loop through the levels
            for level in levels:
                # Get feature importance dict
                exp_full_name = 'learn__' + experiment + '_' + level + '_' + modality
                if 'feature_importance_analysis.json' in os.listdir(path_experiments / exp_full_name):
                    fa_dict = load_json(path_experiments / exp_full_name / 'feature_importance_analysis.json')
                else:
                    fa_dict = feature_imporance_analysis(path_experiments / exp_full_name)
                
                # Organize data
                if use_times_selected:
                    feature_data = {
                        'features': list(fa_dict.keys()),
                        'mean_importance': [fa_dict[feature]['importance_mean'] for feature in fa_dict.keys()],
                        'times_selected': [fa_dict[feature]['times_selected'] for feature in fa_dict.keys()],
                    }
                    # Convert sample to df
                    df = pd.DataFrame(feature_data)

                    # Apply weight to the lines
                    df['final_coefficient'] = (df['mean_importance'] + df['times_selected']) / 2
                else:
                    feature_data = {
                        'features': list(fa_dict.keys()),
                        'mean_importance': [fa_dict[feature]['importance_mean'] for feature in fa_dict.keys()],
                    }
                    
                    # Convert sample to df
                    df = pd.DataFrame(feature_data)

                    # Apply weight to the lines
                    df['final_coefficient'] =  df['mean_importance']
                
                # Normalize the final coefficients between 0 and 1
                df['final_coefficient'] = (df['final_coefficient'] - df['final_coefficient'].min()) \
                    / (df['final_coefficient'].max() - df['final_coefficient'].min())
                
                # Apply weight to the lines
                if accumulate_importance or use_times_selected:
                    df['final_coefficient'] *= weight_lines

                # Assing complexity level to each feature
                for i, row in df['features'].items():
                    level_name = row.split('__')[1].lower()
                    feature_name = row.split('__')[2].lower()

                    # Morph
                    if level_name.startswith('morph'):
                        # Update coefficient if accumulate_importance is True
                        if accumulate_importance:
                            df['final_coefficient'][i] /= usage_accross_levels[0]
                        # Update outcome-original connection
                        styles_outcome_levels[0] = "solid"
                        colors_outcome_levels[0] = selected_feat_color
                        width_outcome_levels[0] += df['final_coefficient'][i]

                        # Update original-morph connection
                        styles_original_levels[0] = "solid"
                        colors_original_levels[0] = selected_feat_color
                        width_original_levels[0] += df['final_coefficient'][i]

                    # Intensity
                    elif level_name.startswith('intensity'):
                        # Update coefficient if accumulate_importance is True
                        if accumulate_importance:
                            df['final_coefficient'][i] /= usage_accross_levels[1]
                        # Update outcome-original connection
                        styles_outcome_levels[0] = "solid"
                        colors_outcome_levels[0] = selected_feat_color
                        width_outcome_levels[0] += df['final_coefficient'][i]

                        # Update original-int connection
                        styles_original_levels[1] = "solid"
                        colors_original_levels[1] = selected_feat_color
                        width_original_levels[1] += df['final_coefficient'][i]

                    # Texture
                    elif level_name.startswith('texture'):
                        # Update coefficient if accumulate_importance is True
                        if accumulate_importance:
                            df['final_coefficient'][i] /= usage_accross_levels[2]
                        # Update outcome-original connection
                        styles_outcome_levels[0] = "solid"
                        colors_outcome_levels[0] = selected_feat_color
                        width_outcome_levels[0] += df['final_coefficient'][i]

                        # Update original-texture connection
                        styles_original_levels[2] = "solid"
                        colors_original_levels[2] = selected_feat_color
                        width_original_levels[2] += df['final_coefficient'][i]

                        # Update texture-families connection
                        if feature_name.startswith('_glcm'):
                            styles_texture_families[0] = "solid"
                            colors_texture_families[0] = selected_feat_color
                            width_texture_families[0] += df['final_coefficient'][i]
                        elif feature_name.startswith('_ngtdm'):
                            styles_texture_families[1] = "solid"
                            colors_texture_families[1] = selected_feat_color
                            width_texture_families[1] += df['final_coefficient'][i]
                        elif feature_name.startswith('_ngldm'):
                            styles_texture_families[2] = "solid"
                            colors_texture_families[2] = selected_feat_color
                            width_texture_families[2] += df['final_coefficient'][i]
                        elif feature_name.startswith('_glrlm'):
                            styles_texture_families[3] = "solid"
                            colors_texture_families[3] = selected_feat_color
                            width_texture_families[3] += df['final_coefficient'][i]
                        elif feature_name.startswith('_gldzm'):
                            styles_texture_families[4] = "solid"
                            colors_texture_families[4] = selected_feat_color
                            width_texture_families[4] += df['final_coefficient'][i]
                        elif feature_name.startswith('_glszm'):
                            styles_texture_families[5] = "solid"
                            colors_texture_families[5] = selected_feat_color
                            width_texture_families[5] += df['final_coefficient'][i]
                        else:
                            raise ValueError(f'Family of the feature {feature_name} not recognized')
                        
                    # Linear filters
                    elif level_name.startswith('mean') \
                        or level_name.startswith('log') \
                        or level_name.startswith('laws') \
                        or level_name.startswith('gabor') \
                        or level_name.startswith('wavelet') \
                        or level_name.startswith('coif'):

                        # Update coefficient if accumulate_importance is True
                        if accumulate_importance:
                            df['final_coefficient'][i] /= usage_accross_levels[3]
                        # Update outcome-original connection
                        styles_outcome_levels[1] = "solid"
                        colors_outcome_levels[1] = selected_feat_color
                        width_outcome_levels[1] += df['final_coefficient'][i]

                        # seperate intensity and texture then update the connections
                        if feature_name.startswith('_int'):
                            styles_lf_levels[0] = "solid"
                            colors_lf_levels[0] = selected_feat_color
                            width_lf_levels[0] += df['final_coefficient'][i]
                        elif feature_name.startswith(tuple(['_glcm', '_gldzm', '_glrlm', '_glszm', '_ngtdm', '_ngldm'])):
                            # Update lf-texture connection
                            styles_lf_levels[1] = "solid"
                            colors_lf_levels[1] = selected_feat_color
                            width_lf_levels[1] += df['final_coefficient'][i]

                        # Update lf-texture-families connection
                        if not feature_name.startswith('_int'):
                            if feature_name.startswith('_glcm'):
                                styles_lftexture_families[0] = "solid"
                                colors_lftexture_families[0] = selected_feat_color
                                width_lftexture_families[0] += df['final_coefficient'][i]
                            elif feature_name.startswith('_ngtdm'):
                                styles_lftexture_families[1] = "solid"
                                colors_lftexture_families[1] = selected_feat_color
                                width_lftexture_families[1] += df['final_coefficient'][i]
                            elif feature_name.startswith('_ngldm'):
                                styles_lftexture_families[2] = "solid"
                                colors_lftexture_families[2] = selected_feat_color
                                width_lftexture_families[2] += df['final_coefficient'][i]
                            elif feature_name.startswith('_glrlm'):
                                styles_lftexture_families[3] = "solid"
                                colors_lftexture_families[3] = selected_feat_color
                                width_lftexture_families[3] += df['final_coefficient'][i]
                            elif feature_name.startswith('_gldzm'):
                                styles_lftexture_families[4] = "solid"
                                colors_lftexture_families[4] = selected_feat_color
                                width_lftexture_families[4] += df['final_coefficient'][i]
                            elif feature_name.startswith('_glszm'):
                                styles_lftexture_families[5] = "solid"
                                colors_lftexture_families[5] = selected_feat_color
                                width_lftexture_families[5] += df['final_coefficient'][i]
                            else:
                                raise ValueError(f'Family of the feature {feature_name} not recognized')
                    # Textural filters
                    elif level_name.startswith('glcm'):
                        # Update coefficient if accumulate_importance is True
                        if accumulate_importance:
                            df['final_coefficient'][i] /= usage_accross_levels[4]
                        # Update outcome-original connection
                        styles_outcome_levels[2] = "solid"
                        colors_outcome_levels[2] = selected_feat_color
                        width_outcome_levels[2] += df['final_coefficient'][i]

                        # seperate intensity and texture then update the connections
                        if feature_name.startswith('_int'):
                            styles_tf_levels[0] = "solid"
                            colors_tf_levels[0] = selected_feat_color
                            width_tf_levels[0] += df['final_coefficient'][i]
                        elif feature_name.startswith(tuple(['_glcm', '_gldzm', '_glrlm', '_glszm', '_ngtdm', '_ngldm'])):
                            # Update tf-texture connection
                            styles_tf_levels[1] = "solid"
                            colors_tf_levels[1] = selected_feat_color
                            width_tf_levels[1] += df['final_coefficient'][i]

                        # Update tf-texture-families connection
                        if not feature_name.startswith('_int'):
                            if feature_name.startswith('_glcm'):
                                styles_tftexture_families[0] = "solid"
                                colors_tftexture_families[0] = selected_feat_color
                                width_tftexture_families[0] += df['final_coefficient'][i]
                            elif feature_name.startswith('_ngtdm'):
                                styles_tftexture_families[1] = "solid"
                                colors_tftexture_families[1] = selected_feat_color
                                width_tftexture_families[1] += df['final_coefficient'][i]
                            elif feature_name.startswith('_ngldm'):
                                styles_tftexture_families[2] = "solid"
                                colors_tftexture_families[2] = selected_feat_color
                                width_tftexture_families[2] += df['final_coefficient'][i]
                            elif feature_name.startswith('_glrlm'):
                                styles_tftexture_families[3] = "solid"
                                colors_tftexture_families[3] = selected_feat_color
                                width_tftexture_families[3] += df['final_coefficient'][i]
                            elif feature_name.startswith('_gldzm'):
                                styles_tftexture_families[4] = "solid"
                                colors_tftexture_families[4] = selected_feat_color
                                width_tftexture_families[4] += df['final_coefficient'][i]
                            elif feature_name.startswith('_glszm'):
                                styles_tftexture_families[5] = "solid"
                                colors_tftexture_families[5] = selected_feat_color
                                width_tftexture_families[5] += df['final_coefficient'][i]
                            else:
                                raise ValueError(f'Family of the feature {feature_name} not recognized')
            
            # Get AUC values and p-values for each radiomics complexity level
            if use_auc_pvalues:
                exp_full_names = ['learn__' + experiment_levels + '_' + level_name + '_' + modality for level_name in levels_names]
                auc_pvalues = np.zeros((len(exp_full_names), 2))
                for idxexp, exp_full_name in enumerate(exp_full_names):
                    if 'results_avg.json' in os.listdir(path_levels_results / exp_full_name):
                        metrics_dict = load_json(path_levels_results / exp_full_name / 'results_avg.json')
                    else:
                        metrics_dict = self.average_results(path_levels_results / exp_full_name)
                    
                    # Update AUCs and p-values
                    auc_pvalues[idxexp, 0] = metrics_dict['test']['AUC_mean']
                
                # Sort AUCs
                temp_sort = np.argsort(auc_pvalues[:, 0])
                sorted_auc_idx = temp_sort.copy()

                # Get pvalues values and sort them according to the AUCs (improvement of performance or not)
                for idxexp in range(len(sorted_auc_idx)-1):
                    p_value = self.get_ttest_p_value(
                        path_levels_results, 
                        experiment_levels, 
                        [levels_names[sorted_auc_idx[idxexp]], levels_names[sorted_auc_idx[idxexp+1]]],
                        [modality]
                    )
                    # Switch radiomics levels (less complex to more complex) if AUC is not improved
                    if p_value > 0.05 and sorted_auc_idx[idxexp] < sorted_auc_idx[idxexp+1]:
                        sorted_auc_idx[idxexp], sorted_auc_idx[idxexp+1] = sorted_auc_idx[idxexp+1], sorted_auc_idx[idxexp]
                sorted_auc_idx = sorted_auc_idx.tolist()
            
            # Determine the most important level
            index_best_importance = np.argmax(width_outcome_levels)
            
            # Update graph data with the best importance
            colors_outcome_levels[index_best_importance] = important_lvl_color

            # For original level update color for sub-levels
            if index_best_importance == 0:
                colors_original_levels[np.argmax(width_original_levels)] = important_lvl_color
            
            # For esthetic purposes
            experiment_sep = experiment.replace('_', '\n')

            # Design the graph
            G = nx.Graph()

            # Original level
            G.add_edge(experiment_sep, 'Original', color=colors_outcome_levels[0], width=width_outcome_levels[0], style=styles_outcome_levels[0])
            G.add_edge('Original', 'Morph', color=colors_original_levels[0], width=width_original_levels[0], style=styles_original_levels[0])
            G.add_edge('Original', 'Int', color=colors_original_levels[1], width=width_original_levels[1], style=styles_original_levels[1])
            G.add_edge('Original', 'Text', color=colors_original_levels[2], width=width_original_levels[2], style=styles_original_levels[2])
            G.add_edge('Text', 'GLCM', color=colors_texture_families[0], width=width_texture_families[0], style=styles_texture_families[0])
            G.add_edge('Text', 'NGTDM', color=colors_texture_families[1], width=width_texture_families[1], style=styles_texture_families[1])
            G.add_edge('Text', 'NGLDM', color=colors_texture_families[2], width=width_texture_families[2], style=styles_texture_families[2])
            G.add_edge('Text', 'GLRLM', color=colors_texture_families[3], width=width_texture_families[3], style=styles_texture_families[3])
            G.add_edge('Text', 'GLDZM', color=colors_texture_families[4], width=width_texture_families[4], style=styles_texture_families[4])
            G.add_edge('Text', 'GLSZM', color=colors_texture_families[5], width=width_texture_families[5], style=styles_texture_families[5])

            # Linear Filters level
            G.add_edge(experiment_sep, 'LF', color=colors_outcome_levels[1], width=width_outcome_levels[1], style=styles_outcome_levels[1])
            G.add_edge('LF', 'LF\nInt', color=colors_lf_levels[0], width=width_lf_levels[0], style=styles_lf_levels[0])
            G.add_edge('LF', 'LF\nText', color=colors_lf_levels[1], width=width_lf_levels[1], style=styles_lf_levels[1])
            G.add_edge('LF\nText', 'LF\nGLCM', color=colors_lftexture_families[0], width=width_lftexture_families[0], style=styles_lftexture_families[0])
            G.add_edge('LF\nText', 'LF\nNGTDM', color=colors_lftexture_families[1], width=width_lftexture_families[1], style=styles_lftexture_families[1])
            G.add_edge('LF\nText', 'LF\nNGLDM', color=colors_lftexture_families[2], width=width_lftexture_families[2], style=styles_lftexture_families[2])
            G.add_edge('LF\nText', 'LF\nGLRLM', color=colors_lftexture_families[3], width=width_lftexture_families[3], style=styles_lftexture_families[3])
            G.add_edge('LF\nText', 'LF\nGLDZM', color=colors_lftexture_families[4], width=width_lftexture_families[4], style=styles_lftexture_families[4])
            G.add_edge('LF\nText', 'LF\nGLSZM', color=colors_lftexture_families[5], width=width_lftexture_families[5], style=styles_lftexture_families[5])

            # Textural Filters level
            G.add_edge(experiment_sep, 'TF', color=colors_outcome_levels[2], width=width_outcome_levels[2], style=styles_outcome_levels[2])
            G.add_edge('TF', 'TF\nInt', color=colors_tf_levels[0], width=width_tf_levels[0], style=styles_tf_levels[0])
            G.add_edge('TF', 'TF\nText', color=colors_tf_levels[1], width=width_tf_levels[1], style=styles_tf_levels[1])
            G.add_edge('TF\nText', 'TF\nGLCM', color=colors_tftexture_families[0], width=width_tftexture_families[0], style=styles_tftexture_families[0])
            G.add_edge('TF\nText', 'TF\nNGTDM', color=colors_tftexture_families[1], width=width_tftexture_families[1], style=styles_tftexture_families[1])
            G.add_edge('TF\nText', 'TF\nNGLDM', color=colors_tftexture_families[2], width=width_tftexture_families[2], style=styles_tftexture_families[2])
            G.add_edge('TF\nText', 'TF\nGLRLM', color=colors_tftexture_families[3], width=width_tftexture_families[3], style=styles_tftexture_families[3])
            G.add_edge('TF\nText', 'TF\nGLDZM', color=colors_tftexture_families[4], width=width_tftexture_families[4], style=styles_tftexture_families[4])
            G.add_edge('TF\nText', 'TF\nGLSZM', color=colors_tftexture_families[5], width=width_tftexture_families[5], style=styles_tftexture_families[5])

            # Graph layout
            pos = graphviz_layout(G, root=experiment_sep, prog="dot")

            # Create the plot: figure and axis
            fig = plt.figure(figsize=figsize, dpi=300)
            ax = fig.add_subplot(1, 1, 1)

            # Get the attributes of the edges
            colors = nx.get_edge_attributes(G,'color').values()
            widths = nx.get_edge_attributes(G,'width').values()
            style = nx.get_edge_attributes(G,'style').values()

            # Custom color map
            cmap = np.zeros((29, 4))
            if use_auc_pvalues:
                cpalette = cm.get_cmap('Blues', 20)
                start = 12
                cmap[0, :] = cpalette(start)
                cmap[1, :] = cpalette(start + max(sorted_auc_idx.index(0), sorted_auc_idx.index(1), sorted_auc_idx.index(2)))
                cmap[2, :] = cpalette(start + sorted_auc_idx.index(0))
                cmap[3, :] = cpalette(start + sorted_auc_idx.index(1))
                cmap[4:11, :] = cpalette(start + sorted_auc_idx.index(2))
                cmap[11, :] = cpalette(start + sorted_auc_idx.index(3))
                cmap[12, :] = cpalette(start + sorted_auc_idx.index(3))
                cmap[13, :] = cpalette(start + sorted_auc_idx.index(3))
                cmap[14:20, :] = cpalette(start + sorted_auc_idx.index(3))
                cmap[20, :] = cpalette(start + sorted_auc_idx.index(4))
                cmap[21, :] = cpalette(start + sorted_auc_idx.index(4))
                cmap[22, :] = cpalette(start + sorted_auc_idx.index(4))
                cmap[23:29, :] = cpalette(start + sorted_auc_idx.index(4))
            else:
                cpalette = cm.get_cmap('winter_r', 20)
                start = 5
                cmap[0, :] = cpalette(start + 0)
                cmap[1, :] = cpalette(start + 1)
                cmap[2, :] = cpalette(start + 2)
                cmap[3, :] = cpalette(start + 3)
                cmap[4:12, :] = cpalette(start + 4)
                cmap[12, :] = cpalette(start + 5)
                cmap[13, :] = cpalette(start + 6)
                cmap[14:21, :] = cpalette(start + 7)
                cmap[21, :] = cpalette(start + 8)
                cmap[22, :] = cpalette(start + 9)
                cmap[23:29, :] = cpalette(start + 10)

            # Draw the graph
            nx.draw(
                G,
                pos=pos,
                ax=ax,
                edge_color=colors,
                width=list(widths),
                with_labels=True,
                node_color=cmap,
                node_size=1700,
                font_size=8,
                font_color='white',
                font_weight='bold',
                node_shape='o',
                style=style
            )

            # Create custom legend
            custom_legends = [
                Line2D([0], [0], color=selected_feat_color, lw=4, linestyle='solid', label='Selected (thickness reflects impact)'),
                Line2D([0], [0], color='black', lw=4, linestyle='dashed', label='Not selected'),
                Line2D([0], [0], color=important_lvl_color, lw=4, linestyle='solid', label='Level with highest impact')
            ]
            figure_keys = [
                mpatches.Patch(color='none', label='Morph: Morphological'),
                mpatches.Patch(color='none', label='Int: Intensity'),
                mpatches.Patch(color='none', label='Text: Textural'),
                mpatches.Patch(color='none', label='LF: Linear filters'),
                mpatches.Patch(color='none', label='TF: Textural filters'),
            ]

            options_legend = []
            if use_times_selected:
                options_legend.append(Line2D([], [], color='limegreen', marker='o', linestyle='None',markersize=10, label='Times a feature was selected'))
            if accumulate_importance:
                options_legend.append(Line2D([], [], color='limegreen', marker='o', linestyle='None',markersize=10, label='Accumulated importance'))

            # Set title
            if title:
                ax.set_title(title, fontsize=20)
            else:
                if use_auc_pvalues:
                    ax.set_title(f'Radiomics Optimality tree: {experiment} - {level} - {modality}', fontsize=20)
                else:
                    ax.set_title(f'Radiomics complexity tree: {experiment} - {level} - {modality}', fontsize=20)

            # Apply the custom legend
            legend = plt.legend(handles=custom_legends, loc='upper right', fontsize=15, frameon=True, title = "Legend")
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(2.0)

            # Abbrevations legend
            legend_keys = plt.legend(handles=figure_keys, loc='center left', fontsize=15, frameon=True, title = "Abbreviations", handlelength=0)
            legend_keys.get_frame().set_edgecolor('black')
            legend_keys.get_frame().set_linewidth(2.0)

            # Options legend
            if len(options_legend) > 0:
                legend_options = plt.legend(handles=options_legend, loc='upper left', fontsize=15, frameon=True, title = "Options", handlelength=0.5)
                legend_options.get_frame().set_edgecolor('black')
                legend_options.get_frame().set_linewidth(2.0)
                plt.gca().add_artist(legend_keys)
            plt.gca().add_artist(legend)

            # Apply the custom colorbar
            _, idx = np.unique(cmap, axis=0, return_index=True)
            newcmp = ListedColormap(cmap[idx[::-1], :])
            psm = ax.pcolormesh(np.array([[0, 1]]), cmap=newcmp)

            # Update the colorbar attributes
            cbar = plt.colorbar(psm, ax=ax, orientation='horizontal', fraction=0.05, pad=0.05)
            if use_auc_pvalues:
                cbar.ax.set_xlabel('Optimality (From optimal to insufficient)')
            else:
                cbar.ax.set_xlabel('From less complex to more complex')
            cbar.set_ticks([])
            
            # Tight layout
            fig.tight_layout()
            
            # Save the plot (Mandatory, since the plot is not well displayed on matplotlib)
            fig_name_ext = ''
            if accumulate_importance:
                fig_name_ext += '_acc'
            if use_auc_pvalues:
                fig_name_ext += '_auc'
            if use_times_selected:
                fig_name_ext += '_times'
            fig.savefig(path_experiments / f'{experiment}_{level}_{modality}{fig_name_ext}_tree.png', dpi=300)

    def to_json(
            self, 
            response_train: list = None, 
            response_test: list = None,
            response_holdout: list = None, 
            patients_train: list = None,
            patients_test: list = None, 
            patients_holdout: list = None
        ) -> dict:
        """
        Creates a dictionary with the results of the model using the class attributes.

        Args:
            response_train (list): List of machine learning model predictions for the training set.
            response_test (list): List of machine learning model predictions for the test set.
            patients_train (list): List of patients in the training set.
            patients_test (list): List of patients in the test set.
            patients_holdout (list): List of patients in the holdout set.
        
        Returns:
            Dict: Dictionary with the the responses of the model and the patients used for training, testing and holdout.
        """
        run_results = dict()
        run_results[self.model_id] = self.model_dict

        # Training results info
        run_results[self.model_id]['train'] = dict()
        run_results[self.model_id]['train']['patients'] = patients_train
        run_results[self.model_id]['train']['response'] = response_train.tolist() if response_train is not None else []

        # Testing results info
        run_results[self.model_id]['test'] = dict()
        run_results[self.model_id]['test']['patients'] = patients_test
        run_results[self.model_id]['test']['response'] = response_test.tolist() if response_test is not None else []

        # Holdout results info
        run_results[self.model_id]['holdout'] = dict()
        run_results[self.model_id]['holdout']['patients'] = patients_holdout
        run_results[self.model_id]['holdout']['response'] = response_holdout.tolist() if response_holdout is not None else []

        # keep a copy of the results
        self.results_dict = run_results

        return run_results
