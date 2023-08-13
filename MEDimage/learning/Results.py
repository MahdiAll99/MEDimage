import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpyencoder import NumpyEncoder
from sklearn import metrics

from MEDimage.learning.ml_utils import find_best_model
from MEDimage.utils.json_utils import load_json, save_json


class Results:
    def __init__(self, model_dict: dict = {}, model_id: str = "") -> None:
        """
        Constructor of the class Results
        """
        self.model_dict = model_dict
        self.model_id = model_id
        self.results_dict = {}

    def __calculate_performance(self, 
                                response: list, 
                                labels: pd.DataFrame, 
                                thresh: float) -> dict:
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

    def __get_metrics_failure_dict(self, metrics: list = ['AUC', 'Sensitivity', 'Specificity', 
                                                          'BAC', 'AUPRC', 'Precision', 
                                                          'NPV', 'Accuracy', 'F1_score', 'MCC',
                                                          'TP', 'TN', 'FP', 'FN']) -> dict:
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
            stat: str = 'mean',
            problems: List = [], 
            modalities: List = [], 
            metric: str = 'AUC',
            save: bool = False
        ) -> None:
        # Make sure the metric is in the list of metrics
        list_metrics = [
            'AUC', 'Sensitivity', 'Specificity', 
            'BAC', 'AUPRC', 'Precision', 
            'NPV', 'Accuracy', 'F1_score', 'MCC',
            'TP', 'TN', 'FP', 'FN'
        ]
        assert metric in list_metrics, f'Given metric {metric} is not in the list of metrics. Please choose from {list_metrics}'

        # Make sure the chosen stat is in the list of stats
        list_stats = ['mean', 'std', 'max', 'min', '2.5%', '97.5%']
        assert stat in list_stats, f'Given stat {stat} is not in the list of stats. Please choose from {list_stats}'

        # Prepare the data for the heatmap
        if problems:            
            # Load the models resutls
            results_dicts = []
            patients_count = dict.fromkeys(problems)
            for problem in problems:
                for level in levels:
                    exp_full_name = 'learn__' + experiment + '_' + level + '_' + problem
                    results_dict = self.average_results(path_experiments / exp_full_name)
                    results_dicts.append(results_dict)

                    # Patient count
                patients_count[problem] = self.count_patients(path_experiments / exp_full_name)

            # Update problem name to include the number of patients
            problems = [problem + f' ({patients_count[problem]["train"]} train, {patients_count[problem]["test"]} test)' for problem in problems]

            # Create the heatmap data using the metric of interest
            heatmap_data = np.zeros((len(problems), len(levels)))
            for i in range(len(problems)):
                for j in range(len(levels)):
                    results_dict = results_dicts[i * len(levels) + j]
                    heatmap_data[i, j] = results_dict['test'][metric + '_' + stat]
            
            # Convert the numpy array to a DataFrame for Seaborn
            df = pd.DataFrame(heatmap_data, columns=levels, index=problems)
        
        elif modalities:
            # Load the models resutls
            results_dicts = []
            patients_count = dict.fromkeys(modalities)
            for modality in modalities:
                for level in levels:
                    exp_full_name = 'learn__' + experiment + '_' + level + '_' + modality
                    results_dict = self.average_results(path_experiments / exp_full_name)
                    results_dicts.append(results_dict)

                # Patient count
                patients_count[modality] = self.count_patients(path_experiments / exp_full_name)

            # Update modality name to include the number of patients
            modalities = [modality + f' ({patients_count[modality]["train"]} train, {patients_count[modality]["test"]} test)' for modality in modalities]
            
            # Create the heatmap data using the metric of interest
            heatmap_data = np.zeros((len(modalities), len(levels)))
            for i in range(len(modalities)):
                for j in range(len(levels)):
                    results_dict = results_dicts[i * len(levels) + j]
                    heatmap_data[i, j] = results_dict['test'][metric + '_' + stat]
            
            # Convert the numpy array to a DataFrame for Seaborn
            df = pd.DataFrame(heatmap_data, columns=levels, index=modalities)

        else:
            # Load the models resutls
            results_dicts = []
            for level in levels:
                exp_full_name = 'learn__' + experiment + '_' + level
                results_dict = self.average_results(path_experiments / exp_full_name)
                results_dicts.append(results_dict)
            
            # Patient count
            patients_count = self.count_patients(path_experiments / exp_full_name)

            # Update modality name to include the number of patients
            index = [f'({patients_count["train"]} train, {patients_count["test"]} test)']

            # Create the heatmap data using the metric of interest
            heatmap_data = np.zeros((1, len(results_dicts)))
            for j, results_dict in enumerate(results_dicts):
                    heatmap_data[0, j] = results_dict['test'][metric + '_' + stat]
            
            # Convert the numpy array to a DataFrame for Seaborn
            df = pd.DataFrame(heatmap_data, columns=levels, index=index)
            
        # Count the patients used in learning
        patients_count = self.count_patients(path_experiments / exp_full_name)

        # Create the heatmap using seaborn
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.2)
        if metric == 'MCC':
            sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", cbar=True, linewidths=0.5, vmin=-1, vmax=1)
        else:
            sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", cbar=True, linewidths=0.5, vmin=0, vmax=1)
        
        plt.title("Testing Results Heatmap")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.title(f'{metric}_{stat} heatmap')
        plt.show()

        # Save the heatmap
        if save:
            plt.savefig(path_experiments / f'{metric}_{stat}_heatmap.png')

    def plot_models_performance(path_results: Path) -> None:
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
                'Precision', 'NPV', 'F1_score', 'Accuracy', 'MCC',
                'TN', 'FP', 'FN', 'TP']
        
        # Organize metrics in a dataframe
        metrics_df = pd.DataFrame(columns=test_names)

        # Process metrics
        for dataset in ['train', 'test', 'holdout']:
            for metric in metrics:
                metric_values = []
                for path_test in list_path_tests:
                    results_dict = load_json(path_test / 'run_results.json')
                    metric_values.append(results_dict[list(results_dict.keys())[0]][dataset]['metrics'][metric])

                    # fill the dataframe
                    metrics_df.loc[metric, path_test.name] = metric_values[-1]
        
        # Plot a heatmap of the metrics
        sns.heatmap(metrics_df, annot=True, cmap='coolwarm')
        plt.show()

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
