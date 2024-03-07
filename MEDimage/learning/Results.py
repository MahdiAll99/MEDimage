# Description: Class Results to store and analyze the results of experiments.

import os
from pathlib import Path
from typing import List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from networkx.drawing.nx_pydot import graphviz_layout
from numpyencoder import NumpyEncoder
from sklearn import metrics

from MEDimage.learning.ml_utils import feature_imporance_analysis, list_metrics
from MEDimage.learning.Stats import Stats
from MEDimage.utils.json_utils import load_json, save_json
from MEDimage.utils.texture_features_names import *


class Results:
    """
    A class to analyze the results of a given machine learning experiment, including the assessment of the model's performance,

    Args:
        model_dict (dict, optional): Dictionary containing the model's parameters. Defaults to {}.
        model_id (str, optional): ID of the model. Defaults to "".

    Attributes:
        model_dict (dict): Dictionary containing the model's parameters.
        model_id (str): ID of the model.
        results_dict (dict): Dictionary containing the results of the model's performance.
    """
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
            response (list): List of the probabilities of class "1" for all instances (prediction)
            labels (pd.Dataframe): Column vector specifying the outcome status (1 or 0) for all instances.
            thresh (float): Optimal threshold selected from the ROC curve.

        Returns:
            Dict: Dictionary containing the performance metrics.
        """
        # Recording results
        results_dict = dict()

        # Removing Nans
        df = labels.copy()
        outcome_name = labels.columns.values[0]
        df['response'] = response
        df.dropna(axis=0, how='any', inplace=True)

        # Confusion matrix elements:
        results_dict['TP'] = ((df['response'] >= thresh) & (df[outcome_name] == 1)).sum()
        results_dict['TN'] = ((df['response'] < thresh) & (df[outcome_name] == 0)).sum()
        results_dict['FP'] = ((df['response'] >= thresh) & (df[outcome_name] == 0)).sum()
        results_dict['FN'] = ((df['response'] < thresh) & (df[outcome_name] == 1)).sum()
        
        # Copying confusion matrix elements
        TP = results_dict['TP']
        TN = results_dict['TN']
        FP = results_dict['FP']
        FN = results_dict['FN']

        # AUC
        results_dict['AUC'] = metrics.roc_auc_score(df[outcome_name], df['response'])

        # AUPRC
        results_dict['AUPRC'] = metrics.average_precision_score(df[outcome_name], df['response'])

        # Sensitivity
        try:
            results_dict['Sensitivity'] = TP / (TP + FN)
        except:
            print('TP + FN = 0, Division by 0, replacing sensitivity by 0.0')
            results_dict['Sensitivity'] = 0.0

        # Specificity
        try:
            results_dict['Specificity'] = TN / (TN + FP)
        except:
            print('TN + FP= 0, Division by 0, replacing specificity by 0.0')
            results_dict['Specificity'] = 0.0

        # Balanced accuracy
        results_dict['BAC'] = (results_dict['Sensitivity'] + results_dict['Specificity']) / 2

        # Precision
        results_dict['Precision'] = TP / (TP + FP)

        # NPV (Negative Predictive Value)
        results_dict['NPV'] = TN / (TN + FN)

        # Accuracy
        results_dict['Accuracy'] = (TP + TN) / (TP + TN + FP + FN)

        # F1 score
        results_dict['F1_score'] = 2 * TP / (2 * TP + FP + FN)

        # mcc (mathews correlation coefficient)
        results_dict['MCC'] = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

        return results_dict
    
    def __get_metrics_failure_dict(
            self, 
            metrics: list = list_metrics
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
        failure_struct = {metric: np.nan for metric in metrics}

        return failure_struct
    
    def __count_percentage_levels(self, features_dict: dict, fda: bool = False) -> list:
        """
        Counts the percentage of each radiomics level in a given features dictionary.

        Args:
            features_dict (dict): Dictionary of features.
            fda (bool, optional): If True, meaning the features are from the FDA logging dict and will be
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

    def __count_patients(self, path_results: Path) -> dict:
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
        
        # Retrieve metrics
        for dataset in ['train', 'test', 'holdout']:
            dataset_dict = results_avg[dataset]
            for metric in list_metrics:
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
    
    def get_model_performance(
            self, 
            response: list, 
            outcome_table: pd.DataFrame
        ) -> None:
        """
        Calculates the performance of the model
        Args:
            response (list): List of machine learning model predictions.
            outcome_table (pd.DataFrame): Outcome table with binary labels.
        
        Returns:
            None: Updates the ``run_results`` attribute.
        """
        # Calculating performance metrics for the training set
        try:
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
            
            # Calculating performance
            results_dict = self.__calculate_performance(response, outcome_binary.to_frame(), self.model_dict['threshold'])

            return results_dict
        
        except Exception as e:
            print(f"Error: ", e, "filling metrics with nan...")
            return self.__get_metrics_failure_dict()
    
    def get_optimal_level(
            self, 
            path_experiments: Path, 
            experiment: str,
            modalities: List, 
            levels: List,
            metric: str = 'AUC_mean',
            p_value_test: str = 'wilcoxon',
            aggregate: bool = False,
        ) -> None:
        """
        This function plots a heatmap of the metrics values for the performance of the models in the given experiment.

        Args:
            path_experiments (Path): Path to the folder containing the experiments.
            experiment (str): Name of the experiment to plot. Will be used to find the results.
            modalities (List): List of imaging modalities to include in the plot.
            levels (List): List of radiomics levels to include in plot. For example: ['morph', 'intensity'].
                You can also use list of variants to plot the best variant for each level. For example: [['morph', 'morph5'], 'intensity'].
            metric (str, optional): Metric to plot. Defaults to 'AUC_mean'.
            p_value_test (str, optional): Method to use to calculate the p-value. Defaults to 'wilcoxon'.
                Available options:
                    
                    - 'delong': Delong test.
                    - 'ttest': T-test.
                    - 'wilcoxon': Wilcoxon signed rank test.
                    - 'bengio': Bengio and Nadeau corrected t-test.
            aggregate (bool, optional): If True, aggregates the results of all the splits and computes one final p-value.
                Only valid for the Delong test when cross-validation is used. Defaults to False.
        
        Returns:
            None.
        """
        assert metric.split('_')[0] in list_metrics, f'Given metric {list_metrics} is not in the list of metrics. Please choose from {list_metrics}'
        
        # Initialization
        optimal_lvls = [""] * len(modalities)

        # Prepare the data for the heatmap
        for idx_m, modality in enumerate(modalities):
            best_levels = []
            results_dict_best = dict()
            results_dicts = []
            best_exp = ""
            for level in levels:
                metric_compare = -1.0
                if type(level) != list:
                    level = [level]
                for variant in level:
                    exp_full_name = 'learn__' + experiment + '_' + variant + '_' + modality
                    if 'results_avg.json' in os.listdir(path_experiments / exp_full_name):
                        results_dict = load_json(path_experiments / exp_full_name / 'results_avg.json')
                    else:
                        results_dict = self.average_results(path_experiments / exp_full_name)
                    if metric_compare < results_dict['test'][metric]:
                        metric_compare = results_dict['test'][metric]
                        results_dict_best = results_dict
                        best_exp = variant
                best_levels.append(best_exp)
                results_dicts.append(results_dict_best)

            # Create the heatmap data using the metric of interest
            heatmap_data = np.zeros((2, len(best_levels)))

            # Fill the heatmap data
            for j in range(len(best_levels)):
                # Get metrics and p-values
                results_dict = results_dicts[j]
                if aggregate and 'delong' in p_value_test:
                    metric_stat = round(Stats.get_aggregated_metric(
                        path_experiments, 
                        experiment, 
                        best_levels[j], 
                        modality,
                        metric.split('_')[0] if '_' in metric else metric
                    ), 2)
                else:
                    metric_stat = round(results_dict['test'][metric], 2)
                heatmap_data[0, j] = metric_stat
            
            # Statistical analysis
            # Initializations
            optimal_lvls[idx_m] = best_levels[0]
            init_metric = heatmap_data[0][0]
            idx_d = 0
            start_level = 0

            # Get p-values for all the levels
            while idx_d < len(best_levels) - 1:
                metric_val = heatmap_data[0][idx_d+1]
                # Get p-value only if the metric is improving
                if metric_val > init_metric:
                    # Instantiate the Stats class
                    stats = Stats(
                        path_experiments, 
                        experiment, 
                        [best_levels[start_level], best_levels[idx_d+1]], 
                        [modality]
                    )

                    # Get p-value
                    p_value = stats.get_p_value(
                        p_value_test,
                        metric=metric if '_' not in metric else metric.split('_')[0],
                        aggregate=aggregate
                    )
                    
                    # If p-value is less than 0.05, change starting level
                    if p_value <= 0.05:
                        optimal_lvls[idx_m] = best_levels[idx_d+1]
                        init_metric = metric_val
                        start_level = idx_d + 1

                # Go to next column
                idx_d += 1

        return optimal_lvls
    
    def plot_features_importance_histogram(
            self, 
            path_experiments: Path, 
            experiment: str,
            level: str,
            modalities: List,
            sort_option: str = 'importance',
            title: str = None,
            save: bool = True,
            figsize: tuple = (12, 12)
        ) -> None:
        """
        Plots a histogram of the features importance for the given experiment.

        Args:
            path_experiments (Path): Path to the folder containing the experiments.
            experiment (str): Name of the experiment to plot. Will be used to find the results.
            level (str): Radiomics level to plot. For example: 'morph'.
            modalities (List): List of imaging modalities to use for the plot. A plot for each modality.
            sort_option (str, optional): Option used to sort the features. Available options:
                - 'importance': Sorts the features by importance.
                - 'times_selected': Sorts the features by the number of times they were selected across the different splits.
                - 'both': Sorts the features by importance and then by the number of times they were selected.
            title (str, optional): Title of the plot. Defaults to None.
            save (bool, optional): Whether to save the plot. Defaults to True.
            figsize (tuple, optional): Size of the figure. Defaults to (12, 12).

        Returns:
            None. Plots the figure or saves it.
        """

        # checks 
        assert sort_option in ['importance', 'times_selected', 'both'], \
            f'sort_option must be either "importance", "times_selected" or "both". Given: {sort_option}'

        # For each modality, load features importance dict
        for modality in modalities:
            exp_full_name = 'learn__' + experiment + '_' + level + '_' + modality

            # Load features importance dict
            if 'feature_importance_analysis.json' in os.listdir(path_experiments / exp_full_name):
                feat_imp_dict = load_json(path_experiments / exp_full_name / 'feature_importance_analysis.json')
            else:
                raise FileNotFoundError(f'feature_importance_analysis.json not found in {path_experiments / exp_full_name}')

            # Organize the data in a dataframe
            keys = list(feat_imp_dict.keys())
            mean_importances = []
            times_selected = []
            for key in keys:
                times_selected
                mean_importances.append(feat_imp_dict[key]['importance_mean'])
                times_selected.append(feat_imp_dict[key]['times_selected'])
            df = pd.DataFrame({'feature': keys, 'importance': mean_importances, 'times_selected': times_selected})
            df = df.sort_values(by=[sort_option], ascending=True)

            # Plot the histogram
            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            if sort_option == 'importance':
                color = 'deepskyblue'
            else:
                color = 'darkorange'
            plt.figure(figsize=figsize)
            plt.xlabel(sort_option)
            plt.ylabel('Features')
            plt.barh(df['feature'], df[sort_option], color=color)

            # Add title
            if title:
                plt.title(title, weight='bold')
            else:
                plt.title(f'Features importance histogram \n {experiment} - {level} - {modality}', weight='bold')
            plt.tight_layout()
            
            # Save the plot
            if save:
                plt.savefig(path_experiments / f'features_importance_histogram_{level}_{modality}_{sort_option}.png')
            else:
                plt.show()
    
    def plot_heatmap(
            self, 
            path_experiments: Path, 
            experiment: str,
            modalities: List, 
            levels: List,
            metric: str = 'AUC_mean',
            stat_extra: list = [],
            plot_p_values: bool = True,
            p_value_test: str = 'wilcoxon',
            aggregate: bool = False,
            title: str = None,
            save: bool = False,
            figsize: tuple = (8, 8)
        ) -> None:
        """
        This function plots a heatmap of the metrics values for the performance of the models in the given experiment.

        Args:
            path_experiments (Path): Path to the folder containing the experiments.
            experiment (str): Name of the experiment to plot. Will be used to find the results.
            modalities (List): List of imaging modalities to include in the plot.
            levels (List): List of radiomics levels to include in plot. For example: ['morph', 'intensity'].
                You can also use list of variants to plot the best variant for each level. For example: [['morph', 'morph5'], 'intensity'].
            metric (str, optional): Metric to plot. Defaults to 'AUC_mean'.
            stat_extra (list, optional): List of extra statistics to include in the plot. Defaults to [].
            plot_p_values (bool, optional): If True plots the p-value of the choosen test. Defaults to True.
            p_value_test (str, optional): Method to use to calculate the p-value. Defaults to 'wilcoxon'. Available options:
                    
                    - 'delong': Delong test.
                    - 'ttest': T-test.
                    - 'wilcoxon': Wilcoxon signed rank test.
                    - 'bengio': Bengio and Nadeau corrected t-test.
            aggregate (bool, optional): If True, aggregates the results of all the splits and computes one final p-value.
                Only valid for the Delong test when cross-validation is used. Defaults to False.
            extra_xlabels (List, optional): List of extra x-axis labels. Defaults to [].
            title (str, optional): Title of the plot. Defaults to None.
            save (bool, optional): Whether to save the plot. Defaults to False.
            figsize (tuple, optional): Size of the figure. Defaults to (8, 8).
        
        Returns:
            None.
        """
        assert metric.split('_')[0] in list_metrics, f'Given metric {list_metrics} is not in the list of metrics. Please choose from {list_metrics}'

        # Prepare the data for the heatmap
        fig, axs = plt.subplots(len(modalities), figsize=figsize)
        for idx_m, modality in enumerate(modalities):
            # Initializations
            best_levels = []
            results_dict_best = dict()
            results_dicts = []
            best_exp = ""
            patients_count = dict.fromkeys([modality])

            # Loop over the levels and find the best variant for each level
            for level in levels:
                metric_compare = -1.0
                if type(level) != list:
                    level = [level]
                for idx, variant in enumerate(level):
                    exp_full_name = 'learn__' + experiment + '_' + variant + '_' + modality
                    if 'results_avg.json' in os.listdir(path_experiments / exp_full_name):
                        results_dict = load_json(path_experiments / exp_full_name / 'results_avg.json')
                    else:
                        results_dict = self.average_results(path_experiments / exp_full_name)
                    if metric_compare < results_dict['test'][metric]:
                        metric_compare = results_dict['test'][metric]
                        results_dict_best = results_dict
                        best_exp = variant
                best_levels.append(best_exp)
                results_dicts.append(results_dict_best)
            
            # Patient count
            patients_count[modality] = self.__count_patients(path_experiments / exp_full_name)

            # Create the heatmap data using the metric of interest
            if plot_p_values:
                heatmap_data = np.zeros((2, len(best_levels)))
            else:
                heatmap_data = np.zeros((1, len(best_levels)))

            # Fill the heatmap data
            labels = heatmap_data.tolist()
            labels_draw = heatmap_data.tolist()
            heatmap_data_draw = heatmap_data.tolist()
            for j in range(len(best_levels)):
                # Get metrics and p-values
                results_dict = results_dicts[j]
                if aggregate and 'delong' in p_value_test:
                    metric_stat = round(Stats.get_aggregated_metric(
                        path_experiments, 
                        experiment, 
                        best_levels[j], 
                        modality,
                        metric.split('_')[0] if '_' in metric else metric
                    ), 2)
                else:
                    metric_stat = round(results_dict['test'][metric], 2)
                if plot_p_values:
                    heatmap_data[0, j] = metric_stat
                else:
                    heatmap_data[1, j] = metric_stat
                
                # Extra statistics
                if stat_extra:
                    if plot_p_values:
                        labels[0][j] = f'{metric_stat}'
                        if j < len(best_levels) - 1:
                            labels[1][j+1] = f'{round(heatmap_data[1, j+1], 5)}'
                            labels[1][0] = '-'
                        for extra_stat in stat_extra:
                            if aggregate and ('sensitivity' in extra_stat.lower() or 'specificity' in extra_stat.lower()):
                                extra_metric_stat = round(Stats.get_aggregated_metric(
                                    path_experiments, 
                                    experiment, 
                                    best_levels[j], 
                                    modality,
                                    extra_stat.split('_')[0]
                                ), 2)
                                extra_stat = extra_stat.split('_')[0] + '_agg' if '_' in extra_stat else extra_stat
                                labels[0][j] += f'\n{extra_stat}: {extra_metric_stat}'
                            else:
                                extra_metric_stat = round(results_dict['test'][extra_stat], 2)
                                labels[0][j] += f'\n{extra_stat}: {extra_metric_stat}'
                    else:
                        labels[0][j] = f'{metric_stat}'
                        for extra_stat in stat_extra:
                            extra_metric_stat = round(results_dict['test'][extra_stat], 2)
                            labels[0][j] += f'\n{extra_stat}: {extra_metric_stat}'
                else:
                    labels = np.array(heatmap_data).round(4).tolist()
            
            # Update modality name to include the number of patients for training and testing
            modalities_label = [modality + f' ({patients_count[modality]["train"]} train, {patients_count[modality]["test"]} test)']

            # Data to draw
            heatmap_data_draw = heatmap_data.copy()
            labels_draw = labels.copy()
            labels_draw[1] = [''] * len(labels[1])
            heatmap_data_draw[1] = np.array([-1] * heatmap_data[1].shape[0]) if 'MCC' in metric else np.array([0] * heatmap_data[1].shape[0])
            
            # Set up the rows (modalities and p-values)
            if plot_p_values:
                modalities_temp = modalities_label.copy()
                modalities_label = ['p-values'] * len(modalities_temp) * 2
                for idx in range(len(modalities_label)):
                    if idx % 2 == 0:
                        modalities_label[idx] = modalities_temp[idx // 2]
            
            # Convert the numpy array to a DataFrame for Seaborn
            df = pd.DataFrame(heatmap_data_draw, columns=best_levels, index=modalities_label)

            # To avoid bugs, convert axs to list if only one modality is used
            if len(modalities) == 1:
                axs = [axs]

            # Create the heatmap using seaborn
            sns.heatmap(
                df, 
                annot=labels_draw, 
                ax=axs[idx_m],
                fmt="", 
                cmap="Blues", 
                cbar=True, 
                linewidths=0.5, 
                vmin=-1 if 'MCC' in metric else 0, 
                vmax=1, 
                annot_kws={"weight": "bold", "fontsize": 8}
            )            
            
            # Plot p-values
            if plot_p_values:
                # Initializations
                extent_x = axs[idx_m].get_xlim()
                step_x = 1
                start_x = extent_x[0] + 0.5
                end_x = start_x + step_x
                step_y = 1 / extent_x[1]
                start_y = 1
                endpoints_x = []
                endpoints_y = []
                init_metric = heatmap_data[0][0]
                idx_d = 0
                start_level = 0

                # p-values for all levels
                while idx_d < len(best_levels) - 1:
                    # Retrieve the metric value
                    metric_val = heatmap_data[0][idx_d+1]

                    # Instantiate the Stats class
                    stats = Stats(
                        path_experiments, 
                        experiment, 
                        [best_levels[start_level], best_levels[idx_d+1]], 
                        [modality]
                    )

                    # Get p-value only if the metric is improving
                    if metric_val > init_metric:
                        p_value = stats.get_p_value(
                            p_value_test,
                            metric=metric if '_' not in metric else metric.split('_')[0],
                            aggregate=aggregate
                        )
                        
                        # round the pvalue
                        p_value = round(p_value, 3)

                        # Set color, red if p-value > 0.05, green otherwise
                        color = 'r' if p_value > 0.05 else 'g'

                        # Plot the p-value (line and value)
                        axs[idx_m].axhline(start_y + step_y, xmin=start_x/extent_x[1], xmax=end_x/extent_x[1], color=color)
                        axs[idx_m].text(start_x + step_x/2, start_y + step_y, p_value, va='center', color=color, ha='center', backgroundcolor='w')
                        
                        # Plot endpoints
                        endpoints_x = [start_x, end_x]
                        endpoints_y = [start_y + step_y, start_y + step_y]
                        axs[idx_m].scatter(endpoints_x, endpoints_y, color=color)
                        
                        # Move to next line
                        step_y += 1 / extent_x[1]
                        
                        # If p-value is less than 0.05, change starting level
                        if p_value <= 0.05:
                            init_metric = metric_val
                            start_x = end_x
                            start_level = idx_d + 1

                    # Go to next column
                    end_x += step_x
                    idx_d += 1

            # Rotate xticks
            axs[idx_m].set_xticks(axs[idx_m].get_xticks(), best_levels, rotation=45)
        
        # Set title
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(f'{metric} heatmap')

        # Tight layout
        fig.tight_layout()

        # Save the heatmap
        if save:
            if title:
                fig.savefig(path_experiments / f'{title}.png')
            else:
                fig.savefig(path_experiments / f'{metric}_heatmap.png')
        else:
            fig.show()
    
    def plot_radiomics_starting_percentage(
            self, 
            path_experiments: Path, 
            experiment: str,
            levels: List,
            modalities: List, 
            title: str = None,
            figsize: tuple = (15, 10),
            save: bool = False
        ) -> None:
        """
        This function plots a pie chart of the percentage of features used in experiment per radiomics level.

        Args:
            path_experiments (Path): Path to the folder containing the experiments.
            experiment (str): Name of the experiment to plot. Will be used to find the results.
            levels (List): List of radiomics levels to include in the plot.
            modalities (List): List of imaging modalities to include in the plot.
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
                # Use the first test folder to get the results dict
                if 'test__001' in os.listdir(path_experiments / exp_full_name):
                    run_results_dict = load_json(path_experiments / exp_full_name / 'test__001' / 'run_results.json')
                else:
                    raise FileNotFoundError(f'no test file named test__001 in {path_experiments / exp_full_name}')

                # Extract percentage of features per level
                perc_levels = np.round(self.__count_percentage_radiomics(run_results_dict), 2)
                
                # Plot the pie chart of the percentages
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
            modalities: List, 
            title: str = None,
            save: bool = False
        ) -> None:
        """
        This function plots a heatmap of the percentage of stable features and final features selected by FDA for a given experiment.

        Args:
            path_experiments (Path): Path to the folder containing the experiments.
            experiment (str): Name of the experiment to plot. Will be used to find the results.
            levels (List): List of radiomics levels to include in plot. For example: ['morph', 'intensity'].
            modalities (List): List of imaging modalities to include in the plot.
            title(str, optional): Title and name used to save the plot. Defaults to None.
            save (bool, optional): Whether to save the plot. Defaults to False.
        
        Returns:
            None.
        """
        # Initialization - Levels names
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

        # Initialization - Colors
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
                
                # Plot pie chart of stable features
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

                # Legends
                legends = [f'{level} - {perc_levels_stable[idx]}' for idx, level in enumerate(level_names_stable)]
                axes[i*2, j].legend(legends, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 13})
                
                # Plot pie chart of the final features selected
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
        This function plots a pie chart of the percentage of the final features used to train the model per radiomics level.

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
                
                # Plot the pie chart of percentages for the final features
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

    def plot_original_level_tree(
            self, 
            path_experiments: Path,
            experiment: str,
            level: str,
            modalities: list,
            initial_width: float = 4,
            lines_weight: float = 1,
            title: str = None,
            figsize: tuple = (12,10),
        ) -> None:
        """
        Plots a tree explaining the impact of features in the original radiomics complexity level.

        Args:
            path_experiments (Path): Path to the folder containing the experiments.
            experiment (str): Name of the experiment to plot. Will be used to find the results.
            level (List): Radiomics complexity level to use for the plot.
            modalities (List, optional): List of imaging modalities to include in the plot. Defaults to [].
            initial_width (float, optional): Initial width of the lines. Defaults to 1. For aesthetic purposes.
            lines_weight (float, optional): Weight applied to the lines of the tree. Defaults to 2. For aesthetic purposes.
            title(str, optional): Title and name used to save the plot. Defaults to None.
            figsize(tuple, optional): Size of the figure. Defaults to (20, 10).
        
        Returns:
            None.
        """
        # Fill tree data for each modality
        for modality in modalities:
            # Initialization
            selected_feat_color = 'limegreen'
            optimal_lvl_color = 'darkorange'

            # Initialization - outcome - levels
            styles_outcome_levels = ["dashed"] * 3
            colors_outcome_levels = ["black"] * 3
            width_outcome_levels = [initial_width] * 3

            # Initialization - original - sublevels
            styles_original_levels = ["dashed"] * 3
            colors_original_levels = ["black"] * 3
            width_original_levels = [initial_width] * 3

            # Initialization - texture-families
            styles_texture_families = ["dashed"] * 6
            colors_texture_families = ["black"] * 6
            width_texture_families = [initial_width] * 6
            families_names = ["glcm", "ngtdm", "ngldm", "glrlm", "gldzm", "glszm"]
            
            # Get feature importance dict
            exp_full_name = 'learn__' + experiment + '_' + level + '_' + modality
            if 'feature_importance_analysis.json' in os.listdir(path_experiments / exp_full_name):
                fa_dict = load_json(path_experiments / exp_full_name / 'feature_importance_analysis.json')
            else:
                fa_dict = feature_imporance_analysis(path_experiments / exp_full_name)
            
            # Organize data
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
        
            # Applying the lines weight
            df['final_coefficient'] *= lines_weight

            # Assign complexity level to each feature
            for i, row in df['features'].items():
                level_name = row.split('__')[1].lower()
                family_name = row.split('__')[2].lower()

                # Morph
                if level_name.startswith('morph'):
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
                    # Update outcome-original connection
                    styles_outcome_levels[0] = "solid"
                    colors_outcome_levels[0] = selected_feat_color
                    width_outcome_levels[0] += df['final_coefficient'][i]

                    # Update original-texture connection
                    styles_original_levels[2] = "solid"
                    colors_original_levels[2] = selected_feat_color
                    width_original_levels[2] += df['final_coefficient'][i]
              
            # Determine the most important level
            index_best_level = np.argmax(width_outcome_levels)
            colors_outcome_levels[index_best_level] = optimal_lvl_color

            # Update color for the best sub-level
            colors_original_levels[np.argmax(width_original_levels)] = optimal_lvl_color
            
            # If texture features are the optimal
            if np.argmax(width_original_levels) == 2:
                for i, row in df['features'].items():
                    level_name = row.split('__')[1].lower()
                    family_name = row.split('__')[2].lower()

                    # Update texture-families connection
                    if level_name.startswith('texture'):
                        if family_name.startswith('_glcm'):
                            styles_texture_families[0] = "solid"
                            colors_texture_families[0] = selected_feat_color
                            width_texture_families[0] += df['final_coefficient'][i]
                        elif family_name.startswith('_ngtdm'):
                            styles_texture_families[1] = "solid"
                            colors_texture_families[1] = selected_feat_color
                            width_texture_families[1] += df['final_coefficient'][i]
                        elif family_name.startswith('_ngldm'):
                            styles_texture_families[2] = "solid"
                            colors_texture_families[2] = selected_feat_color
                            width_texture_families[2] += df['final_coefficient'][i]
                        elif family_name.startswith('_glrlm'):
                            styles_texture_families[3] = "solid"
                            colors_texture_families[3] = selected_feat_color
                            width_texture_families[3] += df['final_coefficient'][i]
                        elif family_name.startswith('_gldzm'):
                            styles_texture_families[4] = "solid"
                            colors_texture_families[4] = selected_feat_color
                            width_texture_families[4] += df['final_coefficient'][i]
                        elif family_name.startswith('_glszm'):
                            styles_texture_families[5] = "solid"
                            colors_texture_families[5] = selected_feat_color
                            width_texture_families[5] += df['final_coefficient'][i]
                        else:
                            raise ValueError(f'Family of the feature {family_name} not recognized')
                
                # Update color
                colors_texture_families[np.argmax(width_texture_families)] = optimal_lvl_color

                # Find best texture family to continue path
                best_family_name = ""
                index_best_family = np.argmax(width_texture_families)
                best_family_name = families_names[index_best_family]
                features_names = texture_features_all[index_best_family]

                # Update texture-families-features connection
                width_texture_families_feature = [initial_width] * len(features_names)
                colors_texture_families_feature = ["black"] * len(features_names)
                styles_texture_families_feature = ["dashed"] * len(features_names)
                for i, row in df['features'].items():
                    level_name = row.split('__')[1].lower()
                    family_name = row.split('__')[2].lower()
                    feature_name = row.split('__')
                    if level_name.startswith('texture') and family_name.startswith('_' + best_family_name):
                        for feature in features_names:
                            if feature in feature_name:
                                colors_texture_families_feature[features_names.index(feature)] = selected_feat_color
                                styles_texture_families_feature[features_names.index(feature)] = "solid"
                                width_texture_families_feature[features_names.index(feature)] += df['final_coefficient'][i]
                                break
                
                # Update color for the best texture family
                colors_texture_families_feature[np.argmax(width_texture_families_feature)] = optimal_lvl_color
            
            # For esthetic purposes
            experiment_sep = experiment.replace('_', '\n')

            # Design the graph
            G = nx.Graph()

            # Original level
            G.add_edge(experiment_sep, 'Original', color=optimal_lvl_color, width=np.sum(width_original_levels), style="solid")
            if styles_original_levels[0] == "solid":
                G.add_edge('Original', 'Morph', color=colors_original_levels[0], width=width_original_levels[0], style=styles_original_levels[0])
            if styles_original_levels[1] == "solid":
                G.add_edge('Original', 'Int', color=colors_original_levels[1], width=width_original_levels[1], style=styles_original_levels[1])
            if styles_original_levels[2] == "solid":
                G.add_edge('Original', 'Text', color=colors_original_levels[2], width=width_original_levels[2], style=styles_original_levels[2])
            
            # Continue path to the textural features if they are the optimal level
            if np.argmax(width_original_levels) == 2:
                # Put best level index in the middle
                nodes_order = [0, 1, 2, 3, 4, 5]
                nodes_order.insert(3, nodes_order.pop(nodes_order.index(np.argmax(width_texture_families))))
                
                # Reorder nodes names
                nodes_names = ['GLCM', 'NGTDM', 'NGLDM', 'GLRLM', 'GLDZM', 'GLSZM']
                nodes_names = [nodes_names[i] for i in nodes_order]
                colors_texture_families = [colors_texture_families[i] for i in nodes_order]
                width_texture_families = [width_texture_families[i] for i in nodes_order]
                styles_texture_families = [styles_texture_families[i] for i in nodes_order]

                # Add texture features families nodes
                for idx, node_name in enumerate(nodes_names):
                    G.add_edge(
                        'Text', 
                        node_name, 
                        color=colors_texture_families[idx], 
                        width=width_texture_families[idx], 
                        style=styles_texture_families[idx]
                    )
                
                # Continue path to the textural features
                best_node_name = best_family_name.upper()
                for idx, feature in enumerate(features_names):
                    G.add_edge(
                        best_node_name, 
                        feature.replace('_', '\n'),
                        color=colors_texture_families_feature[idx], 
                        width=width_texture_families_feature[idx], 
                        style=styles_texture_families_feature[idx]
                    )
            
            # Graph layout
            pos = graphviz_layout(G, root=experiment_sep, prog="dot")

            # Create the plot: figure and axis
            fig = plt.figure(figsize=figsize, dpi=300)
            ax = fig.add_subplot(1, 1, 1)

            # Get the attributes of the edges
            colors = nx.get_edge_attributes(G,'color').values()
            widths = nx.get_edge_attributes(G,'width').values()
            style = nx.get_edge_attributes(G,'style').values()

            # Draw the graph
            cmap = [to_rgba('b')] * len(pos)
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
                Line2D([0], [0], color=selected_feat_color, lw=4, linestyle='solid', label=f'Selected (thickness reflects impact)'),
                Line2D([0], [0], color='black', lw=4, linestyle='dashed', label='Not selected'),
                Line2D([0], [0], color=optimal_lvl_color, lw=4, linestyle='solid', label='Path with highest impact')
            ]
            
            # Update keys according to the optimal level
            figure_keys = []
            if styles_original_levels[0] == "solid":
                figure_keys.append(mpatches.Patch(color='none', label='Morph: Morphological'))
            if styles_original_levels[1] == "solid":
                figure_keys.append(mpatches.Patch(color='none', label='Int: Intensity'))
            if styles_original_levels[2] == "solid":
                figure_keys.append(mpatches.Patch(color='none', label='Text: Textural'))

            # Set title
            if title:
                ax.set_title(title, fontsize=20)
            else:
                ax.set_title(
                    f'Radiomics explanation tree - Original level:'\
                    + f'\nExperiment: {experiment}'\
                    + f'\nLevel: {level}'\
                    + f'\nModality: {modality}', fontsize=20
                )

            # Apply the custom legend
            legend = plt.legend(handles=custom_legends, loc='upper right', fontsize=15, frameon=True, title = "Legend")
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(2.0)

            # Abbrevations legend
            legend_keys = plt.legend(handles=figure_keys, loc='center right', fontsize=15, frameon=True, title = "Abbreviations", handlelength=0)
            legend_keys.get_frame().set_edgecolor('black')
            legend_keys.get_frame().set_linewidth(2.0)

            # Options legend
            plt.gca().add_artist(legend_keys)
            plt.gca().add_artist(legend)
            
            # Tight layout
            fig.tight_layout()
            
            # Save the plot (Mandatory, since the plot is not well displayed on matplotlib)
            fig.savefig(path_experiments / f'Original_level_{experiment}_{level}_{modality}_explanation.png', dpi=300)

    def plot_lf_level_tree(
            self, 
            path_experiments: Path,
            experiment: str,
            level: str,
            modalities: list,
            initial_width: float = 4,
            lines_weight: float = 1,
            title: str = None,
            figsize: tuple = (12,10),
        ) -> None:
        """
        Plots a tree explaining the impact of features in the linear filters radiomics complexity level.

        Args:
            path_experiments (Path): Path to the folder containing the experiments.
            experiment (str): Name of the experiment to plot. Will be used to find the results.
            level (List): Radiomics complexity level to use for the plot.
            modalities (List, optional): List of imaging modalities to include in the plot. Defaults to [].
            initial_width (float, optional): Initial width of the lines. Defaults to 1. For aesthetic purposes.
            lines_weight (float, optional): Weight applied to the lines of the tree. Defaults to 2. For aesthetic purposes.
            title(str, optional): Title and name used to save the plot. Defaults to None.
            figsize(tuple, optional): Size of the figure. Defaults to (20, 10).
        
        Returns:
            None.
        """
        # Fill tree data 
        for modality in modalities:
            # Initialization
            selected_feat_color = 'limegreen'
            optimal_lvl_color = 'darkorange'

            # Initialization - outcome - levels
            styles_outcome_levels = ["dashed"] * 3
            colors_outcome_levels = ["black"] * 3
            width_outcome_levels = [initial_width] * 3

            # Initialization - lf - sublevels
            filters_names = ['mean', 'log', 'laws', 'gabor', 'coif']
            styles_lf_levels = ["dashed"] * 2
            colors_lf_levels = ["black"] * 2
            width_lf_levels = [initial_width] * 2

            # Initialization - texture-families
            styles_texture_families = ["dashed"] * 6
            colors_texture_families = ["black"] * 6
            width_texture_families = [initial_width] * 6
            families_names = ["glcm", "ngtdm", "ngldm", "glrlm", "gldzm", "glszm"]
            
            # Get feature importance dict
            exp_full_name = 'learn__' + experiment + '_' + level + '_' + modality
            if 'feature_importance_analysis.json' in os.listdir(path_experiments / exp_full_name):
                fa_dict = load_json(path_experiments / exp_full_name / 'feature_importance_analysis.json')
            else:
                fa_dict = feature_imporance_analysis(path_experiments / exp_full_name)
            
            # Organize data
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
        
            # Applying the lines weight
            df['final_coefficient'] *= lines_weight

            # Finding linear filters features and updating the connections
            for i, row in df['features'].items():
                level_name = row.split('__')[1].lower()
                family_name = row.split('__')[2].lower()
                    
                # Linear filters
                if level_name.startswith('mean') \
                    or level_name.startswith('log') \
                    or level_name.startswith('laws') \
                    or level_name.startswith('gabor') \
                    or level_name.startswith('wavelet') \
                    or level_name.startswith('coif'):

                    # Update outcome-original connection
                    styles_outcome_levels[1] = "solid"
                    colors_outcome_levels[1] = selected_feat_color
                    width_outcome_levels[1] += df['final_coefficient'][i]
            
            # Find the best performing filter
            width_lf_filters = [initial_width] * 5
            for i, row in df['features'].items():
                level_name = row.split('__')[1].lower()
                family_name = row.split('__')[2].lower()
                if level_name.startswith('mean'):
                    width_lf_filters[0] += df['final_coefficient'][i]
                elif level_name.startswith('log'):
                    width_lf_filters[1] += df['final_coefficient'][i]
                elif level_name.startswith('laws'):
                    width_lf_filters[2] += df['final_coefficient'][i]
                elif level_name.startswith('gabor'):
                    width_lf_filters[3] += df['final_coefficient'][i]
                elif level_name.startswith('wavelet'):
                    width_lf_filters[4] += df['final_coefficient'][i]
                elif level_name.startswith('coif'):
                    width_lf_filters[4] += df['final_coefficient'][i]
            
            # Get best filter
            index_best_filter = np.argmax(width_lf_filters)
            best_filter = filters_names[index_best_filter]
            
            # Seperate intensity and texture then update the connections
            for i, row in df['features'].items():
                level_name = row.split('__')[1].lower()
                family_name = row.split('__')[2].lower()
                if level_name.startswith(best_filter):
                    if family_name.startswith('_int'):
                        width_lf_levels[0] += df['final_coefficient'][i]
                    elif family_name.startswith(tuple(['_glcm', '_gldzm', '_glrlm', '_glszm', '_ngtdm', '_ngldm'])):
                        width_lf_levels[1] += df['final_coefficient'][i]

            # If Texture features are more impacful, update the connections
            if width_lf_levels[1] > width_lf_levels[0]:
                colors_lf_levels[1] = optimal_lvl_color
                styles_lf_levels[1] = "solid"

                # Update lf-texture-families connection
                for i, row in df['features'].items():
                    level_name = row.split('__')[1].lower()
                    family_name = row.split('__')[2].lower()
                    if not family_name.startswith('_int') and level_name.startswith(best_filter):
                        if family_name.startswith('_glcm'):
                            styles_texture_families[0] = "solid"
                            colors_texture_families[0] = selected_feat_color
                            width_texture_families[0] += df['final_coefficient'][i]
                        elif family_name.startswith('_ngtdm'):
                            styles_texture_families[1] = "solid"
                            colors_texture_families[1] = selected_feat_color
                            width_texture_families[1] += df['final_coefficient'][i]
                        elif family_name.startswith('_ngldm'):
                            styles_texture_families[2] = "solid"
                            colors_texture_families[2] = selected_feat_color
                            width_texture_families[2] += df['final_coefficient'][i]
                        elif family_name.startswith('_glrlm'):
                            styles_texture_families[3] = "solid"
                            colors_texture_families[3] = selected_feat_color
                            width_texture_families[3] += df['final_coefficient'][i]
                        elif family_name.startswith('_gldzm'):
                            styles_texture_families[4] = "solid"
                            colors_texture_families[4] = selected_feat_color
                            width_texture_families[4] += df['final_coefficient'][i]
                        elif family_name.startswith('_glszm'):
                            styles_texture_families[5] = "solid"
                            colors_texture_families[5] = selected_feat_color
                            width_texture_families[5] += df['final_coefficient'][i]
                        else:
                            raise ValueError(f'Family of the feature {family_name} not recognized')
                
                # Update color
                colors_texture_families[np.argmax(width_texture_families)] = optimal_lvl_color
                
            else:
                colors_lf_levels[0] = optimal_lvl_color
                styles_lf_levels[0] = "solid"
            
            # If texture features are the optimal level, continue path
            if width_lf_levels[1] > width_lf_levels[0]:

                # Get best texture family
                best_family_name = ""
                index_best_family = np.argmax(width_texture_families)
                best_family_name = families_names[index_best_family]
                features_names = texture_features_all[index_best_family]

                # Update texture-families-features connection
                width_texture_families_feature = [initial_width] * len(features_names)
                colors_texture_families_feature = ["black"] * len(features_names)
                styles_texture_families_feature = ["dashed"] * len(features_names)
                for i, row in df['features'].items():
                    level_name = row.split('__')[1].lower()
                    family_name = row.split('__')[2].lower()
                    feature_name = row.split('__')
                    if family_name.startswith('_' + best_family_name) and level_name.startswith(best_filter):
                        for feature in features_names:
                            if feature in feature_name:
                                colors_texture_families_feature[features_names.index(feature)] = selected_feat_color
                                styles_texture_families_feature[features_names.index(feature)] = "solid"
                                width_texture_families_feature[features_names.index(feature)] += df['final_coefficient'][i]
                                break
                
                # Update color for the best texture family
                colors_texture_families_feature[np.argmax(width_texture_families_feature)] = optimal_lvl_color
            
            # For esthetic purposes
            experiment_sep = experiment.replace('_', '\n')

            # Design the graph
            G = nx.Graph()

            # Linear filters level
            G.add_edge(experiment_sep, 'LF', color=optimal_lvl_color, width=np.sum(width_lf_filters), style=styles_outcome_levels[1])

            # Add best filter
            best_filter = best_filter.replace('_', '\n')
            G.add_edge('LF', best_filter.upper(), color=optimal_lvl_color, width=width_lf_filters[index_best_filter], style="solid")

            # Int or Text
            if width_lf_levels[1] <= width_lf_levels[0]:
                G.add_edge(best_filter.upper(), 'LF\nInt', color=colors_lf_levels[0], width=width_lf_levels[0], style=styles_lf_levels[0])
            else:
                G.add_edge(best_filter.upper(), 'LF\nText', color=colors_lf_levels[1], width=width_lf_levels[1], style=styles_lf_levels[1])
                
                # Put best level index in the middle
                nodes_order = [0, 1, 2, 3, 4, 5]
                nodes_order.insert(3, nodes_order.pop(nodes_order.index(np.argmax(width_texture_families))))
                
                # Reorder nodes names
                nodes_names = ['LF\nGLCM', 'LF\nNGTDM', 'LF\nNGLDM', 'LF\nGLRLM', 'LF\nGLDZM', 'LF\nGLSZM']
                nodes_names = [nodes_names[i] for i in nodes_order]
                colors_texture_families = [colors_texture_families[i] for i in nodes_order]
                width_texture_families = [width_texture_families[i] for i in nodes_order]
                styles_texture_families = [styles_texture_families[i] for i in nodes_order]

                # Add texture features families nodes
                for idx, node_name in enumerate(nodes_names):
                    G.add_edge(
                        'LF\nText', 
                        node_name, 
                        color=colors_texture_families[idx], 
                        width=width_texture_families[idx], 
                        style=styles_texture_families[idx]
                    )

                # Continue path to the textural features
                best_node_name = f'LF\n{best_family_name.upper()}'
                for idx, feature in enumerate(features_names):
                    G.add_edge(
                        best_node_name, 
                        feature.replace('_', '\n'),
                        color=colors_texture_families_feature[idx], 
                        width=width_texture_families_feature[idx], 
                        style=styles_texture_families_feature[idx]
                    )
            
            # Graph layout
            pos = graphviz_layout(G, root=experiment_sep, prog="dot")

            # Create the plot: figure and axis
            fig = plt.figure(figsize=figsize, dpi=300)
            ax = fig.add_subplot(1, 1, 1)

            # Get the attributes of the edges
            colors = nx.get_edge_attributes(G,'color').values()
            widths = nx.get_edge_attributes(G,'width').values()
            style = nx.get_edge_attributes(G,'style').values()

            # Draw the graph
            cmap = [to_rgba('b')] * len(pos)
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
                Line2D([0], [0], color=selected_feat_color, lw=4, linestyle='solid', label=f'Selected (thickness reflects impact)'),
                Line2D([0], [0], color='black', lw=4, linestyle='dashed', label='Not selected'),
                Line2D([0], [0], color=optimal_lvl_color, lw=4, linestyle='solid', label='Path with highest impact')
            ]
            
            # Update keys according to the optimal level
            figure_keys = []
            figure_keys.append(mpatches.Patch(color='none', label='LF: Linear Filters'))
            if width_lf_levels[1] > width_lf_levels[0]:
                figure_keys.append(mpatches.Patch(color='none', label='Text: Textural'))
            else:
                figure_keys.append(mpatches.Patch(color='none', label='Int: Intensity'))

            # Set title
            if title:
                ax.set_title(title, fontsize=20)
            else:
                ax.set_title(
                    f'Radiomics explanation tree:'\
                    + f'\nExperiment: {experiment}'\
                    + f'\nLevel: {level}'\
                    + f'\nModality: {modality}', fontsize=20
                )

            # Apply the custom legend
            legend = plt.legend(handles=custom_legends, loc='upper right', fontsize=15, frameon=True, title = "Legend")
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(2.0)

            # Abbrevations legend
            legend_keys = plt.legend(handles=figure_keys, loc='center right', fontsize=15, frameon=True, title = "Abbreviations", handlelength=0)
            legend_keys.get_frame().set_edgecolor('black')
            legend_keys.get_frame().set_linewidth(2.0)

            # Options legend
            plt.gca().add_artist(legend_keys)
            plt.gca().add_artist(legend)
            
            # Tight layout
            fig.tight_layout()
            
            # Save the plot (Mandatory, since the plot is not well displayed on matplotlib)
            fig.savefig(path_experiments / f'LF_level_{experiment}_{level}_{modality}_explanation_tree.png', dpi=300)

    def plot_tf_level_tree(
            self, 
            path_experiments: Path,
            experiment: str,
            level: str,
            modalities: list,
            initial_width: float = 4,
            lines_weight: float = 1,
            title: str = None,
            figsize: tuple = (12,10),
        ) -> None:
        """
        Plots a tree explaining the impact of features in the textural filters radiomics complexity level.

        Args:
            path_experiments (Path): Path to the folder containing the experiments.
            experiment (str): Name of the experiment to plot. Will be used to find the results.
            level (List): Radiomics complexity level to use for the plot.
            modalities (List, optional): List of imaging modalities to include in the plot. Defaults to [].
            initial_width (float, optional): Initial width of the lines. Defaults to 1. For aesthetic purposes.
            lines_weight (float, optional): Weight applied to the lines of the tree. Defaults to 2. For aesthetic purposes.
            title(str, optional): Title and name used to save the plot. Defaults to None.
            figsize(tuple, optional): Size of the figure. Defaults to (20, 10).
        
        Returns:
            None.
        """
        # Fill tree data 
        for modality in modalities:
            # Initialization
            selected_feat_color = 'limegreen'
            optimal_lvl_color = 'darkorange'

            # Initialization - outcome - levels
            styles_outcome_levels = ["dashed"] * 3
            colors_outcome_levels = ["black"] * 3
            width_outcome_levels = [initial_width] * 3

            # Initialization - tf - sublevels
            styles_tf_levels = ["dashed"] * 2
            colors_tf_levels = ["black"] * 2
            width_tf_levels = [initial_width] * 2

            # Initialization - tf - best filter
            width_tf_filters = [initial_width] * len(glcm_features_names)

            # Initialization - texture-families
            styles_texture_families = ["dashed"] * 6
            colors_texture_families = ["black"] * 6
            width_texture_families = [initial_width] * 6
            families_names = ["glcm", "ngtdm", "ngldm", "glrlm", "gldzm", "glszm"]
            
            # Get feature importance dict
            exp_full_name = 'learn__' + experiment + '_' + level + '_' + modality
            if 'feature_importance_analysis.json' in os.listdir(path_experiments / exp_full_name):
                fa_dict = load_json(path_experiments / exp_full_name / 'feature_importance_analysis.json')
            else:
                fa_dict = feature_imporance_analysis(path_experiments / exp_full_name)
            
            # Organize data
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
        
            # Applying the lines weight
            df['final_coefficient'] *= lines_weight

            # Filling the lines data for textural filters features and updating the connections
            for i, row in df['features'].items():
                level_name = row.split('__')[1].lower()
                family_name = row.split('__')[2].lower()

                # Textural filters
                if level_name.startswith('glcm'):
                    # Update outcome-original connection
                    styles_outcome_levels[2] = "solid"
                    colors_outcome_levels[2] = optimal_lvl_color
                    width_outcome_levels[2] += df['final_coefficient'][i]

                    # Update tf-best filter connection
                    for feature in glcm_features_names:
                        if feature + '__' in row:
                            width_tf_filters[glcm_features_names.index(feature)] += df['final_coefficient'][i]
                            break
            
            # Get best filter
            index_best_filter = np.argmax(width_tf_filters)
            best_filter = glcm_features_names[index_best_filter]
                
            # Seperate intensity and texture then update the connections
            for i, row in df['features'].items():
                level_name = row.split('__')[1].lower()
                family_name = row.split('__')[2].lower()
                if level_name.startswith('glcm') and best_filter + '__' in row:
                    if family_name.startswith('_int'):
                        width_tf_levels[0] += df['final_coefficient'][i]
                    elif family_name.startswith(tuple(['_glcm', '_gldzm', '_glrlm', '_glszm', '_ngtdm', '_ngldm'])):
                        width_tf_levels[1] += df['final_coefficient'][i]
            
            # If Texture features are more impacful, update the connections
            if width_tf_levels[1] > width_tf_levels[0]:
                colors_tf_levels[1] = optimal_lvl_color
                styles_tf_levels[1] = "solid"

                # Update tf-texture-families connection
                for i, row in df['features'].items():
                    level_name = row.split('__')[1].lower()
                    family_name = row.split('__')[2].lower()
                    if level_name.startswith('glcm') and best_filter + '__' in row:
                        if family_name.startswith('_glcm'):
                            styles_texture_families[0] = "solid"
                            colors_texture_families[0] = selected_feat_color
                            width_texture_families[0] += df['final_coefficient'][i]
                        elif family_name.startswith('_ngtdm'):
                            styles_texture_families[1] = "solid"
                            colors_texture_families[1] = selected_feat_color
                            width_texture_families[1] += df['final_coefficient'][i]
                        elif family_name.startswith('_ngldm'):
                            styles_texture_families[2] = "solid"
                            colors_texture_families[2] = selected_feat_color
                            width_texture_families[2] += df['final_coefficient'][i]
                        elif family_name.startswith('_glrlm'):
                            styles_texture_families[3] = "solid"
                            colors_texture_families[3] = selected_feat_color
                            width_texture_families[3] += df['final_coefficient'][i]
                        elif family_name.startswith('_gldzm'):
                            styles_texture_families[4] = "solid"
                            colors_texture_families[4] = selected_feat_color
                            width_texture_families[4] += df['final_coefficient'][i]
                        elif family_name.startswith('_glszm'):
                            styles_texture_families[5] = "solid"
                            colors_texture_families[5] = selected_feat_color
                            width_texture_families[5] += df['final_coefficient'][i]
                    
                # Get best texture family
                best_family_name = ""
                index_best_family = np.argmax(width_texture_families)
                best_family_name = families_names[index_best_family]
                features_names = texture_features_all[index_best_family]

                # Update texture-families-features connection
                width_texture_families_feature = [initial_width] * len(features_names)
                colors_texture_families_feature = ["black"] * len(features_names)
                styles_texture_families_feature = ["dashed"] * len(features_names)
                for i, row in df['features'].items():
                    level_name = row.split('__')[1].lower()
                    family_name = row.split('__')[2].lower()
                    feature_name = row.split('__')
                    if level_name.startswith('glcm') and family_name.startswith('_' + best_family_name) and best_filter + '__' in row:
                        for feature in features_names:
                            if feature in feature_name:
                                colors_texture_families_feature[features_names.index(feature)] = selected_feat_color
                                styles_texture_families_feature[features_names.index(feature)] = "solid"
                                width_texture_families_feature[features_names.index(feature)] += df['final_coefficient'][i]
                                break
                
                # Update color for the best texture family
                colors_texture_families_feature[np.argmax(width_texture_families_feature)] = optimal_lvl_color
                
                # Update color
                colors_texture_families[np.argmax(width_texture_families)] = optimal_lvl_color
            else:
                colors_tf_levels[0] = optimal_lvl_color
                styles_tf_levels[0] = "solid"

            # For esthetic purposes
            experiment_sep = experiment.replace('_', '\n')

            # Design the graph
            G = nx.Graph()
            G.add_edge(experiment_sep, 'TF', color=colors_outcome_levels[2], width=width_outcome_levels[2], style=styles_outcome_levels[2])

            # Add best filter
            best_filter = best_filter.replace('_', '\n')
            G.add_edge('TF', best_filter.upper(), color=optimal_lvl_color, width=width_tf_filters[index_best_filter], style="solid")

            # Check which level is the best (intensity or texture)
            if width_tf_levels[1] <= width_tf_levels[0]:
                G.add_edge(best_filter.upper(), 'TF\nInt', color=colors_tf_levels[0], width=width_tf_levels[0], style=styles_tf_levels[0])
            else:
                G.add_edge(best_filter.upper(), 'TF\nText', color=colors_tf_levels[1], width=width_tf_levels[1], style=styles_tf_levels[1])
                
                # Put best level index in the middle
                nodes_order = [0, 1, 2, 3, 4, 5]
                nodes_order.insert(3, nodes_order.pop(nodes_order.index(np.argmax(width_texture_families))))
                
                # Reorder nodes names
                nodes_names = ['TF\nGLCM', 'TF\nNGTDM', 'TF\nNGLDM', 'TF\nGLRLM', 'TF\nGLDZM', 'TF\nGLSZM']
                nodes_names = [nodes_names[i] for i in nodes_order]
                colors_texture_families = [colors_texture_families[i] for i in nodes_order]
                width_texture_families = [width_texture_families[i] for i in nodes_order]
                styles_texture_families = [styles_texture_families[i] for i in nodes_order]
                
                # Add texture features families nodes
                for idx, node_names in enumerate(nodes_names):
                    G.add_edge(
                        'TF\nText',
                        node_names,
                        color=colors_texture_families[idx],
                        width=width_texture_families[idx],
                        style=styles_texture_families[idx]
                    )
                
                # Continue path to the textural features
                best_node_name = f'TF\n{best_family_name.upper()}'
                for idx, feature in enumerate(features_names):
                    G.add_edge(
                        best_node_name, 
                        feature.replace('_', '\n'),
                        color=colors_texture_families_feature[idx], 
                        width=width_texture_families_feature[idx], 
                        style=styles_texture_families_feature[idx]
                        )
            
            # Graph layout
            pos = graphviz_layout(G, root=experiment_sep, prog="dot")

            # Create the plot: figure and axis
            fig = plt.figure(figsize=figsize, dpi=300)
            ax = fig.add_subplot(1, 1, 1)

            # Get the attributes of the edges
            colors = nx.get_edge_attributes(G,'color').values()
            widths = nx.get_edge_attributes(G,'width').values()
            style = nx.get_edge_attributes(G,'style').values()

            # Draw the graph
            cmap = [to_rgba('b')] * len(pos)
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
                Line2D([0], [0], color=selected_feat_color, lw=4, linestyle='solid', label=f'Selected (thickness reflects impact)'),
                Line2D([0], [0], color='black', lw=4, linestyle='dashed', label='Not selected')
            ]
            figure_keys = []
            
            # Update keys according to the optimal level
            figure_keys = [mpatches.Patch(color='none', label='TF: Linear Filters')]
            if width_tf_levels[1] > width_tf_levels[0]:
                figure_keys.append(mpatches.Patch(color='none', label='Text: Textural'))
            else:
                figure_keys.append(mpatches.Patch(color='none', label='Int: Intensity'))
            
            custom_legends.append(
                Line2D([0], [0], color=optimal_lvl_color, lw=4, linestyle='solid', label='Path with highest impact')
            )

            # Set title
            if title:
                ax.set_title(title, fontsize=20)
            else:
                ax.set_title(
                    f'Radiomics explanation tree:'\
                    + f'\nExperiment: {experiment}'\
                    + f'\nLevel: {level}'\
                    + f'\nModality: {modality}', fontsize=20
                )

            # Apply the custom legend
            legend = plt.legend(handles=custom_legends, loc='upper right', fontsize=15, frameon=True, title = "Legend")
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(2.0)

            # Abbrevations legend
            legend_keys = plt.legend(handles=figure_keys, loc='center right', fontsize=15, frameon=True, title = "Abbreviations", handlelength=0)
            legend_keys.get_frame().set_edgecolor('black')
            legend_keys.get_frame().set_linewidth(2.0)

            # Options legend
            plt.gca().add_artist(legend_keys)
            plt.gca().add_artist(legend)
            
            # Tight layout
            fig.tight_layout()
            
            # Save the plot (Mandatory, since the plot is not well displayed on matplotlib)
            fig.savefig(path_experiments / f'TF_{experiment}_{level}_{modality}_explanation_tree.png', dpi=300)

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
