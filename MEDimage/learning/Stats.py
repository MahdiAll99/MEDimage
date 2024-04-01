# Description: All the functions related to statistics (p-values, metrics, etc.)

import os
from pathlib import Path
from typing import List, Tuple
import warnings

import numpy as np
import pandas as pd
import scipy
from sklearn import metrics

from MEDimage.utils.json_utils import load_json


class Stats:
    """
    A class to perform statistical analysis on experiment results.

    This class provides methods to retrieve patient IDs, predictions, and metrics from experiment data,
    as well as compute the p-values for model comparison using various methods.

    Args:
        path_experiment (Path): Path to the folder containing the experiment data.
        experiment (str): Name of the experiment.
        levels (List): List of radiomics levels to analyze.
        modalities (List): List of modalities to analyze.

    Attributes:
        path_experiment (Path): Path to the folder containing the experiment data.
        experiment (str): Name of the experiment.
        levels (List): List of radiomics levels to analyze.
        modalities (List): List of modalities to analyze.
    """
    def __init__(self, path_experiment: Path, experiment: str, levels: List, modalities: List):
        # Initialization
        self.path_experiment = path_experiment
        self.experiment = experiment
        self.levels = levels
        self.modalities = modalities

        # Safety assertion
        self.__safety_assertion()
    
    def __get_models_dicts(self, split_idx: int) -> Path:
        """
        Retrieves the models dictionaries for a given split.

        Args:
            split_idx (int): Index of the split.

        Returns:
            List: List of paths to the models dictionaries.
        """
        # Get level and modality
        if len(self.modalities) == 1:
            # Load ground truths and predictions
            path_json_1 = self.__get_path_json(self.levels[0], self.modalities[0], split_idx)
            path_json_2 = self.__get_path_json(self.levels[1], self.modalities[0], split_idx)
        else:
            # Load ground truths and predictions
            path_json_1 = self.__get_path_json(self.levels[0], self.modalities[0], split_idx)
            path_json_2 = self.__get_path_json(self.levels[0], self.modalities[1], split_idx)
        return path_json_1, path_json_2
        
    def __safety_assertion(self):
        """
        Asserts that the input parameters are correct.
        """
        if len(self.modalities) == 1:
            assert len(self.levels) == 2, \
                "For statistical analysis, the number of levels must be 2 for a single modality, or 1 for two modalities"
        elif len(self.modalities) == 2:
            assert len(self.levels) == 1, \
                "For statistical analysis, the number of levels must be 1 for two modalities, or 2 for a single modality"
        else:
            raise ValueError("The number of modalities must be 1 or 2")
    
    def __get_path_json(self, level: str, modality: str, split_idx: int) -> Path:
        """
        Retrieves the path to the models dictionary for a given split.

        Args:
            level (str): Radiomics level.
            modality (str): Modality.
            split_idx (int): Index of the split.

        Returns:
            Path: Path to the models dictionary.
        """
        return self.path_experiment / f'learn__{self.experiment}_{level}_{modality}' / f'test__{split_idx:03d}' / 'run_results.json'

    def __get_patients_and_predictions(
            self,
            split_idx: int
        ) -> tuple:
        """
        Retrieves patient IDs, predictions of both models for a given split.

        Args:
            split_idx (int): Index of the split.

        Returns:
            tuple: Tuple containing the patient IDs, predictions of the first model and predictions of the second model.
        """
        # Get models dicts
        path_json_1, path_json_2 = self.__get_models_dicts(split_idx)

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
            # Warn the user
            warnings.warn("The number of patients is different for both models. Patients will be deleted to match the number of patients.")

            # Delete patients
            for patient_id in patients_ids_one:
                if patient_id not in patients_ids_two:
                    patients_delete.append(patient_id)
                    predictions_one.pop(patients_ids_one.index(patient_id))
            for patient in patients_delete:
                patients_ids_one.remove(patient)
        elif len(patients_ids_one) < len(patients_ids_two):
            # Warn the user
            warnings.warn("The number of patients is different for both models. Patients will be deleted to match the number of patients.")

            # Delete patients
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
        
        return patients_ids_one, predictions_one, predictions_two

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
    
    def __corrected_std(self, differences: np.array, n_train: int, n_test: int) -> float:
        """
        Corrects standard deviation using Nadeau and Bengio's approach.

        Args:
            differences (np.array): Vector containing the differences in the score metrics of two models.
            n_train (int): Number of samples in the training set.
            n_test (int): Number of samples in the testing set.
        
        Returns:
            float: Variance-corrected standard deviation of the set of differences.
        
        Reference:
            `Statistical comparison of models 
            <https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html#comparing-two-models-frequentist-approach>.`
        """
        # kr = k times r, r times repeated k-fold crossvalidation,
        # kr equals the number of times the model was evaluated
        kr = len(differences)
        corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
        corrected_std = np.sqrt(corrected_var)
        return corrected_std

    def __compute_midrank(self, x: np.array) -> np.array:
        """
        Computes midranks for Delong p-value.
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

    def __fast_delong(self, predictions_sorted_transposed: np.array, label_1_count: int) -> Tuple[float, float]:
        """
        Computes the empricial AUC and its covariance using the fast version of DeLong's method.

        Args:
            predictions_sorted_transposed (np.array): a 2D numpy.array[n_classifiers, n_examples]
                sorted such as the examples with label "1" are first.
            label_1_count (int): number of examples with label "1".
        
        Returns:
            Tuple(float, float): (AUC value, DeLong covariance)
        
        Reference:
            `Python fast delong implementation <https://github.com/yandexdataschool/roc_comparison/tree/master>.`
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

    def __compute_ground_truth_statistics(self, ground_truth: np.array) -> Tuple[np.array, int]:
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

    def __get_metrics(self, metric: str, split_idx: int) -> tuple:
        """
        Initializes the p-value information that will be used to compute the p-values across all different methods.

        Args:
            metric (str): Metric to retrieve.
            split_idx (int): Index of the split.

        Returns:
            tuple: Tuple containing the metrics of the first model and metrics of the second model.
        """
        # Get models dicts
        path_json_1, path_json_2 = self.__get_models_dicts(split_idx)

        # Load models dicts
        model_one = load_json(path_json_1)
        model_two = load_json(path_json_2)

        # Get name models
        name_model_one = list(model_one.keys())[0]
        name_model_two = list(model_two.keys())[0]

        # Get predictions
        metric_one = model_one[name_model_one]['test']['metrics'][metric]
        metric_two = model_two[name_model_two]['test']['metrics'][metric]

        return metric_one, metric_two

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

    @staticmethod
    def get_aggregated_metric(
            path_experiment: Path, 
            experiment: str, 
            level: str, 
            modality: str,
            metric: str
        ) -> float:
        """
        Calculates the p-value of the Delong test for the given experiment.

        Args:
            path_experiment (Path): Path to the folder containing the experiment.
            experiment (str): Name of the experiment.
            level (str): Radiomics level. For example: 'morph'.
            modality (str): Modality to analyze.
            metric (str): Metric to analyze.
        
        Returns:
            float: p-value of the Delong test.
        """
        
        # Load outcomes dataframe
        try:
            outcomes = pd.read_csv(path_experiment / "outcomes.csv", sep=',')
        except:
            outcomes = pd.read_csv(path_experiment.parent / "outcomes.csv", sep=',')

        # Initialization
        predictions_all = list()
        patients_ids_all = list()
        nb_split = len([x[0] for x in os.walk(path_experiment / f'learn__{experiment}_{level}_{modality}')]) - 1

        # For each split
        for i in range(1, nb_split + 1):
            # Load ground truths and predictions
            path_json = path_experiment / f'learn__{experiment}_{level}_{modality}' / f'test__{i:03d}' / 'run_results.json'

            # Load models dicts
            model = load_json(path_json)

            # Get name models
            name_model = list(model.keys())[0]

            # Get Model's threshold
            thresh = model[name_model]['threshold']

            # Get predictions
            predictions = np.array(model[name_model]['test']['response'])
            predictions = np.reshape(predictions, (predictions.shape[0])).tolist()

            # Bring all predictions to 0.5
            predictions = [prediction - thresh + 0.5 if thresh >= 0.5 else prediction + 0.5 - thresh for prediction in predictions]
            predictions_all.extend(predictions)

            # Get patients ids
            patients_ids = model[name_model]['test']['patients'] 
            
            # After verification, add-up patients IDs
            patients_ids_all.extend(patients_ids)

        # Get ground truth for selected patients
        ground_truth = []
        for patient in patients_ids_all:
            ground_truth.append(outcomes[outcomes['PatientID'] == patient][outcomes.columns[-1]].values[0])
        
        # to numpy array
        ground_truth = np.array(ground_truth)
        
        # Get aggregated metric
        # AUC
        if metric == 'AUC':
            auc = metrics.roc_auc_score(ground_truth, predictions_all)
            return auc

        # AUPRC
        elif metric == 'AUPRC':
            auc = metrics.average_precision_score(ground_truth, predictions_all)
        
        # Confusion matrix-based metrics
        else:
            TP = ((np.array(predictions_all) >= 0.5) & (ground_truth == 1)).sum()
            TN = ((np.array(predictions_all) < 0.5) & (ground_truth == 0)).sum()
            FP = ((np.array(predictions_all) >= 0.5) & (ground_truth == 0)).sum()
            FN = ((np.array(predictions_all) < 0.5) & (ground_truth == 1)).sum()

            # Asserts
            assert TP + FN != 0, "TP + FN = 0, Division by 0"
            assert TN + FP != 0, "TN + FP = 0, Division by 0"

            # Sensitivity
            if metric == 'Sensitivity':
                sensitivity = TP / (TP + FN)
                return sensitivity

            # Specificity
            elif metric == 'Specificity':
                specificity = TN / (TN + FP)
                return specificity

            else:
                raise ValueError(f"Metric {metric} not supported. Supported metrics: AUC, AUPRC, Sensitivity, Specificity.\
                                 Update file Stats.py to add the new metric.")
    
    def get_aggregated_delong_p_value(self) -> float:
        """
        Calculates the p-value of the Delong test for the given experiment.
        
        Returns:
            float: p-value of the Delong test.
        """
        
        # Load outcomes dataframe
        try:
            outcomes = pd.read_csv(self.path_experiment / "outcomes.csv", sep=',')
        except:
            outcomes = pd.read_csv(self.path_experiment.parent / "outcomes.csv", sep=',')

        # Initialization
        predictions_one_all = list()
        predictions_two_all = list()
        patients_ids_all = list()
        nb_split = len([x[0] for x in os.walk(self.path_experiment / f'learn__{self.experiment}_{self.levels[0]}_{self.modalities[0]}')]) - 1

        # For each split
        for i in range(1, nb_split + 1):
            # Get predictions and patients ids
            patients_ids, predictions_one, predictions_two = self.__get_patients_and_predictions(i)
            
            # Add-up all information
            predictions_one_all.extend(predictions_one)
            predictions_two_all.extend(predictions_two)
            patients_ids_all.extend(patients_ids)

        # Get ground truth for selected patients
        ground_truth = []
        for patient in patients_ids_all:
            ground_truth.append(outcomes[outcomes['PatientID'] == patient][outcomes.columns[-1]].values[0])

        # to numpy array
        ground_truth = np.array(ground_truth)
        
        # Get p-value
        pvalue = self.__delong_roc_test(ground_truth, predictions_one_all, predictions_two_all).item()

        # Compute the median p-value of all splits
        return pvalue

    def get_bengio_p_value(self) -> float:
        """
        Computes Bengio's right-tailed paired t-test with corrected variance.
        
        Returns:
            float: p-value of the Bengio test.
        """

        # Initialization
        metrics_one_all = list()
        metrics_two_all = list()
        nb_split = len([x[0] for x in os.walk(self.path_experiment / f'learn__{self.experiment}_{self.levels[0]}_{self.modalities[0]}')]) - 1

        # For each split
        for i in range(1, nb_split + 1):
            # Get models dicts
            path_json_1, path_json_2 = self.__get_models_dicts(i)

            # Load patients train and test lists
            patients_train = load_json(path_json_1.parent / 'patientsTrain.json')
            patients_test = load_json(path_json_1.parent / 'patientsTest.json')
            n_train = len(patients_train)
            n_test = len(patients_test) 

            # Load models dicts
            model_one = load_json(path_json_1)
            model_two = load_json(path_json_2)

            # Get name models
            name_model_one = list(model_one.keys())[0]
            name_model_two = list(model_two.keys())[0]

            # Get predictions
            metric_one = model_one[name_model_one]['test']['metrics']['AUC']
            metric_two = model_two[name_model_two]['test']['metrics']['AUC']
            
            # Add-up all information
            metrics_one_all.append(metric_one)
            metrics_two_all.append(metric_two)

        # Check if the number of predictions is the same
        if len(metrics_one_all) != len(metrics_two_all):
            raise ValueError("The number of metrics must be the same for both models")
            
        # Get differences
        differences = np.array(metrics_one_all) - np.array(metrics_two_all)
        df = differences.shape[0] - 1

        # Get corrected std
        mean = np.mean(differences)
        std = self.__corrected_std(differences, n_train, n_test)

        # Get p-value
        t_stat = mean / std
        p_val = scipy.stats.t.sf(np.abs(t_stat), df)  # right-tailed t-test

        return p_val

    def get_delong_p_value(
            self,
            aggregate: bool = False,
        ) -> float:
        """
        Calculates the p-value of the Delong test for the given experiment.

        Args:
            aggregate (bool, optional): If True, aggregates the results of all the splits and computes one final p-value.
        
        Returns:
            float: p-value of the Delong test.
        """
        
        # Check if aggregation is needed
        if aggregate:
            return self.get_aggregated_delong_p_value()
        
        # Load outcomes dataframe
        try:
            outcomes = pd.read_csv(self.path_experiment / "outcomes.csv", sep=',')
        except:
            outcomes = pd.read_csv(self.path_experiment.parent / "outcomes.csv", sep=',')

        # Initialization
        nb_split = len([x[0] for x in os.walk(self.path_experiment / f'learn__{self.experiment}_{self.levels[0]}_{self.modalities[0]}')]) - 1
        list_p_values_temp = list()

        # For each split
        for i in range(1, nb_split + 1):
            # Get predictions and patients ids
            patients_ids, predictions_one, predictions_two = self.__get_patients_and_predictions(i)

            # Get ground truth for selected patients
            ground_truth = []
            for patient in patients_ids:
                ground_truth.append(outcomes[outcomes['PatientID'] == patient][outcomes.columns[-1]].values[0])
            
            # to numpy array
            ground_truth = np.array(ground_truth)
            
            # Get p-value
            pvalue = self.__delong_roc_test(ground_truth, predictions_one, predictions_two).item()

            list_p_values_temp.append(pvalue)
            
        # Compute the median p-value of all splits
        return np.median(list_p_values_temp)

    def get_ttest_p_value(self, metric: str = 'AUC',) -> float:
        """
        Calculates the p-value using the t-test for two related samples of scores.

        Args:
            metric (str, optional): Metric to use for comparison. Defaults to 'AUC'.
        
        Returns:
            float: p-value of the Delong test.
        """

        # Initialization
        metric = metric.split('_')[0] if '_' in metric else metric
        metrics_one_all = list()
        metrics_two_all = list()
        nb_split = len([x[0] for x in os.walk(self.path_experiment / f'learn__{self.experiment}_{self.levels[0]}_{self.modalities[0]}')]) - 1

        # For each split
        for i in range(1, nb_split + 1):
            # Get metrics of the first and second model
            metric_one, metric_two = self.__get_metrics(metric, i)
            
            # Add-up all information
            metrics_one_all.append(metric_one)
            metrics_two_all.append(metric_two)

        # Check if the number of predictions is the same
        if len(metrics_one_all) != len(metrics_two_all):
            raise ValueError("The number of metrics must be the same for both models")
        
        # Compute p-value by performing paired t-test
        _, p_value = scipy.stats.ttest_rel(metrics_one_all, metrics_two_all)

        return p_value

    def get_wilcoxin_p_value(self, metric: str = 'AUC',) -> float:
        """
        Calculates the p-value using the t-test for two related samples of scores.

        Args:
            metric (str, optional): Metric to analyze. Defaults to 'AUC'.
        
        Returns:
            float: p-value of the Delong test.
        """

        # Initialization
        metric = metric.split('_')[0] if '_' in metric else metric
        metrics_one_all = list()
        metrics_two_all = list()
        nb_split = len([x[0] for x in os.walk(self.path_experiment / f'learn__{self.experiment}_{self.levels[0]}_{self.modalities[0]}')]) - 1

        # For each split
        for i in range(1, nb_split + 1):
            # Get metrics of the first and second model
            metric_one, metric_two = self.__get_metrics(metric, i)
            
            # Add-up all information
            metrics_one_all.append(metric_one)
            metrics_two_all.append(metric_two)

        # Check if the number of predictions is the same
        if len(metrics_one_all) != len(metrics_two_all):
            raise ValueError("The number of metrics must be the same for both models")
        
        # Compute p-value by performing wilcoxon signed rank test
        _, p_value = scipy.stats.wilcoxon(metrics_one_all, metrics_two_all)
        
        return p_value

    def get_p_value(
            self,
            method: str,
            metric: str = 'AUC',
            aggregate: bool = False
        ) -> float:
        """
        Calculates the p-value of the given method.

        Args:
            method (str): Method to use to calculate the p-value. Available options:
                - 'delong': Delong test.
                - 'ttest': T-test.
                - 'wilcoxon': Wilcoxon signed rank test.
                - 'bengio': Bengio and Nadeau corrected t-test.
            metric (str, optional): Metric to analyze. Defaults to 'AUC'.
            aggregate (bool, optional): If True, aggregates the results of all the splits and computes one final p-value.
        
        Returns:
            float: p-value of the Delong test.
        """
        # Assertions
        assert method in ['delong', 'ttest', 'wilcoxon', 'bengio'], \
            f'method must be either "delong", "ttest", "wilcoxon" or "bengio". Given: {method}'
        
        # Get p-value
        if method == 'delong':
            return self.get_delong_p_value(aggregate)
        elif method == 'ttest':
            return self.get_ttest_p_value(metric)
        elif method == 'wilcoxon':
            return self.get_wilcoxin_p_value(metric)
        elif method == 'bengio':
            return self.get_bengio_p_value()
