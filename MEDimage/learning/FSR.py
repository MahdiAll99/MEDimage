from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numpyencoder import NumpyEncoder

from MEDimage.learning.ml_utils import (combine_rad_tables, finalize_rad_table,
                                        get_stratified_splits,
                                        intersect_var_tables)
from MEDimage.utils.get_full_rad_names import get_full_rad_names
from MEDimage.utils.json_utils import save_json


class FSR:
    def __init__(self, method: str = 'fda') -> None:
        """
        Feature set reduction class constructor.

        Args:
            method (str): Method of feature set reduction. Can be "FDA", "LASSO" or "mRMR".
        """
        self.method = method

    def __get_fda_corr_table(
            self, 
            variable_table: pd.DataFrame, 
            outcome_table_binary: pd.DataFrame,
            n_splits: int, 
            corr_type: str, 
            seed: int
        ) -> pd.DataFrame:
        """
        Calculates the correlation table of the FDA algorithm.

        Args:
            variable_table (pd.DataFrame): variable table to check for stability.
            outcome_table_binary (pd.DataFrame): outcome table with binary labels.
            n_splits (int): Number of splits in the FDA algorithm (Ex: 100).
            corr_type: String specifying the correlation type that we are investigating. 
                Must be either 'Pearson' or 'Spearman'.
            seed (int): Random generator seed.
        
        Returns:
            pd.DataFrame: Correlation table of the FDA algorithm. Rows are splits, columns are features.
        """
        # Setting the seed
        np.random.seed(seed)

        # Initialization
        row_names = []
        corr_table = pd.DataFrame()
        fraction_for_splits = 1/3
        number_of_splits = 1

        # For each split, we calculate the correlation table
        for s in range(n_splits):
            row_names.append("Split_{0:03}".format(s))

            # Keep only variables that are in both tables
            _, outcome_table_binary = intersect_var_tables(variable_table, outcome_table_binary)

            # Under-sample the outcome table to equalize the number of positive and negative outcomes
            #outcome_table_binary_balanced = under_sample(outcome_table_binary)

            # Get the patient teach split
            patients_teach_splits = get_stratified_splits(
                outcome_table_binary, 
                number_of_splits, 
                fraction_for_splits, 
                seed, 
                flag_by_cat=True
            )[0]

            # Creating a table with both the variables and the outcome with
            # only the patient teach splits, ranked for spearman and not for pearson
            if corr_type == 'Spearman':
                full_table = pd.concat([variable_table.loc[patients_teach_splits, :].rank(),
                                        outcome_table_binary.loc[patients_teach_splits,
                                        outcome_table_binary.columns.values[-1]]], axis=1)

            elif corr_type == 'Pearson':
                # Pearson is the base method used by numpy, so we dont have to do any
                # manipulations to the data  like with spearman.
                full_table = pd.concat([variable_table.loc[patients_teach_splits, :],
                                        outcome_table_binary.loc[patients_teach_splits,
                                        outcome_table_binary.columns.values[-1]]], axis=1)
            else:
                raise ValueError("Correlation type not recognized. Please use 'Pearson' or 'Spearman'")

            # calculate the whole correlation table for all variables.
            full_table = np.corrcoef(full_table, rowvar=False)[-1][:-1].reshape((1, -1))
            corr_table = corr_table.append(pd.DataFrame(full_table))

        # Add the metadata to the correlation table
        corr_table.columns = list(variable_table.columns.values)
        corr_table = corr_table.fillna(0)
        corr_table.index = row_names
        corr_table.Properties = {}
        corr_table._metadata += ['Properties']
        corr_table.Properties['description'] = variable_table.Properties['Description']
        corr_table.Properties['userData'] = variable_table.Properties['userData']

        return corr_table
    
    def __find_fda_best_mean(self, corr_tables: pd.DataFrame, min_n_feat_stable: int) -> Tuple[Dict, pd.DataFrame]:
        """
        Finds the best mean correlation of all the stable variables in the table.

        Args:
            corr_tables (Dict): dictionary containing the correlation tables of
                dimension : [n_splits,n_features] for each table.
            min_n_feat_stable (int): minimal number of stable features.

        Returns:
            Tuple[Dict, pd.DataFrame]: Dict containing the name of each stable variables in every table and 
                pd.DataFrame containing the mean correlation of all the stable variables in the table.
        """
        # Initialization
        var_names_stable = {}
        corr_mean_stable = corr_tables
        n_features = 0
        corr_table = corr_tables
        corr_table = corr_table.fillna(0)

        # Calculation of the mean correlation among the n splits (R mean)
        var_names_stable = corr_table.index

        # Calculating the total number of features
        n_features += var_names_stable.size

        # Getting absolute values of the mean correlation
        corr_mean_stable_abs = corr_mean_stable.abs()

        # Keeping only the best features if there are more than min_n_feat_stable features
        if n_features > min_n_feat_stable:
            # Get min_n_feat_stable highest correlations
            best_features = corr_mean_stable_abs.sort_values(ascending=False)[0:min_n_feat_stable]
            var_names_stable = best_features.index.values
            corr_mean_stable = best_features

        return var_names_stable, corr_mean_stable

    def __find_fda_stable(self, corr_table: pd.DataFrame, thresh_stable: float) -> Tuple[Dict, pd.DataFrame]:
        """
        Finds the stable features in each correlation table 
        and the mean correlation of all the stable variables in the table.
        
        Args:
            corr_tables (Dict): dictionary containing the correlation tables of
                dimension : [n_splits,n_features] for each table.
            thresh_stable (float): the threshold deciding if a feature is stable.
        
        Returns:
            Tuple[Dict, pd.DataFrame]: dictionary containing the name of each stable variables in every tables 
                and table containing the mean correlation of all the stable variables in the table. 
                (The keys are the table names and the values are pd.Series).
        """

        # Initialization
        corr_table.fillna(0, inplace=True)

        # Calculation of R mean
        corr_mean_stable = corr_table.mean()
        mean_r = corr_mean_stable

        # Calculation of min and max
        min_r = corr_table.quantile(0.05)
        max_r = corr_table.quantile(0.95)

        # Calculation of unstable features
        unstable = (min_r < thresh_stable) & (mean_r > 0) | (max_r > -thresh_stable) & (mean_r < 0)
        ind_unstable = unstable.index[unstable]

        # Stable variables
        var_names_stable = unstable.index[~unstable].values
        corr_mean_stable = mean_r.drop(ind_unstable)

        return var_names_stable, corr_mean_stable

    def __keep_best_text_param(
            self, 
            corr_table: pd.DataFrame, 
            var_names_stable: List, 
            corr_mean_stable: pd.DataFrame
        ) -> Tuple[List, pd.DataFrame]:
        """
        Keeps the best texture features extraction parameters in the correlation tables
        by dropping the variants of a given feature.

        Args:
            corr_table (pd.DataFrame): Correlation table of dimension : [n_splits,n_features].
            var_names_stable (List): List of the stable variables in the table.
            corr_mean_stable (pd.DataFrame): Table of the mean correlation of the stable variables in the variables table.
        
        Returns:
            Tuple[List, pd.DataFrame]: list of the stable variables in the tables and table containing the mean 
                correlation of all the stable variables.
        """

        # If no stable features for the currect field, continue
        if var_names_stable.size == 0:
            return var_names_stable, corr_mean_stable

        # Get the actual radiomics features names from the sequential names
        full_rad_names = get_full_rad_names( 
            corr_table.Properties['userData']['variables']['var_def'], 
            var_names_stable)

        # Now parsing the full names to get only the rad names and not the variant
        rad_names = np.array([])
        for n in range(full_rad_names.size):
            rad_names = np.append(rad_names, full_rad_names[n].split('__')[1:2])

        # Verifying if two features are the same variant and keeping the best one
        n_var = rad_names.size
        var_to_drop = []
        for rad_name in rad_names:
            # If all the features are unique, break
            if np.unique(rad_names).size == n_var:
                break
            else:
                ind_same = np.where(rad_names == rad_name)[0]
                n_same = ind_same.size
                if n_same > 1:
                    var_to_drop.append(list(corr_mean_stable.iloc[ind_same].sort_values().index[1:].values))
        
        # Dropping the variants
        if len(var_to_drop) > 0:
            # convert to list of lists to list
            var_to_drop = [item for sublist in var_to_drop for item in sublist]
            
            # From the unique values of var_to_drop, drop the variants
            for var in set(var_to_drop):
                var_names_stable = np.delete(var_names_stable, np.where(var_names_stable == var))
                corr_mean_stable = corr_mean_stable.drop(var)

        return var_names_stable, corr_mean_stable

    def __remove_correlated_variables(
            self, 
            variable_table: pd.DataFrame, 
            rank: pd.Series,
            corr_type: str, 
            thresh_inter_corr: float,
            min_n_feat_total: int
        ) -> pd.DataFrame:
        """
        Removes inter-correlated variables given a certain threshold.
        
        Args:
            variable_table (pd.DataFrame): variable table for which we want to remove intercorrelated variables.
                Size: N X M  (observations, features).
            rank (pd.Series): Vector of correlation values per feature (of size 1 X M).
            corr_type (str): String specifying the correlation type that we are investigating. 
                Must be 'Pearson' or 'Spearman'.
            thresh_inter_corr (float): Numerical value specifying the threshold above which two variables are 
                considered to be correlated.
            min_n_feat_total (int): Minimum number of features to keep in the table.
        
        Returns:
            pd.DataFrame: Final variable table with the least correlated variables that are kept.
        """
        # Initialization
        n_features = variable_table.shape[1]

        # Compute correlation matrix
        if corr_type == 'Spearman':
            corr_mat = abs(np.corrcoef(variable_table.rank(), rowvar=False))
        elif corr_type == 'Pearson':
            corr_mat = abs(np.corrcoef(variable_table, rowvar=False))
        else:
            raise ValueError('corr_type must be either "Pearson" or "Spearman"')

        # Set diagonal elements to Nans
        np.fill_diagonal(corr_mat, val=np.nan)

        # Calculate mean inter-variable correlation
        mean_corr = np.nanmean(corr_mat, axis=1)
        
        # Looping over all features once
        # rank variables once, for meaningful variable loop.
        ind_loop = pd.Series(mean_corr).rank(method="first") - 1
        # Create a copy of the correlation matrix (to be modified)
        corr_mat_temp = corr_mat.copy()
        while True:
            for f in range(n_features):
                # Use index loop if not NaN
                try:
                    i = int(ind_loop[f])
                except:
                    i = 0
                # Select the row of the current feature
                row = corr_mat_temp[i][:]
                correlated = 1*(row > thresh_inter_corr)  # to turn into integers

                # While the correlations are above the threshold for the select row, we select another row
                while sum(correlated) > 0 and np.isnan(row).sum != len(row):
                    # Find the variable with the highest correlation and drop the one with the lowest rank
                    ind_max = np.nanargmax(row)
                    ind_min = np.nanargmin(np.array([rank[i], rank[ind_max]]))
                    if ind_min == 0:
                        # Drop the current row if the current feature has the lowest correlation with outcome
                        corr_mat_temp[i][:] = np.nan
                        corr_mat_temp[:][i] = np.nan
                        row[:] = np.nan
                    else:
                        # Drop the feature with the highest correlation to the current feature with the lowest correlation with outcome
                        corr_mat_temp[ind_max][:] = np.nan
                        corr_mat_temp[:][ind_max] = np.nan
                        row[ind_max] = np.nan

                    # Update the correlated vector
                    correlated = row > thresh_inter_corr

                # If all the rows are NaN, we keep the variable with the highest rank
                if (1*np.isnan(corr_mat_temp)).sum() == corr_mat_temp.size:
                    ind_keep = np.nanargmax(rank)
                else:
                    ind_keep = list()
                    for row in range(corr_mat_temp.shape[0]):
                        if 1*np.isnan(corr_mat_temp[row][:]).sum() < corr_mat_temp.shape[1]:
                            ind_keep.append(row)

            # if ind_keep happens to be a numpy type convert it to list for better subscripting
            if isinstance(ind_keep, np.int64):
                ind_keep = [ind_keep.tolist()]  # work around
            elif isinstance(ind_keep, np.ndarray):
                ind_keep = ind_keep.tolist()

            # Update threshold if the number of variables is too small or too large
            if len(ind_keep) < min_n_feat_total:
                # Increase the threshold (less stringent)
                thresh_inter_corr = thresh_inter_corr + 0.05
                corr_mat_temp = corr_mat.copy() # reset the correlation matrix
            else:
                break
        
        # Make sure we have the best 
        if len(ind_keep) != min_n_feat_total:
            # Take the features with the highest rank
            ind_keep = sorted(ind_keep)[:min_n_feat_total]

        #  Creating new variable_table
        columns = [variable_table.columns[idx] for idx in ind_keep]
        variable_table = variable_table.loc[:, columns]
        
        return variable_table

    def apply_fda_one_space(
            self, 
            ml: Dict, 
            variable_table: List, 
            outcome_table_binary: pd.DataFrame,
            del_variants: bool = True,
            logging_dict: Dict = None
        ) -> List:
        """
        Applies false discovery avoidance method.

        Args:
            ml (dict): Machine learning dictionary containing the learning options.
            variable_table (List): Table of variables.
            outcome_table_binary (pd.DataFrame): Table of binary outcomes.
            del_variants (bool, optional): If True, will delete the variants of the same feature. Defaults to True.

        Returns:
            List: Table of variables after feature set reduction.
        """
        # Initilization
        n_splits = ml['fSetReduction']['FDA']['nSplits']
        corr_type = ml['fSetReduction']['FDA']['corrType']
        thresh_stable_start = ml['fSetReduction']['FDA']['threshStableStart']
        thresh_inter_corr = ml['fSetReduction']['FDA']['threshInterCorr']
        min_n_feat_stable = ml['fSetReduction']['FDA']['minNfeatStable']
        min_n_feat_total = ml['fSetReduction']['FDA']['minNfeat']
        seed = ml['fSetReduction']['FDA']['seed']

        # Initialization - logging
        if logging_dict is not None:
            table_level = variable_table.Properties['Description'].split('__')[-1]
            logging_dict['one_space']['unstable'][table_level] = {}
            logging_dict['one_space']['inter_corr'][table_level] = {}

        # Getting the correlation table for the radiomics table
        radiomics_table_temp = variable_table.copy()
        outcome_table_binary_temp = outcome_table_binary.copy()
        
        # Get the correlation table
        corr_table = self.__get_fda_corr_table(
            radiomics_table_temp, 
            outcome_table_binary_temp, 
            n_splits, 
            corr_type, 
            seed
        )

        # Calculating the total numbers of features
        feature_total = radiomics_table_temp.shape[1]

        # Cut unstable features (Rmin cut)
        if feature_total > min_n_feat_stable:
            # starting threshold (set by user)
            thresh_stable = thresh_stable_start
            while True:
                # find which features are stable
                var_names_stable, corrs_stable = self.__find_fda_stable(corr_table, thresh_stable)

                # Keep the best textural parameters per image space (deleting variants)
                if del_variants:
                    var_names_stable, corrs_stable = self.__keep_best_text_param(corr_table, var_names_stable, corrs_stable)

                # count the number of stable features
                n_stable = var_names_stable.size
                
                # stop if the minimum number of stable features is reached, if not, lower the threshold.
                if n_stable >= min_n_feat_stable:
                    break
                else:
                    thresh_stable = thresh_stable - 0.05
                
                # stop if the threshold is zero or below
                if thresh_stable <= 0:
                    break
                
            # take the best mean correlation
            if n_stable > min_n_feat_stable:
                var_names_stable, corr_mean_stable = self.__find_fda_best_mean(corrs_stable, min_n_feat_stable)
            else:
                # Compute mean correlation
                corr_mean_stable = corr_table.mean()

            # Finalize radiomics tables before inter-correlation cut
            if len(var_names_stable) > 0:
                var_names = var_names_stable
                if isinstance(radiomics_table_temp, pd.Series):
                    radiomics_table_temp = radiomics_table_temp[[var_names]]
                else:
                    radiomics_table_temp = radiomics_table_temp[var_names]
                radiomics_table_temp = finalize_rad_table(radiomics_table_temp)
            else:
                radiomics_table_temp = pd.DataFrame()
        else:
            # if there is less features than the minimal number, take them all
            n_stable = feature_total

            # Compute mean correlation
            corr_mean_stable = corr_table.mean()

        # Update logging
        if logging_dict is not None:
            logging_dict['one_space']['unstable'][table_level] = radiomics_table_temp.columns.shape[0]

        # Inter-Correlation Cut
        if radiomics_table_temp.shape[1] > 1 and n_stable > min_n_feat_total:
            radiomics_table_temp = self.__remove_correlated_variables(
                radiomics_table_temp, 
                corr_mean_stable.abs(), 
                corr_type, 
                thresh_inter_corr,
                min_n_feat_total
            )
            
            # Finalize radiomics table
            radiomics_table_temp = finalize_rad_table(radiomics_table_temp)
        
        # Update logging
        if logging_dict is not None:
            logging_dict['one_space']['inter_corr'][table_level] = get_full_rad_names(
                radiomics_table_temp.Properties['userData']['variables']['var_def'], 
                radiomics_table_temp.columns.values
            ).tolist()

        return radiomics_table_temp
    
    def apply_fda(
            self, 
            ml: Dict, 
            variable_table: List, 
            outcome_table_binary: pd.DataFrame,
            logging: bool = True,
            path_save_logging: Path = None
        ) -> List:
        """
        Applies false discovery avoidance method.

        Args:
            ml (dict): Machine learning dictionary containing the learning options.
            variable_table (List): Table of variables.
            outcome_table_binary (pd.DataFrame): Table of binary outcomes.
            logging (bool, optional): If True, will save a dict that tracks features selsected for each level. Defaults to True.
            path_save_logging (Path, optional): Path to save the logging dict. Defaults to None.

        Returns:
            List: Table of variables after feature set reduction.
        """
        # Initialization
        rad_tables = variable_table.copy()
        n_rad_tables = len(rad_tables)
        variable_tables = []
        logging_dict = {'one_space': {'unstable': {}, 'inter_corr': {}}, 'final': {}}

        # Apply FDA for each image space/radiomics table
        for r in range(n_rad_tables):
            if logging:
                variable_tables.append(self.apply_fda_one_space(ml, rad_tables[r], outcome_table_binary, logging_dict=logging_dict))
            else:
                variable_tables.append(self.apply_fda_one_space(ml, rad_tables[r], outcome_table_binary))
        
        # Combine radiomics tables
        variable_table = combine_rad_tables(variable_tables)

        # Apply FDA again on the combined radiomics table
        variable_table = self.apply_fda_one_space(ml, variable_table, outcome_table_binary, del_variants=False)

        # Update logging dict
        if logging:
            logging_dict['final'] = get_full_rad_names(variable_table.Properties['userData']['variables']['var_def'], 
                                                       variable_table.columns.values).tolist()
            if path_save_logging is not None:
                path_save_logging = Path(path_save_logging).parent / 'fda_logging_dict.json'
                save_json(path_save_logging, logging_dict, cls=NumpyEncoder)
        
        return variable_table
    
    def apply_fda_balanced(
            self, 
            ml: Dict, 
            variable_table: List, 
            outcome_table_binary: pd.DataFrame,
        ) -> List:
        """
        Applies false discovery avoidance method but balances the number of features on each level.

        Args:
            ml (dict): Machine learning dictionary containing the learning options.
            variable_table (List): Table of variables.
            outcome_table_binary (pd.DataFrame): Table of binary outcomes.
            logging (bool, optional): If True, will save a dict that tracks features selsected for each level. Defaults to True.
            path_save_logging (Path, optional): Path to save the logging dict. Defaults to None.

        Returns:
            List: Table of variables after feature set reduction.
        """
        # Initilization
        rad_tables = variable_table.copy()
        n_rad_tables = len(rad_tables)
        variable_tables_all_levels = []
        levels = [[], [], []]

        # Organize the tables by level
        for r in range(n_rad_tables):
            if 'morph' in rad_tables[r].Properties['Description'].lower():
                levels[0].append(rad_tables[r])
            elif 'intensity' in rad_tables[r].Properties['Description'].lower():
                levels[0].append(rad_tables[r])
            elif 'texture' in rad_tables[r].Properties['Description'].lower():
                levels[0].append(rad_tables[r])
            elif 'mean' in rad_tables[r].Properties['Description'].lower() or \
                 'laws' in rad_tables[r].Properties['Description'].lower() or \
                 'log' in rad_tables[r].Properties['Description'].lower() or \
                 'gabor' in rad_tables[r].Properties['Description'].lower() or \
                 'coif' in rad_tables[r].Properties['Description'].lower() or \
                 'wavelet' in rad_tables[r].Properties['Description'].lower():
                levels[1].append(rad_tables[r])
            elif 'glcm' in rad_tables[r].Properties['Description'].lower():
                levels[2].append(rad_tables[r])

        # Apply FDA for each image space/radiomics table for each level
        for level in levels:
            variable_tables = []
            if len(level) == 0:
                continue
            for r in range(len(level)):
                variable_tables.append(self.apply_fda_one_space(ml, level[r], outcome_table_binary))
            
            # Combine radiomics tables
            variable_table = combine_rad_tables(variable_tables)

            # Apply FDA again on the combined radiomics table
            variable_table = self.apply_fda_one_space(ml, variable_table, outcome_table_binary, del_variants=False)

            # Add-up the tables
            variable_tables_all_levels.append(variable_table)
        
        # Combine radiomics tables of all 3 major levels (original, linear filters and textures)
        variable_table_all_levels = combine_rad_tables(variable_tables_all_levels)

        # Apply FDA again on the combined radiomics table
        variable_table_all_levels = self.apply_fda_one_space(ml, variable_table_all_levels, outcome_table_binary, del_variants=False)
        
        return variable_table_all_levels

    def apply_random_fsr_one_space(
            self,
            ml: Dict,
            variable_table: pd.DataFrame,
        ) -> List:
        seed = ml['fSetReduction']['FDA']['seed']
        
        # Setting the seed
        np.random.seed(seed)

        # Random select 10 columns (features)
        random_df = np.random.choice(variable_table.columns.values.tolist(), 10, replace=False)
        random_df = variable_table[random_df]

        return finalize_rad_table(random_df)
    
    def apply_random_fsr(
            self, 
            ml: Dict, 
            variable_table: List, 
        ) -> List:
        """
        Applies random feature set reduction by choosing a random number of features.

        Args:
            ml (dict): Machine learning dictionary containing the learning options.
            variable_table (List): Table of variables.
            outcome_table_binary (pd.DataFrame): Table of binary outcomes.

        Returns:
            List: Table of variables after feature set reduction.
        """
        # Iinitilization
        rad_tables = variable_table.copy()
        n_rad_tables = len(rad_tables)
        variable_tables = []

        # Apply FDA for each image space/radiomics table
        for r in range(n_rad_tables):
            variable_tables.append(self.apply_random_fsr_one_space(ml, rad_tables[r]))
        
        # Combine radiomics tables
        variable_table = combine_rad_tables(variable_tables)

        # Apply FDA again on the combined radiomics table
        variable_table = self.apply_random_fsr_one_space(ml, variable_table)
        
        return variable_table
    
    def apply_fsr(self, ml: Dict, variable_table: List, outcome_table_binary: pd.DataFrame, path_save_logging: Path = None) -> List:
        """
        Applies feature set reduction method.

        Args:
            ml (dict): Machine learning dictionary containing the learning options.
            variable_table (List): Table of variables.
            outcome_table_binary (pd.DataFrame): Table of binary outcomes.

        Returns:
            List: Table of variables after feature set reduction.
        """
        if self.method.lower() == "fda":
            variable_table = self.apply_fda(ml, variable_table, outcome_table_binary, path_save_logging=path_save_logging)
        elif self.method.lower() == "fdabalanced":
            variable_table = self.apply_fda_balanced(ml, variable_table, outcome_table_binary)
        elif self.method.lower() == "random":
            variable_table = self.apply_random_fsr(ml, variable_table)
        elif self.method == "LASSO":
            raise NotImplementedError("LASSO not implemented yet.")
        elif self.method == "mRMR":
            raise NotImplementedError("mRMR not implemented yet.")
        else:
            raise ValueError("FSR method is None or unknown: " + self.method)
        return variable_table
