import random
from typing import Dict, List

import numpy as np
import pandas as pd


class DataCleaner:
    """
    Class that will clean features of the csv by removing features with too many missing values,
    too little variation, too many missing values per sample, too little variation per sample,
    and imputing missing values.
    """
    def __init__(self, df_features: pd.DataFrame, type: str = "continuous"):
        """
        Constructor of the class DataCleaner

        Args:
            df_features (pd.DataFrame): Table of features.
            type (str): Type of variable: "continuous", "hcategorical" or "icategorical". Defaults to "continuous".
        """
        self.df_features = df_features
        self.type = type
    
    def __update_df_features(self, var_of_type: List[str], flag_var_out: List[bool]) -> List[str]:
        """
        Updates the variable table by deleting the features that are not in the variable of type

        Args:
            var_of_type (List[str]): List of variable names.
            flag_var_out (List[bool]): List of variables to flag out.
        
        Returns:
            List[str]: List of variable names that were not flagged out.
        """
        var_to_delete = np.delete(var_of_type, [i for i, v in enumerate(flag_var_out) if not v])
        var_of_type = np.delete(var_of_type, [i for i, v in enumerate(flag_var_out) if v])
        self.df_features = self.df_features.drop(var_to_delete, axis=1)
        return var_of_type

    def cut_off_missing_per_sample(self, var_of_type: List[str], missing_cutoff : float = 0.25) -> None:
        """
        Removes observations/samples with more than ``missing_cutoff`` missing features.

        Args:
            var_of_type (List[str]): List of variable names.
            missing_cutoff (float): Maximum percentage cut-offs of missing features per sample. Defaults to 25%.
        
        Returns:
            None.
        """
        # Initialization
        n_observation, n_features = self.df_features.shape
        empty_vec = np.zeros(n_observation, dtype=np.int)
        data = self.df_features[var_of_type]
        empty_vec += data.isna().sum(axis=1).values
        
        # Gathering results
        ind_obs_out = np.where(((empty_vec/n_features) > missing_cutoff) == True)
        self.df_features = self.df_features.drop(self.df_features.index[ind_obs_out])
    
    def cut_off_missing_per_feature(self, var_of_type: List[str], missing_cutoff : float = 0.1) -> List[str]:
        """
        Removes features with more than ``missing_cutoff`` missing patients.

        Args:
            var_of_type (list): List of variable names.
            missing_cutoff (float): maximal percentage cut-offs of missing patient samples per variable.
        
        Returns:
            List[str]: List of variable names that were not flagged out.
        """
        flag_var_out = (((self.df_features[var_of_type].isna().sum()) / self.df_features.shape[0]) > missing_cutoff)
        return self.__update_df_features(var_of_type, flag_var_out)

    def cut_off_variation(self, var_of_type: List[str], cov_cutoff : float = 0.1) -> List[str]:
        """
        Removes features with a coefficient of variation (cov) less than ``cov_cutoff``.

        Args:
            var_of_type (list): List of variable names.
            cov_cutoff (float): minimal coefficient of variation cut-offs over samples per variable. Defaults to 10%.
        
        Returns:
            List[str]: List of variable names that were not flagged out.
        """
        eps = np.finfo(np.float32).eps
        cov_df_features = (self.df_features[var_of_type].std(skipna=True) / self.df_features[var_of_type].mean(skipna=True))
        flag_var_out = cov_df_features.abs().add(eps) < cov_cutoff
        return self.__update_df_features(var_of_type, flag_var_out)
    
    def impute_missing(self, var_of_type: List[str], imputation_method : str = "mean") -> None:
        """
        Imputes missing values of the features of type.

        Args:
            var_of_type (list): List of variable names.
            imputation_method (str): Method of imputation. Can be "mean", "median", "mode" or "random".
                For "random" imputation, a seed can be provided by adding the seed value after the method 
                name, for example "random42".
        
        Returns:
            None.
        """
        if self.type in ['continuous', 'hcategorical']:
            # random imputation
            if 'random' in imputation_method:
                if len(imputation_method) > 6:
                    try:
                        seed = int(imputation_method[7:])
                        random.seed(seed)
                    except Exception as e:
                        print(f"Warning: Seed must be an integer. Random seed will be set to None. str({e})")
                        random.seed(a=None)
                else:
                    random.seed(a=None)
                self.df_features[var_of_type] = self.df_features[var_of_type].apply(lambda x: x.fillna(random.choice(list(x.dropna(axis=0)))))
            
            # Imputation with median
            elif 'median' in imputation_method:
                self.df_features[var_of_type] = self.df_features[var_of_type].fillna(self.df_features[var_of_type].median())
            
            # Imputation with mean
            elif 'mean' in imputation_method:
                self.df_features[var_of_type] = self.df_features[var_of_type].fillna(self.df_features[var_of_type].mean())
            
            else:
                raise ValueError("Imputation method for continuous and hcategorical features must be 'random', 'median' or 'mean'.")
        
        elif self.type in ['icategorical']:
            if 'random' in imputation_method:
                if len(imputation_method) > 6:
                    seed = int(imputation_method[7:])
                    random.seed(seed)
                else:
                    random.seed(a=None)

                self.df_features[var_of_type] = self.df_features[var_of_type].apply(lambda x: x.fillna(random.choice(list(x.dropna(axis=0)))))

            if 'mode' in imputation_method:
                self.df_features[var_of_type] = self.df_features[var_of_type].fillna(self.df_features[var_of_type].mode().max())
        else:
            raise ValueError("Variable type must be 'continuous', 'hcategorical' or 'icategorical'.")
        
    def __call__(self, cleaning_dict: Dict, imputation_method: str = "mean", 
                missing_cutoff_ps: float = 0.25, missing_cutoff_pf: float = 0.1, 
                cov_cutoff:float = 0.1) -> pd.DataFrame:
        """
        Applies data cleaning to the features of type.

        Args:
            cleaning_dict (dict): Dictionary of cleaning parameters (missing cutoffs and coefficient of variation cutoffs etc.).
            var_of_type (list, optional): List of variable names.
            imputation_method (str): Method of imputation. Can be "mean", "median", "mode" or "random".
                For "random" imputation, a seed can be provided by adding the seed value after the method 
                name, for example "random42".
            missing_cutoff_ps (float, optional): maximal percentage cut-offs of missing features per sample.
            missing_cutoff_pf (float, optional): maximal percentage cut-offs of missing samples per variable.
            cov_cutoff (float, optional): minimal coefficient of variation cut-offs over samples per variable.
        
        Returns:
            pd.DataFrame: Cleaned table of features.
        """

        # Initialization
        var_of_type = self.df_features.Properties['userData']['variables']['continuous']

        # Retrieve thresholds from cleaning_dict if not None
        if cleaning_dict is not None:
            missing_cutoff_pf = cleaning_dict['missingCutoffpf']
            missing_cutoff_ps = cleaning_dict['missingCutoffps']
            cov_cutoff = cleaning_dict['covCutoff']
            imputation_method = cleaning_dict['imputation']
        
        # Replace infinite values with NaNs
        self.df_features = self.df_features.replace([np.inf, -np.inf], np.nan)

        # Remove features with more than missing_cutoff_pf missing samples (NaNs)
        var_of_type = self.cut_off_missing_per_feature(var_of_type, missing_cutoff_pf)

        # Check
        if len(var_of_type) == 0:
            return None

        # Remove features with a coefficient of variation less than cov_cutoff
        var_of_type = self.cut_off_variation(var_of_type, cov_cutoff)

        # Check
        if len(var_of_type) == 0:
            return None

        # Remove scans with more than missing_cutoff_ps missing features
        self.cut_off_missing_per_sample(var_of_type, missing_cutoff_ps)

        # Impute missing values
        self.impute_missing(var_of_type, imputation_method)

        return self.df_features
