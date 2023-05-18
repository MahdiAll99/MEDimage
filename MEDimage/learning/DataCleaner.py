from typing import Dict

from .cleaning_utils import *


class DataCleaner:
    """
    Class that will clean features of the csv
    """
    def __init__(self, variable_table: pd.DataFrame, var_names: list = [], type: str = "continuous"):
        """
        Constructor of the class DataCleaner

        Args:
            variable_table (pd.DataFrame): Table of variables.
            var_names (list): List of variable names.
            type (str): Type of variable: "continuous", "hcategorical" or "icategorical". Defaults to "continuous".
        """
        self.variable_table = variable_table
        self.var_names = var_names
        self.type = type
    
    def __update_variable_table(self, variable_table: pd.DataFrame, var_of_type: List, flag_var_out: List):
        """
        Updates the variable table by deleting the variables that are not in the variable of type

        Args:
            variable_table (pd.DataFrame): Table of variables.
            var_of_type (List[str]): List of variable names.
            flag_var_out (List[bool]): List of varaibles to flag out.
        
        Returns:
            pd.DataFrame: Table of variables with ``flag_var_out`` variables removed.
        """
        var_to_delete = np.delete(var_of_type, [i for i, v in enumerate(flag_var_out) if not v])
        var_of_type = np.delete(var_of_type, [i for i, v in enumerate(flag_var_out) if v])
        return variable_table.drop(var_to_delete, axis=1), var_of_type

    def cut_off_missing_per_sample(self, variable_table: pd.DataFrame, var_of_type: List, missing_cutoff : float = 0.25):
        """
        Removes observations/samples with more than ``missing_cutoff`` missing variables.

        Args:
            variable_table (pd.DataFrame): Table of variables.
            var_of_type (List[str]): List of variable names.
            missing_cutoff (float): Maximum percentage cut-offs of missing variables per sample.
        
        Returns:
            variable_table (pd.DataFrame): Table of variables with samples with more than ``missing_cutoff`` 
                missing variables removed.
        """
        # Initialization
        n_features = variable_table.shape[1]
        n_observation = variable_table.shape[0]
        empty_vec = np.zeros(n_observation, dtype=np.int)
        data = variable_table[var_of_type]
        empty_vec += data.isna().sum(axis=1).values
        
        # Gathering results
        ind_obs_out = np.where(((empty_vec/n_features) > missing_cutoff) == True)
        variable_table = variable_table.drop(variable_table.index[ind_obs_out])

        return variable_table
    
    def cut_off_missing_per_feature(self, variable_table: pd.DataFrame, var_of_type: List, missing_cutoff : float = 0.1):
        """
        Removes variables with more than ``missing_cutoff`` missing samples.

        Args:
            variable_table (pd.DataFrame): Table of variables.
            var_of_type (list): List of variable names.
            missing_cutoff (float): maximal percentage cut-offs of missing patient samples per variable.
        
        Returns:
            variable_table (pd.DataFrame): Table of variables with variables with more than ``missing_cutoff``
                missing observations removed.
        """
        flag_var_out = (((variable_table[var_of_type].isna().sum()) / variable_table.shape[0]) > missing_cutoff)
        return self.__update_variable_table(variable_table, var_of_type, flag_var_out)

    def cut_off_variation(self, variable_table: pd.DataFrame, var_of_type: List, cov_cutoff : float = 0.1):
        """
        Removes variables with a coefficient of variation (std/mean) less than ``cov_cutoff``.

        Args:
            variable_table (pd.DataFrame): Table of variables.
            var_of_type (list): List of variable names.
            cov_cutoff (float): minimal coefficient of variation cut-offs over samples per variable.
        
        Returns:
            variable_table (pd.DataFrame): Table of variables with variables with a coefficient of variation
                less than ``cov_cutoff`` removed.
        """
        eps = np.finfo(np.float32).eps
        var_variable_table = (variable_table[var_of_type].std(skipna=True) / variable_table[var_of_type].mean(skipna=True))
        flag_var_out = var_variable_table.abs().add(eps) < cov_cutoff
        variable_table, var_of_type = self.__update_variable_table(variable_table, var_of_type, flag_var_out)
        return variable_table, var_of_type
    
    def impute_missing(self, variable_table: pd.DataFrame, var_of_type: List, imputation_method : str = "mean"):
        """
        Imputes missing values of the variables of type.

        Args:
            variable_table (pd.DataFrame): Table of variables.
            var_of_type (list): List of variable names.
            imputation_method (str): Method of imputation. Can be "mean", "median" or "mode".
        
        Returns:
            variable_table (pd.DataFrame): Table of variables with missing values imputated.
        """
        if self.type in ['continuous', 'hcategorical']:
            # random imputation
            if 'random' in imputation_method:
                if len(imputation_method) > 6:
                    try:
                        seed = int(imputation_method[7:])
                        random.seed(seed)
                    except:
                        print("Warning: Seed must be an integer. Random seed will be set to None.")
                        random.seed(a=None)
                else:
                    random.seed(a=None)
                variable_table[var_of_type] = variable_table[var_of_type].apply(lambda x: x.fillna(random.choice(list(x.dropna(axis=0)))))
            
            # Imputation with median
            elif 'median' in imputation_method:
                variable_table[var_of_type] = variable_table[var_of_type].fillna(variable_table[var_of_type].median())
            
            # Imputation with mean
            elif 'mean' in imputation_method:
                variable_table[var_of_type] = variable_table[var_of_type].fillna(variable_table[var_of_type].mean())
            
            else:
                raise ValueError("Imputation method for continuous and hcategorical variables must be 'random', 'median' or 'mean'.")
        elif self.type in ['icategorical']:
            if 'random' in imputation_method:
                if len(imputation_method) > 6:
                    seed = int(imputation_method[7:])
                    random.seed(seed)
                else:
                    random.seed(a=None)

                variable_table[var_of_type] = variable_table[var_of_type].apply(lambda x: x.fillna(random.choice(list(x.dropna(axis=0)))))

            if 'mode' in imputation_method:
                variable_table[var_of_type] = variable_table[var_of_type].fillna(variable_table[var_of_type].mode().max())
        else:
            raise ValueError("Variable type must be 'continuous', 'hcategorical' or 'icategorical'.")
        
        return variable_table
    
    def check_min_n_per_cat(self, min_n_per_cat):
        return check_min_n_per_cat(self.variable_table, self.var_names, min_n_per_cat, self.type)
    
    def check_max_percent_cat(self, max_percent_cat):
        return check_max_percent_cat(self.variable_table, self.var_names, max_percent_cat)
    
    def one_hot_encode_table(self):
        return one_hot_encode_table(self.variable_table)

    def apply_data_cleaning(self, cleaning_dict: Dict, 
                            variable_table: pd.DataFrame = None,
                            imputation_method: str = "mean", missing_cutoff_ps: float = 0.25, 
                            missing_cutoff_pf: float = 0.1, cov_cutoff:float = 0.1) -> pd.DataFrame:
        """
        Applies data cleaning to the variables of type.

        Args:
            cleaning_dict (dict): Dictionary of cleaning parameters (missing cutoffs and coefficient of variation cutoffs etc.).
            variable_table (pd.DataFrame, optional): Table of variables.
            var_of_type (list, optional): List of variable names.
            imputation_method (str, optional): Method of imputation. Can be "mean", "median" or "mode".
            missing_cutoff_ps (float, optional): maximal percentage cut-offs of missing variables per sample.
            missing_cutoff_pf (float, optional): maximal percentage cut-offs of missing samples per variable.
            cov_cutoff (float, optional): minimal coefficient of variation cut-offs over samples per variable.
        
        Returns:
            variable_table (pd.DataFrame): Table of variables with data cleaning applied.
        """

        # Initialization
        if variable_table is None:
            if self.variable_table is None:
                raise ValueError("Variable table must be provided.")
            else:
                variable_table = self.variable_table

        var_of_type = variable_table.Properties['userData']['variables']['continuous']

        # Retrieve thresholds from cleaning_dict if not None
        if cleaning_dict is not None:
            missing_cutoff_pf = cleaning_dict['missingCutoffpf']
            missing_cutoff_ps = cleaning_dict['missingCutoffps']
            cov_cutoff = cleaning_dict['covCutoff']
            imputation_method = cleaning_dict['imputation']

        # Remove variables with more than missing_cutoff_pf missing samples (NaNs)
        variable_table, var_of_type = self.cut_off_missing_per_feature(variable_table, var_of_type, missing_cutoff_pf)

        # Remove variables with a coefficient of variation less than cov_cutoff
        variable_table, var_of_type = self.cut_off_variation(variable_table, var_of_type, cov_cutoff)

        # Remove scans with more than missing_cutoff_ps missing variables
        variable_table = self.cut_off_missing_per_sample(variable_table, var_of_type, missing_cutoff_ps)

        # Impute missing values
        variable_table = self.impute_missing(variable_table, var_of_type, imputation_method)

        return variable_table
