import numpy as np
import pandas as pd
from neuroCombat import neuroCombat

from ..utils.get_institutions_from_ids import get_institutions_from_ids


class Normalization:
    def __init__(
            self, 
            method: str = 'combat', 
            variable_table: pd.DataFrame = None, 
            covariates_df: pd.DataFrame = None, 
            institutions: list = None
        ) -> None:
        """
        Constructor of the Normalization class.
        """
        self.method = method
        self.variable_table = variable_table
        self.covariates_df = covariates_df
        self.institutions = institutions
    
    def apply_combat(
            self, 
            variable_table: pd.DataFrame, 
            covariate_df: pd.DataFrame = None, 
            institutions: list = None
        ) -> pd.DataFrame:
        """
        Applys ComBat Normalization method to the data.
        More details :ref:`this link <https://github.com/Jfortin1/ComBatHarmonization/tree/master/Python>`.

        Args:
            variable_table (pd.DataFrame): pandas data frame on which Combat harmonization will be applied. 
                This table is of size N X F (Observations X Features) and has the IDs as index. 
                Requirements for this table

                    - Does not contain NaNs.
                    - No feature has 0 variance.
                    - All variables are continuous (For example: Radiomics variables).
            covariate_df (pd.DataFrame, optional): N X M pandas data frame, where N must equal the number of 
                observations in variable_table. M is the number of covariates to include in the algorithm.
            institutions (list, optional): List of size n_observations X 1 with the different institutions.
        
        Returns:
            pd.DataFrame: variable_table after Combat harmonization.
        """
        # Initializing the class attributes from the arguments
        if variable_table is None:
            if self.variable_table is None:
                raise ValueError('variable_table must be given.')
        else:
            self.variable_table = variable_table
        if covariate_df is not None:
            self.covariates_df = covariate_df
        if institutions:
            self.institutions = institutions
        
        # Intializing the institutions if not given
        if self.institutions is None:
            patient_ids = pd.Series(self.variable_table.index)
            self.institutions = get_institutions_from_ids(patient_ids)
            all_institutions = self.institutions.unique()
            for n in range(all_institutions.size):
                self.institutions[self.institutions == all_institutions[n]] = n+1
        self.institutions = self.institutions.to_numpy(dtype=int)
        self.institutions = np.reshape(self.institutions, (-1, 1))
        
        # No harmonization will be applied if there is only one institution
        if np.unique(self.institutions).size < 2:
            return self.variable_table
        
        # Initializing the covariates if not given
        if self.covariates_df is not None:
            self.covariates_df['institution'] = self.institutions
        else:
            # the covars matrix is only a row with the institution
            self.covariates_df = pd.DataFrame(
                self.institutions, 
                columns=['institution'], 
                index=self.variable_table.index.values
            )

        # Apply combat
        n_features = self.variable_table.shape[1]
        batch_col = 'institution'
        if n_features == 1:
            # combat does not work with a single feature so a temporary one is added,
            # then removed later (this has no effect on the algorithm).
            self.variable_table['temp'] = pd.Series(
                np.ones(self.variable_table.shape[0]),
                index=self.variable_table.index
            )
            data_combat = neuroCombat(
                self.variable_table.transpose(), 
                self.covariates_df, 
                batch_col
            )
            self.variable_table = pd.DataFrame(self.variable_table.drop('temp', axis=1))
            vt_combat = pd.DataFrame(data_combat[:][0].transpose())
        else:
            data_combat = neuroCombat(
                self.variable_table.transpose(), 
                self.covariates_df, 
                batch_col
            )
            vt_combat = pd.DataFrame(data_combat['data']).transpose()

        self.variable_table[:] = vt_combat.values

        return self.variable_table
