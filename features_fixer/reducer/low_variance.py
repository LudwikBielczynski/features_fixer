from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from .abstract import ReducerAbstract

class LowVariance(ReducerAbstract):
    reducer: VarianceThreshold

    def transform(self,
                  df: pd.DataFrame,
                  threshold: int = 0,
                  ) -> pd.DataFrame:
        '''
        Uses  automatic principal components number selection from Minka, 2000 "Automatic choice
        of dimensionality for PCA". Yields best results for dense data.
        '''
        self.reducer = VarianceThreshold(threshold)
        self.reducer.fit(X=df)
        df_reduced = self.reducer.transform(df)

        return df_reduced
