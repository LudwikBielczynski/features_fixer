from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
from sklearn.decomposition import PCA as PCASklearn

from .abstract import ReducerAbstract

class PCA(ReducerAbstract):
    reducer: PCASklearn

    def transform(self,
                  df: pd.DataFrame,
                  components_number: Union[str, int] = 'mle',
                  ) -> pd.DataFrame:
        '''
        Uses  automatic principal components number selection from Minka, 2000 "Automatic choice
        of dimensionality for PCA". Yields best results for dense data.
        '''
        self.reducer = PCASklearn(n_components=components_number)
        self.reducer.fit(X=df)
        df_reduced = self.reducer.transform(df)

        return df_reduced
