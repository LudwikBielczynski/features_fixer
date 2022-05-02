from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.decomposition import TruncatedSVD

from .abstract import ReducerAbstract

class SVD(ReducerAbstract):
    reducer: TruncatedSVD

    def transform(self, df: pd.DataFrame, components_number: int = 2) -> pd.DataFrame:
        '''
        Mostly used for sparse data. If dense data PCA often yields better results. Needs a number
        of components to be selected. Cross-validation with a different amount of components is
        advised to find the best number.
        '''
        self.reducer = TruncatedSVD(n_components=components_number)
        self.reducer.fit(X=df)
        df_reduced = self.reducer.transform(df)

        return df_reduced
