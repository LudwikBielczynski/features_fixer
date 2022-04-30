from typing import Optional

from sklearn import preprocessing
import pandas as pd

from .abstract import ScalerAbstract

class Normalizer(ScalerAbstract):
    '''The idea behind normalization is to scale sample to a unit norm (l2)'''
    scaler = preprocessing.Normalizer()

    def transform(self,
                  df: pd.DataFrame,
                  columns: Optional[list[str]] = None,
                  transpose: bool = True,
                  ) -> pd.DataFrame:
        '''Needs to transpose the data'''
        return super().transform(df, columns, transpose)

    def inverse_transform(self,
                          df: pd.DataFrame,
                          columns: Optional[list[str]] = None,
                          transpose: bool = True,
                          ) -> pd.DataFrame:
        return super().inverse_transform(df, columns, transpose)
