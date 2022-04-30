from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin

class ScalerAbstract(ABC):

    @property
    @abstractmethod
    def scaler(self) -> 'TransformerMixin':
        pass

    def _get_columns(self, df: pd.DataFrame, columns: Optional[list[str]] = None) -> list[str]:
        if columns is None:
            columns = df.columns
        return columns

    def transform(self,
                  df: pd.DataFrame,
                  columns: Optional[list[str]] = None,
                  transpose: bool = False,
                  ) -> pd.DataFrame:
        columns = self._get_columns(df, columns)

        if transpose:
            self.scaler.fit(df[columns].T)
            df.loc[:, columns] = self.scaler.transform(df[columns].T).T
        else:
            self.scaler.fit(df[columns])
            df.loc[:, columns] = self.scaler.transform(df[columns])

        return df

    def inverse_transform(self,
                          df: pd.DataFrame,
                          columns: Optional[list[str]] = None,
                          transpose: bool = False,
                          ) -> pd.DataFrame:
        columns = self._get_columns(df, columns)

        if transpose:
            df.loc[:, columns] = self.scaler.inverse_transform(df[columns].T).T
        else:
            df.loc[:, columns] = self.scaler.inverse_transform(df[columns])

        return df
