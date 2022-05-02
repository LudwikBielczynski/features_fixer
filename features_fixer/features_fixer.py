from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union

import pandas as pd

if TYPE_CHECKING:
    from logging import Logger
    from features_fixer.scaler import Standardizer, Normalizer
    from features_fixer.reducer import PCA, SVD, LowVariance

class FeaturesFixerAbstract(ABC):

    @abstractmethod
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    @abstractmethod
    def reduce_features_number(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

class FeaturesFixer(FeaturesFixerAbstract):

    def __init__(self,
                 logger: 'Logger',
                 scaler: Optional[Union['Normalizer', 'Standardizer']] = None,
                 reducer: Optional[Union['PCA', 'SVD', 'LowVariance']] = None
                 ) -> None:
        self.logger = logger
        self.scaler = scaler
        self.reducer = reducer

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            raise Exception('Missing any scaler')

        else:
            df = self.scaler.transform(df)
            self.logger.info('Scaling performed')

        return df

    def reduce_features_number(self, df: pd.DataFrame) -> pd.DataFrame:
        features_nr_initial = df.shape[1]
        if self.reducer is None:
            raise Exception('No object to reduce features number of the dataframe was given')

        else:
            df = self.reducer.transform(df)
            self.logger.info(f'Reduced features number from {features_nr_initial} to {df.shape[1]}')

        return df
