from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from logging import Logger
    from features_fixer.scaler import Standardizer, Normalizer

class FeatureFixerAbstract(ABC):

    @abstractmethod
    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    @abstractmethod
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

class FeatureFixer(FeatureFixerAbstract):

    def __init__(self,
                 logger: Logger,
                 standardizer: 'Standardizer',
                 normalizer: 'Normalizer',
                 ) -> None:
        self.logger = logger
        self.standardizer = standardizer
        self.normalizer = normalizer

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.standardizer.transform(df)
        return df

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.normalizer.transform(df)
        return df
