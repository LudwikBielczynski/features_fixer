from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from logging import Logger

class FeatureFixerAbstract(ABC):

    @abstractmethod
    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    @abstractmethod
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


class FeatureFixer(FeatureFixerAbstract):

    def __init__(self, logger: Logger):
        self.logger = logger

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        return super().standardize(df)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        return super().normalize(df)