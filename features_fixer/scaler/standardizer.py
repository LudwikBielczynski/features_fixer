from typing import Optional

import pandas as pd
from sklearn import preprocessing

from .abstract import ScalerAbstract

class Standardizer(ScalerAbstract):
    '''The idea behind standarization is to scale the data to have a zero mean and a unit variance'''
    scaler = preprocessing.StandardScaler()
