from sklearn import preprocessing

from .abstract import ScalerAbstract

class Normalizer(ScalerAbstract):
    '''The idea behind normalization is to scale sample to a unit norm (l2)'''
    scaler = preprocessing.Normalizer()
