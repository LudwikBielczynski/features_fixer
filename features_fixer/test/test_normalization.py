import numpy as np
import pandas as pd

from features_fixer.scaler import Normalizer

def test_basic_normalization():
    # Arrange
    scaler = Normalizer()
    df = pd.DataFrame({'a': [4, 1, 2, 2]})

    # Act
    df_normalized = scaler.transform(df.copy(deep=True))

    # Assert
    norm_l2 = np.sqrt(np.square(df_normalized).sum(axis=0))['a']
    assert norm_l2 == 1.0

def test_selected_column():
    # Arrange
    scaler = Normalizer()
    values = [4, 1, 2, 2]
    df = pd.DataFrame({'a': values,
                       'b': values
                       })

    # Act
    df_normalized = scaler.transform(df, columns=['a'])

    # Assert
    norm_l2 = np.sqrt(np.square(df_normalized).sum(axis=0))
    assert norm_l2['a'] == 1.0
    assert norm_l2['b'] != 1.0
