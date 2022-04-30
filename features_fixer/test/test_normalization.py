import pandas as pd
import pytest

from features_fixer.scaler import Normalizer

def test_basic_normalization():
    # Arrange
    normalizer = Normalizer()
    df = pd.DataFrame({'a': [4, 1, 2, 2]})

    # Act
    df_normalized = normalizer.transform(df)

    assert df_normalized['a'].mean() == pytest.approx(0., 0.01)
    assert df_normalized['a'].std() == pytest.approx(1., 0.3)

def test_selected_column():
    # Arrange
    standardizer = Normalizer()
    values = [0, 1, 2, 3, 5]
    df = pd.DataFrame({'a': values,
                       'b': values
                       })

    # Act
    df_standardize = standardizer.transform(df, columns=['a'])

    assert df_standardize['a'].mean() == pytest.approx(0., 0.01)
    assert df_standardize['a'].std() == pytest.approx(1., 0.3)

    assert df_standardize['b'].tolist() == values
