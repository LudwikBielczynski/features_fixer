import pandas as pd
import pytest

from features_fixer.scaler import Standardizer

def test_basic_standardization():
    # Arrange
    standardizer = Standardizer()
    df = pd.DataFrame({'a': [0, 1, 2, 3, 5]})

    # Act
    df_standardized = standardizer.transform(df)

    assert df_standardized['a'].mean() == pytest.approx(0., 0.01)
    assert df_standardized['a'].std() == pytest.approx(1., 0.3)

def test_selected_column():
    # Arrange
    standardizer = Standardizer()
    values = [0, 1, 2, 3, 5]
    df = pd.DataFrame({'a': values,
                       'b': values
                       })

    # Act
    df_standardized = standardizer.transform(df, columns=['a'])

    assert df_standardized['a'].mean() == pytest.approx(0., 0.01)
    assert df_standardized['a'].std() == pytest.approx(1., 0.3)

    assert df_standardized['b'].tolist() == values
