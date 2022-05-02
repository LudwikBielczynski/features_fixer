from logger_builder import LoggerBuilder
from logger_builder.formatter import create_basic_formatter
from logger_builder.handler import StreamHandler
import pandas as pd
from sklearn.datasets import load_diabetes

from features_fixer import FeaturesFixer
from features_fixer.reducer import PCA
from features_fixer.scaler import Standardizer

if __name__ == '__main__':
    # Initialize logging
    formatter = create_basic_formatter()
    stream_handler = StreamHandler(formatter)

    handlers = [stream_handler]
    logger_builder = LoggerBuilder(handlers)
    logger = logger_builder.create_logger('FeaturesFixer')

    # Initialize a scaler and dimensionality reducing objects
    scaler = Standardizer()
    reducer = PCA()

    ff = FeaturesFixer(logger, scaler=scaler, reducer=reducer)

    # Example on the diabetes dataset
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes['data'], columns=diabetes['feature_names'])
    df = ff.scale_features(df)
    df = ff.reduce_features_number(df)

    logger.info(df.head())