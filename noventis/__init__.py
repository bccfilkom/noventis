__version__ = "0.1.0"


from .data_cleaner.auto import data_cleaner
from .data_cleaner.encoding import NoventisEncoder
from .data_cleaner.scaling import NoventisScaler
from .data_cleaner.imputing import NoventisImputer
from .data_cleaner.outlier_handling import NoventisOutlierHandler


from .eda_auto.eda_auto import NoventisAutoEDA


from .predictor.manual import NoventisManualPredictor
from .predictor.auto import NoventisAutoML

__all__ = [
    "data_cleaner",
    "NoventisEncoder",
    "NoventisScaler",
    "NoventisImputer",
    "NoventisOutlierHandler",
    "NoventisAutoEDA",
    "NoventisManualPredictor",  
    "NoventisAutoML",
]

