<<<<<<< Updated upstream
# Mengimpor dari data_cleaner
from .eda_auto.eda_auto import NoventisAutoEDA
=======
__version__ = "0.1.0"

>>>>>>> Stashed changes
from .data_cleaner.auto import data_cleaner
from .data_cleaner.encoding import NoventisEncoder
from .data_cleaner.scaling import NoventisScaler
from .data_cleaner.imputing import NoventisImputer
from .data_cleaner.outlier_handling import NoventisOutlierHandler
<<<<<<< Updated upstream

=======
from .eda_auto.eda_auto import NoventisAutoEDA
>>>>>>> Stashed changes

from .predictor.manual import NoventisManualPredictor
from .predictor.auto import NoventisAutoML

__all__ = [
<<<<<<< Updated upstream
    'NoventisAutoEDA',
    'data_cleaner',
    'NoventisEncoder',
    'NoventisScaler',
    'NoventisImputer',
    'NoventisOutlierHandler',
    'ManualPredictor',
    'NoventisAutoML'
]
=======
    "data_cleaner",
    "NoventisEncoder",
    "NoventisScaler",
    "NoventisImputer",
    "NoventisOutlierHandler",
    "NoventisAutoEDA",
    "NoventisManualPredictor",  
    "NoventisAutoML",
]
>>>>>>> Stashed changes
