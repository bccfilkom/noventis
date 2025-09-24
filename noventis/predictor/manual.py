import pandas as pd
import numpy as np
import pickle
import time
import logging
import warnings
<<<<<<< HEAD
import yaml
import os
from typing import Dict, Any, List, Union, Optional, Tuple

import io
import base64
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
=======
import os
from datetime import datetime
from typing import Dict, Any, List, Union, Optional

import io
import base64
import matplotlib.pyplot as plt
from IPython.display import display, HTML
warnings.filterwarnings('ignore') # Suppress warnings for a cleaner output
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2

import optuna
import shap
<<<<<<< HEAD
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, RobustScaler
=======
optuna.logging.set_verbosity(optuna.logging.ERROR) # Show only errors from Optuna


from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_curve
)
from sklearn.ensemble import StackingClassifier, StackingRegressor

<<<<<<< HEAD
=======
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    ImbPipeline = Pipeline

from joblib import Memory
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

<<<<<<< HEAD
=======
# =======================================================================
#                         MODEL LIBRARIES
# =======================================================================
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
>>>>>>> b6f4cb8c8878da79e8254289c9869fb6d5e8bb73
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
<<<<<<< HEAD
=======

<<<<<<< HEAD
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.ERROR)
=======
# =======================================================================
#                      OPTIONAL DEPENDENCIES
# =======================================================================
>>>>>>> b6f4cb8c8878da79e8254289c9869fb6d5e8bb73
try:
    # Attempt to import a custom data cleaner if available
    from noventis_beta.data_cleaner import NoventisDataCleaner
except ImportError:
    NoventisDataCleaner = None # If not found, set to None to avoid errors

<<<<<<< HEAD
=======
# =======================================================================
#                          LOGGING SETUP
# =======================================================================
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
>>>>>>> b6f4cb8c8878da79e8254289c9869fb6d5e8bb73
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


<<<<<<< HEAD
=======
<<<<<<< HEAD
def get_rf_params(trial):
=======
# =======================================================================
#               MODEL & HYPERPARAMETER CONFIGURATION
# =======================================================================
# These functions define the hyperparameter search space for Optuna.
# Each function takes an Optuna 'trial' object and suggests parameter values.
>>>>>>> b6f4cb8c8878da79e8254289c9869fb6d5e8bb73

def get_rf_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines the search space for Random Forest Classifier."""
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
    return {
        'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 1000),
        'model__max_depth': trial.suggest_int('model__max_depth', 5, 50),
        'model__min_samples_split': trial.suggest_int('model__min_samples_split', 2, 20),
        'model__min_samples_leaf': trial.suggest_int('model__min_samples_leaf', 1, 10),
        'model__class_weight': trial.suggest_categorical('model__class_weight', ['balanced', None]),
    }

def get_xgb_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines the search space for XGBoost Classifier."""
    return {
<<<<<<< HEAD
        'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 2000),
        'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.3),
        'model__max_depth': trial.suggest_int('model__max_depth', 3, 12),
        'model__subsample': trial.suggest_float('model__subsample', 0.6, 1.0),
        'model__colsample_bytree': trial.suggest_float('model__colsample_bytree', 0.6, 1.0),
        'model__gamma': trial.suggest_float('model__gamma', 0, 5),
=======
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
    }

def get_lgbm_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines the search space for LightGBM Classifier."""
    return {
<<<<<<< HEAD
        'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 2000),
        'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.3),
        'model__num_leaves': trial.suggest_int('model__num_leaves', 20, 150),
        'model__max_depth': trial.suggest_int('model__max_depth', 3, 15),
        'model__reg_alpha': trial.suggest_float('model__reg_alpha', 0.0, 1.0),
        'model__reg_lambda': trial.suggest_float('model__reg_lambda', 0.0, 1.0),
        'model__is_unbalance': trial.suggest_categorical('model__is_unbalance', [True, False]),
=======
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'is_unbalance': trial.suggest_categorical('is_unbalance', [True, False]),
    }

def get_dt_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines the search space for Decision Tree (both Classifier and Regressor)."""
    return {
        'max_depth': trial.suggest_int('max_depth', 3, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
    }

def get_gb_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines the search space for Gradient Boosting (both Classifier and Regressor)."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
    }
    
def get_catboost_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines the search space for CatBoost (both Classifier and Regressor)."""
    return {
<<<<<<< HEAD
        'model__iterations': trial.suggest_int('model__iterations', 100, 1000),
        'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.3),
        'model__depth': trial.suggest_int('model__depth', 4, 10),
        'model__l2_leaf_reg': trial.suggest_float('model__l2_leaf_reg', 1.0, 10.0),
    }


def get_rf_reg_params(trial):
=======
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
    }

def get_rf_reg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines the search space for Random Forest Regressor."""
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
    return {
        'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 1000),
        'model__max_depth': trial.suggest_int('model__max_depth', 5, 50),
        'model__min_samples_split': trial.suggest_int('model__min_samples_split', 2, 20),
        'model__min_samples_leaf': trial.suggest_int('model__min_samples_leaf', 1, 10),
    }

def get_xgb_reg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines the search space for XGBoost Regressor."""
    return {
<<<<<<< HEAD
        'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 2000),
        'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.3),
        'model__max_depth': trial.suggest_int('model__max_depth', 3, 12),
        'model__subsample': trial.suggest_float('model__subsample', 0.6, 1.0),
        'model__colsample_bytree': trial.suggest_float('model__colsample_bytree', 0.6, 1.0),
=======
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
    }

def get_lgbm_reg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines the search space for LightGBM Regressor."""
    return {
<<<<<<< HEAD
        'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 2000),
        'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.3),
        'model__num_leaves': trial.suggest_int('model__num_leaves', 20, 150),
        'model__max_depth': trial.suggest_int('model__max_depth', 3, 15),
=======
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
    }

# Central registry mapping model names to their classes and parameter search spaces.
MODEL_CONFIG = {
    'classification': {
<<<<<<< HEAD
        'logistic_regression': {'model': LogisticRegression, 'params': None},
=======
        'logistic_regression': {'model': LogisticRegression, 'params': None}, # No tuning for this one
        'decision_tree': {'model': DecisionTreeClassifier, 'params': get_dt_params},
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
        'random_forest': {'model': RandomForestClassifier, 'params': get_rf_params},
        'xgboost': {'model': xgb.XGBClassifier, 'params': get_xgb_params},
        'lightgbm': {'model': lgb.LGBMClassifier, 'params': get_lgbm_params},
        'catboost': {'model': cb.CatBoostClassifier, 'params': get_catboost_params},

    },
    'regression': {
<<<<<<< HEAD
        'linear_regression': {'model': LinearRegression, 'params': None},
=======
        'linear_regression': {'model': LinearRegression, 'params': None}, # No tuning for this one
        'decision_tree': {'model': DecisionTreeRegressor, 'params': get_dt_params},
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
        'random_forest': {'model': RandomForestRegressor, 'params': get_rf_reg_params},
        'xgboost': {'model': xgb.XGBRegressor, 'params': get_xgb_reg_params},
        'lightgbm': {'model': lgb.LGBMRegressor, 'params': get_lgbm_reg_params},
        'catboost': {'model': cb.CatBoostRegressor, 'params': get_catboost_params},

    }
}
class ManualPredictor:
    """
    Automated Machine Learning Pipeline for Classification and Regression Tasks
    
    This class provides an end-to-end solution for:
    - Data preprocessing and feature engineering
    - Model training and hyperparameter optimization
    - Performance evaluation and comparison
    - Results visualization and reporting
    
    Parameters
    ----------
    model_name : str or List[str]
        Name(s) of the model(s) to train

    task : str
        Type of ML task ('classification' or 'regression')
        
    random_state : int, optional (default=42)
        Random seed for reproducibility
    
    data_cleaner : object, optional
        External data cleaning utility
    
    tune_hyperparameters : bool, optional (default=False)
        Whether to perform hyperparameter optimization
    
    n_trials : int, optional (default=50)
        Number of optimization trials if tuning is enabled
    
    cv_folds : int, optional (default=3)
        Number of cross-validation folds
    
    enable_feature_engineering : bool, optional (default=False)
        Whether to perform automated feature engineering
    
    cv_strategy : str, optional (default='repeated')
        Cross-validation strategy ('repeated' or 'stratified')
    
    show_tuning_plots : bool, optional (default=False)
        Whether to display optimization plots
    
    output_dir : str, optional
        Directory to save all outputs (plots, models, reports)
    
    Examples
    --------
    >>> predictor = ManualPredictor(
    ...     model_name=['random_forest', 'xgboost'],
    ...     task='classification',
    ...     tune_hyperparameters=True,
    ...     output_dir='outputs'
    ... )
    >>> results = predictor.run_pipeline(df, target_column='target')
    """
    def __init__(
        self,
        model_name: Union[str, List[str]],
        task: str,
        random_state: int = 42,
        tune_hyperparameters: bool = False,
        n_trials: int = 50,
        cv_folds: int = 3,
<<<<<<< HEAD
        imbalance_handler: Optional[str] = None,
        feature_engineering: Optional[List[str]] = None,
        show_tuning_plots: bool = False,
        enable_caching: bool = False,
        cache_dir: str = './.noventis_cache',
        use_mlflow: bool = False,
        mlflow_experiment_name: str = 'Noventis_Manual_Predictor'
=======
        enable_feature_engineering: bool = False,
        cv_strategy: str = 'repeated',
        show_tuning_plots: bool = False,
        output_dir: Optional[str] = None  
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
    ):
        # --- Konfigurasi Pipeline ---
        self.model_name = model_name
        self.task = task.lower()
        self.random_state = random_state

        # --- Konfigurasi Proses ---
        self.tune_hyperparameters = tune_hyperparameters
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.show_tuning_plots = show_tuning_plots
        
<<<<<<< HEAD
        # --- Fitur Lanjutan ---
        self.imbalance_handler = imbalance_handler
        self.feature_engineering = feature_engineering or []
        self.enable_caching = enable_caching
        self.cache_dir = cache_dir
        
        # --- MLOps (MLflow) ---
        self.use_mlflow = use_mlflow
        self.mlflow_experiment_name = mlflow_experiment_name

        # --- Atribut Internal untuk Menyimpan Hasil ---
        self.best_model_info_ = {}
        self.all_results_ = []
        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = None, None, None, None
        self.recommendations_ = []
        self.is_fitted_ = False
        
        # --- Validasi & Setup ---
        self._validate_params()
        self._setup_cache()
        if self.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.set_experiment(self.mlflow_experiment_name)
=======
        # Initialize output directory if provided
        self.output_dir = self._setup_output_directory(output_dir)

        # Attributes to store results
        self.best_model_info: Dict[str, Any] = {}
        self.all_results: List[Dict[str, Any]] = []
        self.X_train_final, self.X_test_final, self.y_test_final = None, None, None
        self.preprocessor: Optional[ColumnTransformer] = None # Attribute to store the fitted preprocessor
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2

    def _validate_params(self):
        if self.task not in ['classification', 'regression']:
            raise ValueError("Task harus 'classification' atau 'regression'.")
        if self.imbalance_handler and not IMBLEARN_AVAILABLE:
            raise ImportError("Paket 'imbalanced-learn' dibutuhkan untuk imbalance handling. Silakan install dengan 'pip install imbalanced-learn'.")
        if self.imbalance_handler and self.task == 'regression':
            logging.warning("Imbalance handler tidak relevan untuk task regresi dan akan diabaikan.")
            self.imbalance_handler = None
        if self.use_mlflow and not MLFLOW_AVAILABLE:
            logging.warning("Paket 'mlflow' tidak ditemukan. Logging dinonaktifkan. Install dengan 'pip install mlflow'.")
            self.use_mlflow = False

<<<<<<< HEAD
    def _setup_cache(self):
        if self.enable_caching:
            logging.info(f"Caching diaktifkan. Direktori cache: {self.cache_dir}")
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            self.memory = Memory(self.cache_dir, verbose=0)
        else:
            self.memory = None

    def _get_base_model(self, name: str) -> Any:
=======
    def _setup_output_directory(self, output_dir: Optional[str]) -> Optional[str]:
        """Creates a timestamped subdirectory to store run artifacts."""
        if not output_dir:
            return None
            
        # Create base directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a unique, timestamped subdirectory for this specific run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(output_dir, f'run_{timestamp}')
        os.makedirs(run_dir, exist_ok=True)
        
        # Create subdirectories for different output types for better organization
        os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(run_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(run_dir, 'reports'), exist_ok=True)
        
        logging.info(f"Output directory created at: {run_dir}")
        return run_dir

    def _apply_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies polynomial and interaction features to numeric columns."""
        logging.info("🔧 Applying feature engineering (Polynomial & Interaction)...")
        numeric_cols = X.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            logging.warning("No numeric columns found for feature engineering. Skipping...")
            return X
        
        # Make a copy and fill NaNs to prevent errors in PolynomialFeatures
        X_numeric_clean = X[numeric_cols].copy()
        X_numeric_clean.fillna(X_numeric_clean.median(), inplace=True)
            
        # Create interaction features (e.g., feat1 * feat2)
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        X_poly = poly.fit_transform(X_numeric_clean)
        
        poly_feature_names = poly.get_feature_names_out(numeric_cols)
        X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)
        
        # Combine new polynomial features with original non-numeric features
        X_final = X.drop(columns=numeric_cols).join(X_poly_df)
        logging.info(f"✅ Feature engineering complete. New data shape: {X_final.shape}")
        return X_final

    def _show_tuning_insights(self, study: optuna.Study, model_name: str):
        """Displays visualizations of the hyperparameter tuning process."""
        logging.info(f"Displaying tuning visualizations for {model_name.upper()}...")
        try:
            # Plot 1: Optimization history
            fig1 = optuna.visualization.plot_optimization_history(study)
            fig1.update_layout(title=f'Optimization History for {model_name}')
            if self.output_dir:
                fig1.write_image(os.path.join(self.output_dir, 'plots', f'{model_name}_optimization_history.png'))
            fig1.show()

            # Plot 2: Parameter importance
            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.update_layout(title=f'Parameter Importances for {model_name}')
            if self.output_dir:
                fig2.write_image(os.path.join(self.output_dir, 'plots', f'{model_name}_param_importance.png'))
            fig2.show()
        except (ValueError, ImportError) as e:
            logging.warning(f"Could not create tuning visualizations: {e}")

    def _load_single_model(self, name: str) -> Any:
        """Factory method to instantiate a model from the MODEL_CONFIG."""
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
        name = name.lower()
        config = MODEL_CONFIG[self.task].get(name)
        if config is None:
            raise ValueError(f"Model '{name}' is not recognized for task '{self.task}'.")
        
        model_class = config['model']
        
        # Common parameters
        params = {'random_state': self.random_state} if 'random_state' in model_class().get_params() else {}
        
<<<<<<< HEAD
        # Nonaktifkan verbosity untuk model-model tertentu
        if name in ['catboost', 'xgboost', 'lightgbm']:
            params.update({'verbose': 0, 'verbosity': 0} if name != 'lightgbm' else {'verbose': -1})
        if name == 'xgboost':
            params.update({
                'use_label_encoder': False,
                'eval_metric': 'logloss' if self.task == 'classification' else 'rmse'
            })
            
        return model_class(**params)

    def _build_pipeline(self, model: Any, numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
        # Step 1: Preprocessor untuk imputasi, scaling, dan encoding
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )

        # Buat daftar langkah-langkah pipeline
        steps = [('preprocessor', preprocessor)]

        # Step 2 (Opsional): Imbalance Handling
        if self.imbalance_handler and self.task == 'classification':
            if self.imbalance_handler == 'smote':
                handler = SMOTE(random_state=self.random_state)
            elif self.imbalance_handler == 'random_oversampler':
                handler = RandomOverSampler(random_state=self.random_state)
            else:
                raise ValueError(f"Imbalance handler '{self.imbalance_handler}' tidak didukung.")
            steps.append(('resampler', handler))

        # Step 3 (Opsional): Feature Engineering
        if 'polynomial' in self.feature_engineering:
            # Note: PolynomialFeatures harus dijalankan setelah imputasi & scaling
            # Ini adalah contoh sederhana; implementasi yang lebih canggih mungkin diperlukan
            # Untuk sekarang, kita akan membuat pipeline terpisah untuk ini
            # TODO: Integrasikan FE lebih baik ke dalam preprocessor utama
            pass # Implementasi lebih lanjut diperlukan untuk integrasi yang mulus

        # Step 4: Model
        steps.append(('model', model))

        # Pilih tipe pipeline (ImbPipeline menangani resampler dengan benar)
        pipeline_class = ImbPipeline if self.imbalance_handler else Pipeline
        return pipeline_class(steps, memory=self.memory)

    def _calculate_all_metrics(self, y_true, y_pred, y_proba=None) -> Dict:
=======
        # Model-specific parameters to reduce verbosity
        if name == 'catboost':
            params['verbose'] = 0
        if name == 'xgboost':
            params.update({
                'use_label_encoder': False, # Avoid deprecation warning
                'eval_metric': 'logloss' if self.task == 'classification' else 'rmse',
                'verbosity': 0 # Silence XGBoost's own logging
            })
        if name == 'lightgbm':
            params['verbose'] = -1 # Silence LightGBM's logging
            
        return model_class(**params)

    def _calculate_all_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculates and returns a dictionary of relevant metrics for the task."""
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
        if self.task == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0)
            }
<<<<<<< HEAD
            if y_proba is not None and len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            return metrics
=======
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
        else: # Regression
            mse = mean_squared_error(y_true, y_pred)
            return {
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2_score': r2_score(y_true, y_pred)
            }
            
    def _check_imbalance(self, y: pd.Series, threshold: float = 0.4):
        """Mendeteksi jika dataset tidak seimbang."""
        if self.task != 'classification':
            return
        
        counts = y.value_counts(normalize=True)
        if len(counts) > 1 and counts.min() < threshold:
            minority_class = counts.idxmin()
            minority_percentage = counts.min() * 100
            logging.warning(f"Potensi imbalance terdeteksi. Kelas minoritas '{minority_class}' hanya {minority_percentage:.2f}% dari data.")
            if not self.imbalance_handler:
                self.recommendations_.append(
                    "Dataset Anda terlihat tidak seimbang. Pertimbangkan untuk menggunakan parameter `imbalance_handler='smote'` "
                    "untuk meningkatkan performa model pada kelas minoritas."
                )

<<<<<<< HEAD
    def _tune_with_optuna(self, pipeline: Pipeline, model_name: str, X_train, y_train) -> Dict:
        logging.info(f"🔬 Memulai hyperparameter tuning untuk {model_name.upper()}...")
=======
    def _tune_with_optuna(self, model_name: str, X_train, y_train) -> Dict[str, Any]:
        """Performs hyperparameter tuning for a given model using Optuna."""
        logging.info(f"🔬 Starting hyperparameter tuning for {model_name.upper()}...")
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
        
        param_func = MODEL_CONFIG[self.task][model_name.lower()].get('params')
        if not param_func:
            logging.warning(f"No hyperparameter search space defined for '{model_name}'. Using default parameters.")
            return {}

        def objective(trial: optuna.Trial) -> float:
            """The objective function that Optuna will try to maximize."""
            params = param_func(trial)
            pipeline.set_params(**params)
            
<<<<<<< HEAD
            # Tambahkan early stopping untuk model yang mendukung
            fit_params = {}
            if 'lgbm' in model_name and 'model__n_estimators' in params:
                fit_params['model__callbacks'] = [lgb.early_stopping(15, verbose=False)]
            elif 'xgb' in model_name and 'model__n_estimators' in params:
                 fit_params['model__early_stopping_rounds'] = 15
            
            metric = 'f1_weighted' if self.task == 'classification' else 'r2'
            
            # Lakukan cross-validation pada seluruh pipeline
            score = cross_val_score(pipeline, X_train, y_train, n_jobs=-1, cv=self.cv_folds, scoring=metric, fit_params=fit_params).mean()
            return score

        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=self.n_trials, timeout=600)
=======
            # Select cross-validation strategy
            if self.task == 'classification':
                if self.cv_strategy == 'repeated':
                    cv = RepeatedStratifiedKFold(n_splits=self.cv_folds, n_repeats=2, random_state=self.random_state)
                else: # 'stratified'
                    cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            else: # Regression doesn't need stratification
                cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

            scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                # Split data for this fold
                X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Train and evaluate
                model.fit(X_train_fold, y_train_fold)
                preds = model.predict(X_val_fold)
                
                # Use the primary metric for optimization
                metric = 'f1_score' if self.task == 'classification' else 'r2_score'
                scores.append(self._calculate_all_metrics(y_val_fold, preds)[metric])
            
            return np.mean(scores)

        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(direction='maximize', pruner=pruner)
        study.optimize(objective, n_trials=self.n_trials, timeout=600, n_jobs=-1, show_progress_bar=False)
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
        
        if self.show_tuning_plots:
            optuna.visualization.plot_optimization_history(study).show()
            optuna.visualization.plot_param_importances(study).show()
        
        logging.info(f"✅ Tuning complete. Best parameters found: {study.best_params}")
        return study.best_params
<<<<<<< HEAD
    
=======

    def _run_single_model_pipeline(self, model_name: str, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """Executes the full pipeline (tune, train, evaluate) for a single model."""
        logging.info(f"--- Processing model: {model_name.upper()} ---")
        
        best_params = {}
        if self.tune_hyperparameters:
            best_params = self._tune_with_optuna(model_name, X_train, y_train)

        # Instantiate model with best (or default) parameters
        model = self._load_single_model(model_name)
        if best_params:
            model.set_params(**best_params)

        # Train the model and time it
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        logging.info(f"Training finished in {training_time:.2f} seconds.")
        
        # Evaluate the model
        predictions = model.predict(X_test)
        metrics = self._calculate_all_metrics(y_test, predictions)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        return {
            'model_name': model_name,
            'model_object': model,
            'predictions': predictions,
            'prediction_proba': y_pred_proba,
            'actual': y_test,
            'metrics': metrics,
            'training_time_seconds': training_time,
            'best_params': best_params
        }

>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
    def run_pipeline(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        display_report: bool = False
    ) -> Dict[str, Any]:
        """
        Runs the main ML pipeline: data splitting, preprocessing, model training, and evaluation.

        Args:
            df (pd.DataFrame): The input dataframe containing features and target.
            target_column (str): The name of the target variable column.
            test_size (float): The proportion of the dataset to allocate to the test set.
            compare (bool): If True, prints a comparison table of all models.
            explain (bool): If True, creates and displays a metric comparison plot.
            chosen_metric (Optional[str]): The metric to use for plotting if `explain` is True.
            display_report (bool): If True, renders the full HTML report in the output cell.

        Returns:
            Dict[str, Any]: A dictionary containing details of the best model and results for all models.
        """
        # 1. Split data into features (X) and target (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 2. Split data into training and test sets
        # Use stratification for classification to maintain target distribution
        stratify = y if self.task == 'classification' else None
        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify
        )
        
<<<<<<< HEAD
        self._check_imbalance(self.y_train_)

        numeric_features = self.X_train_.select_dtypes(include=np.number).columns.tolist()
        categorical_features = self.X_train_.select_dtypes(exclude=np.number).columns.tolist()
        
=======
        # 3. Apply optional external data cleaner
        if self.data_cleaner and NoventisDataCleaner is not None:
            logging.info("External DataCleaner detected. Fitting and transforming data...")
            self.data_cleaner.fit(X_train, y_train)
            X_train = self.data_cleaner.transform(X_train)
            X_test = self.data_cleaner.transform(X_test)
            logging.info("✅ External DataCleaner processing complete.")
        
        # 4. Set up and run internal preprocessor
        logging.info("Running internal preprocessor to handle data types and missing values...")
        numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

        # Create a pipeline to handle numeric and categorical data separately
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough' # Keep other columns (if any)
        )

        self.preprocessor.fit(X_train)
        X_train_transformed = self.preprocessor.transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)
        logging.info("✅ Internal preprocessing complete. All data is now numeric.")
        
        # 5. Apply optional feature engineering
        if self.enable_feature_engineering:
            # Note: This step is complex. For now, it is disabled for stability.
            # To re-enable, you would need to handle feature names carefully after transformation.
            logging.warning("Feature engineering is currently disabled for stability.")
        
        # Store the final, processed data for later use (e.g., in explainability)
        self.X_train_final, self.X_test_final, self.y_test_final = X_train_transformed, X_test_transformed, y_test

        # 6. Run the training and evaluation pipeline for each specified model
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
        model_list = self.model_name if isinstance(self.model_name, list) else [self.model_name]
        self.all_results_ = []
        
        parent_run_id = None
        if self.use_mlflow:
             # Mulai parent run jika menjalankan lebih dari satu model
            if len(model_list) > 1:
                parent_run = mlflow.start_run(run_name=f"Pipeline Run - {int(time.time())}", nested=False)
                parent_run_id = parent_run.info.run_id
                mlflow.log_param("model_list", model_list)
                mlflow.log_param("task", self.task)

        for name in model_list:
            run_name = f"{name} - {int(time.time())}"
            try:
                with mlflow.start_run(run_name=run_name, nested=True) if self.use_mlflow else self._null_context_manager():
                    if self.use_mlflow:
                        mlflow.log_param("model_name", name)
                        mlflow.log_param("imbalance_handler", self.imbalance_handler)

                    logging.info(f"--- Memproses model: {name.upper()} ---")
                    base_model = self._get_base_model(name)
                    pipeline = self._build_pipeline(base_model, numeric_features, categorical_features)

                    best_params = {}
                    if self.tune_hyperparameters:
                        best_params = self._tune_with_optuna(pipeline, name, self.X_train_, self.y_train_)
                        pipeline.set_params(**best_params)
                        if self.use_mlflow:
                            mlflow.log_params({k: v for k, v in best_params.items()})

                    start_time = time.time()
                    pipeline.fit(self.X_train_, self.y_train_)
                    training_time = time.time() - start_time
                    
                    predictions = pipeline.predict(self.X_test_)
                    y_proba = pipeline.predict_proba(self.X_test_) if hasattr(pipeline, 'predict_proba') else None
                    
                    metrics = self._calculate_all_metrics(self.y_test_, predictions, y_proba)
                    if self.use_mlflow:
                        mlflow.log_metrics(metrics)
                        mlflow.sklearn.log_model(pipeline, "model")

                    self.all_results_.append({
                        'model_name': name,
                        'pipeline_object': pipeline,
                        'predictions': predictions,
                        'prediction_proba': y_proba,
                        'actual': self.y_test_,
                        'metrics': metrics,
                        'training_time_seconds': training_time,
                        'best_params': best_params
                    })
                    logging.info(f"✅ Selesai: {name.upper()} | Metrik utama: {list(metrics.values())[3 if self.task=='classification' else -1]:.4f}")

            except Exception as e:
<<<<<<< HEAD
                logging.error(f"Gagal memproses model {name}: {e}")
                self.all_results_.append({'model_name': name, 'metrics': {}, 'error': str(e)})

        if parent_run_id and self.use_mlflow:
            mlflow.end_run()

        successful_results = [res for res in self.all_results_ if 'error' not in res]
        if not successful_results:
            raise RuntimeError("Tidak ada model yang berhasil dilatih.")
=======
                logging.error(f"Failed to process model {name}: {e}")
                self.all_results.append({'model_name': name, 'metrics': {}, 'error': str(e)})

        # 7. Identify the best model from successful runs
        successful_results = [res for res in self.all_results if 'error' not in res]
        if not successful_results:
            raise RuntimeError("No models were trained successfully. Please check your data or configuration.")
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2

        primary_metric = 'f1_score' if self.task == 'classification' else 'r2_score'
        self.best_model_info_ = max(successful_results, key=lambda x: x['metrics'].get(primary_metric, -1))
        self.is_fitted_ = True
        
<<<<<<< HEAD
        logging.info(f"\n--- Proses Selesai ---")
        logging.info(f"🏆 Model Terbaik: {self.best_model_info_['model_name'].upper()} dengan {primary_metric} = {self.best_model_info_['metrics'][primary_metric]:.4f}")

=======
        logging.info(f"\n--- Process Complete ---")
        logging.info(f"🏆 Best Model: {self.best_model_info['model_name'].upper()} with {primary_metric} = {self.best_model_info['metrics'][primary_metric]:.4f}")

        # 8. Display results as requested
        if compare:
            self._print_comparison()
        if explain:
            self._create_metric_plot(chosen_metric)
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
        if display_report:
            self.display_report()

        return {
            'best_model_details': self.best_model_info_,
            'all_model_results': self.all_results_
        }
    
    # --- Metode Publik Baru & Tambahan ---
    
    def optimize_threshold(self, metric: str = 'f1_score', steps: int = 100) -> Tuple[float, float]:
        if not self.is_fitted_ or self.task != 'classification':
            raise RuntimeError("Metode ini hanya untuk task klasifikasi setelah .run_pipeline() dijalankan.")
        
        pipeline = self.best_model_info_['pipeline_object']
        y_proba = pipeline.predict_proba(self.X_test_)[:, 1]
        
        best_threshold, best_score = 0, -1
        
        if metric == 'f1_score':
            precisions, recalls, thresholds = precision_recall_curve(self.y_test_, y_proba)
            f1_scores = (2 * precisions * recalls) / (precisions + recalls)
            f1_scores = f1_scores[~np.isnan(f1_scores)]
            best_score = np.max(f1_scores)
            best_threshold = thresholds[np.argmax(f1_scores)]
        else:
            # Implementasi untuk metrik lain jika perlu
            raise ValueError(f"Metrik '{metric}' belum didukung untuk optimisasi threshold.")
            
        logging.info(f"Threshold optimal ditemukan: {best_threshold:.4f} (Meningkatkan {metric} menjadi {best_score:.4f})")
        return best_threshold, best_score

    def stack_models(self, model_names: Optional[List[str]] = None, top_n: int = 3, final_estimator_name: str = 'logistic_regression'):
        if not self.is_fitted_:
            raise RuntimeError("Jalankan .run_pipeline() terlebih dahulu.")
        
        if model_names:
            estimators_info = [res for res in self.all_results_ if res['model_name'] in model_names]
        else:
            estimators_info = sorted(self.all_results_, key=lambda x: x['metrics'].get('f1_score' if self.task=='classification' else 'r2_score', -1), reverse=True)[:top_n]
        
        estimators = [(info['model_name'], info['pipeline_object']) for info in estimators_info]
        final_estimator = self._get_base_model(final_estimator_name)

        logging.info(f"Membangun Stacking model dengan: {[e[0] for e in estimators]}")
        
        if self.task == 'classification':
            stacking_model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=self.cv_folds)
        else:
            stacking_model = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=self.cv_folds)
            
        logging.info("Melatih Stacking model...")
        stacking_model.fit(self.X_train_, self.y_train_)
        
        # Evaluasi
        preds = stacking_model.predict(self.X_test_)
        proba = stacking_model.predict_proba(self.X_test_) if hasattr(stacking_model, 'predict_proba') else None
        metrics = self.calculate_all_metrics(self.y_test_, preds, proba)
        
        logging.info(f"✅ Stacking Selesai. Metrik: {metrics}")
        return stacking_model, metrics

    # --- Metode Penyimpanan, Pemuatan & Helper ---

    def save_model(self, filepath: str):
        if not self.is_fitted_:
            raise ValueError("Tidak ada model terbaik untuk disimpan. Jalankan .run_pipeline() dahulu.")
        
        pipeline_to_save = self.best_model_info_.get('pipeline_object')
        logging.info(f"Menyimpan pipeline lengkap '{self.best_model_info_['model_name']}' ke {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_to_save, f)
        logging.info("✅ Pipeline berhasil disimpan.")

    @staticmethod
    def load_model(filepath: str) -> Pipeline:
        logging.info(f"Memuat pipeline dari {filepath}...")
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
        logging.info("✅ Pipeline berhasil dimuat.")
        return pipeline

    def explain_model(self, pipeline_object=None, plot_type='summary'):
        if not self.is_fitted_:
            raise RuntimeError("Jalankan .run_pipeline() terlebih dahulu.")
        
        pipeline_to_explain = pipeline_object or self.best_model_info_.get('pipeline_object')
        model = pipeline_to_explain.named_steps['model']
        preprocessor = pipeline_to_explain.named_steps['preprocessor']

        logging.info(f"Membuat SHAP Explainer untuk model di dalam pipeline...")
        
        # Transformasi data test untuk explainer
        X_test_transformed = preprocessor.transform(self.X_test_)
        
        # SHAP memerlukan DataFrame untuk nama fitur yang bagus
        try:
            feature_names = preprocessor.get_feature_names_out()
            X_test_transformed_df = pd.DataFrame(X_test_transformed.toarray(), columns=feature_names)
        except:
            X_test_transformed_df = pd.DataFrame(X_test_transformed)
        
        explainer = shap.Explainer(model, X_test_transformed_df)
        shap_values = explainer(X_test_transformed_df)
        
        plt.figure()
        if plot_type == 'summary':
            shap.summary_plot(shap_values, X_test_transformed_df, show=False)
        elif plot_type == 'beeswarm':
             shap.plots.beeswarm(shap_values, show=False)
        else:
             logging.warning(f"Plot tipe '{plot_type}' belum didukung.")

        plt.tight_layout()
        plt.show()

    # --- Metode Laporan HTML (Sudah Dioptimalkan) ---

    def generate_html_report(self, filepath: Optional[str] = None) -> str:
        # ... (Kode HTML template dan CSS tetap sama) ...
        # Perubahan utama ada pada fungsi-fungsi yang dipanggil di bawah ini
        
        summary_html = self._get_summary_html()
        recommendations_html = self._get_recommendations_html()
        comparison_table_html = self._get_comparison_table_html()
        plots_html = self._get_plots_html()
        diagnostics_html = self._get_diagnostics_html()
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="id">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Laporan Manual Predictor</title>
            <style>
                /* ... CSS LENGKAP ANDA DI SINI ... */
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>Laporan Analisis Manual Predictor</h1>
                    <p>Hasil detail dari pipeline pelatihan dan evaluasi model.</p>
                </header>
                <nav class="navbar">
                    <button class="nav-btn active" onclick="showTab(event, 'summary')">Ringkasan</button>
                    <button class="nav-btn" onclick="showTab(event, 'recommendations')">Rekomendasi</button>
                    <button class="nav-btn" onclick="showTab(event, 'comparison')">Perbandingan</button>
                    <button class="nav-btn" onclick="showTab(event, 'plots')">Visualisasi</button>
                    <button class="nav-btn" onclick="showTab(event, 'diagnostics')">Diagnostik</button>
                </nav>
                <main>
                    <section id="summary" class="content-section" style="display: block;">
                        <h2>Ringkasan Eksekusi</h2>
                        {summary_html}
                    </section>
                    <section id="recommendations" class="content-section">
                        <h2>💡 Rekomendasi Otomatis</h2>
                        {recommendations_html}
                    </section>
                    <section id="comparison" class="content-section">
                        <h2>Detail Perbandingan Metrik</h2>
                        {comparison_table_html}
                    </section>
                    <section id="plots" class="content-section">
                        <h2>Visualisasi Hasil</h2>
                        {plots_html}
                    </section>
                    <section id="diagnostics" class="content-section">
                        <h2>Diagnostik Model Terbaik</h2>
                        {diagnostics_html}
                    </section>
                </main>
            </div>
            <script>
                // ... (JavaScript Anda dari kode sebelumnya diletakkan di sini) ...
            </script>
        </body>
        </html>
        """
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f: f.write(html_template)
            logging.info(f"✅ Laporan HTML berhasil disimpan di {filepath}")

        return html_template
    
    def _get_summary_html(self) -> str:
        # Sedikit dimodifikasi untuk menampilkan parameter baru
        best_model_info = self.best_model_info_
        primary_metric = 'f1_score' if self.task == 'classification' else 'r2_score'
        
        return f"""
        <div class="grid-container">
            <div class="grid-item">
                <h4>Konfigurasi Pipeline</h4>
                <p><strong>Tipe Task:</strong> {self.task.title()}</p>
                <p><strong>Model Diuji:</strong> {str([res['model_name'] for res in self.all_results_ if 'error' not in res])}</p>
                <p><strong>Imbalance Handler:</strong> {self.imbalance_handler or 'Tidak Aktif'}</p>
                <p><strong>Feature Engineering:</strong> {str(self.feature_engineering) if self.feature_engineering else 'Tidak Aktif'}</p>
            </div>
            <div class="grid-item score-card">
                <h4>🏆 Model Terbaik</h4>
                <p class="model-name">{best_model_info['model_name'].upper()}</p>
                <p class="metric-score">{primary_metric.replace('_', ' ').title()}: {best_model_info['metrics'].get(primary_metric, 0):.4f}</p>
            </div>
            <div class="grid-item">
                <h4>Detail Eksekusi</h4>
                <p><strong>Tuning Hyperparameters:</strong> {'Aktif' if self.tune_hyperparameters else 'Nonaktif'}</p>
                <p><strong>Waktu Pelatihan Terbaik:</strong> {best_model_info.get('training_time_seconds', 0):.2f} detik</p>
                <p><strong>MLflow Logging:</strong> {'Aktif' if self.use_mlflow else 'Nonaktif'}</p>
            </div>
        </div>
        """
        
    def _get_recommendations_html(self) -> str:
        if not self.recommendations_:
            return "<p>Tidak ada rekomendasi spesifik untuk proses ini. Hasil terlihat bagus!</p>"
        
        html = "<ul>"
        for rec in self.recommendations_:
            html += f"<li class='recommendation-item'>{rec}</li>"
        html += "</ul>"
        return html
        
    def _get_diagnostics_html(self) -> str:
        """Membuat plot Learning Curve untuk model terbaik."""
        try:
            pipeline = self.best_model_info_['pipeline_object']
            model_name = self.best_model_info_['model_name'].upper()
            
            train_sizes, train_scores, test_scores = learning_curve(
                pipeline, self.X_train_, self.y_train_, cv=self.cv_folds, n_jobs=-1, 
                train_sizes=np.linspace(.1, 1.0, 5)
            )

            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.grid()
            ax.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                            np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1, color="r")
            ax.fill_between(train_sizes, np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                            np.mean(test_scores, axis=1) + np.std(test_scores, axis=1), alpha=0.1, color="g")
            ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
            ax.set_title(f'Learning Curve untuk {model_name}')
            ax.set_xlabel("Training examples")
            ax.set_ylabel("Score")
            ax.legend(loc="best")
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return f'<div class="plot-container"><img src="data:image/png;base64,{img_base64}" alt="Learning Curve Plot"></div>'

        except Exception as e:
            return f"<p>Gagal membuat plot diagnostik: {e}</p>"
            
    # Metode lainnya (_get_comparison_table_html, _get_plots_html, display_report, etc.)
    # sebagian besar tetap sama, hanya perlu memastikan mereka menggunakan `self.all_results_`
    # dan `self.best_model_info_`
    def get_results_dataframe(self) -> pd.DataFrame:
<<<<<<< HEAD
        if not self.all_results_:
            logging.warning("Tidak ada hasil. Jalankan .run_pipeline() dahulu.")
=======
        """Converts the list of model results into a sorted Pandas DataFrame."""
        if not self.all_results:
            logging.warning("No results available. Please run the pipeline first using .run_pipeline().")
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
            return pd.DataFrame()
        
        # Create a list of records, one for each successful model run
        records = [
            {'model': res['model_name'], **res['metrics']}
            for res in self.all_results_ if 'error' not in res
        ]
        df = pd.DataFrame(records).set_index('model')
        
        # Sort by the primary metric in descending order
        primary_metric = 'f1_score' if self.task == 'classification' else 'r2_score'
        is_higher_better = primary_metric in ['f1_score', 'r2_score', 'accuracy', 'precision', 'recall', 'roc_auc']
        return df.sort_values(by=primary_metric, ascending=not is_higher_better)

<<<<<<< HEAD
=======
    def _save_plot(self, fig: plt.Figure, filename: str):
        """Helper method to save a matplotlib figure to the output directory."""
        if self.output_dir:
            plot_path = os.path.join(self.output_dir, 'plots', filename)
            fig.savefig(plot_path, bbox_inches='tight', dpi=300)
            logging.info(f"Plot saved to: {plot_path}")

    def save_model(self, filepath: Optional[str] = None):
        """Saves the best trained model object to a pickle file."""
        if not self.best_model_info:
            raise ValueError("No best model available to save. Run the pipeline first.")
        
        model_to_save = self.best_model_info.get('model_object')
        model_name = self.best_model_info['model_name']
        
        # Determine the save path
        if filepath:
            save_path = filepath
        elif self.output_dir:
            save_path = os.path.join(self.output_dir, 'models', f'{model_name}_best_model.pkl')
        else:
            raise ValueError("Please provide a 'filepath' or set an 'output_dir' during initialization.")
            
        logging.info(f"Saving model '{model_name}' to {save_path}...")
        with open(save_path, 'wb') as f:
            pickle.dump(model_to_save, f)
        logging.info("✅ Model saved successfully.")

    @staticmethod
    def load_model(filepath: str) -> Any:
        """Loads a model from a pickle file."""
        logging.info(f"Loading model from {filepath}...")
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logging.info("✅ Model loaded successfully.")
        return model
        
    def _print_comparison(self):
        """Prints a formatted comparison table of all model results to the console."""
        if not self.all_results:
            logging.warning("No results to compare.")
            return
        
        print("\n" + "="*80 + "\n📊 MODEL COMPARISON - ALL METRICS\n" + "="*80)
        print(self.get_results_dataframe())
        print("="*80)

    def _create_metric_plot(self, chosen_metric: Optional[str] = None):
        """Creates and displays a bar chart comparing model performance."""
        if not self.all_results:
            logging.warning("No results to plot.")
            return
        
        # Use the primary metric if none is specified
        metric = chosen_metric or ('f1_score' if self.task == 'classification' else 'r2_score')
        
        df_results = self.get_results_dataframe().reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(df_results['model'], df_results[metric], color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_title(f'Model Comparison - {metric.upper()}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add labels on top of each bar
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + (df_results[metric].max() * 0.01), f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Highlight the best performing model's bar
        is_higher_better = metric in ['accuracy', 'precision', 'recall', 'f1_score', 'r2_score']
        best_val = df_results[metric].max() if is_higher_better else df_results[metric].min()
        best_bar_index = df_results[df_results[metric] == best_val].index[0]
        bars[best_bar_index].set_color('gold')
        bars[best_bar_index].set_edgecolor('darkorange')
        bars[best_bar_index].set_linewidth(2)
        
        # Save the plot if an output directory is configured
        if self.output_dir:
            self._save_plot(fig, f'metric_comparison_{metric}.png')
        
        plt.tight_layout()
        plt.show()

    def explain_model(self, model_object: Optional[Any] = None, plot_type: str = 'summary', feature: Optional[str] = None):
        """
        Generates and displays SHAP plots to explain model predictions.

        Args:
            model_object (Optional[Any]): The model to explain. If None, the best model is used.
            plot_type (str): The type of SHAP plot to generate ('summary', 'beeswarm', 'dependence').
            feature (Optional[str]): The feature name, required for 'dependence' plots.
        """
        if self.X_test_final is None:
            raise RuntimeError("The pipeline must be run before explaining the model.")
        
        model_to_explain = model_object or self.best_model_info.get('model_object')
        if not model_to_explain:
            raise ValueError("No model available to explain.")

        model_name = self.best_model_info.get('model_name', 'Model').upper()
        logging.info(f"Creating SHAP Explainer for {model_name}...")

        try:
            # Convert sparse matrix to dense DataFrame if needed and get feature names
            X_test_df = pd.DataFrame(self.X_test_final.toarray() if hasattr(self.X_test_final, 'toarray') else self.X_test_final)
            if self.preprocessor:
                try:
                    feature_names = self.preprocessor.get_feature_names_out()
                    X_test_df.columns = feature_names
                except Exception:
                    logging.warning("Could not retrieve feature names from preprocessor. Plots may use generic names.")
            
            # Initialize explainer and calculate SHAP values
            explainer = shap.Explainer(model_to_explain, self.X_train_final)
            shap_values = explainer(X_test_df)
        except Exception as e:
            logging.error(f"Failed to create SHAP explainer: {e}")
            return
        
        logging.info(f"Generating SHAP '{plot_type}' plot...")
        plt.figure() # Create a new figure for the plot
        title = f"SHAP {plot_type.title()} Plot for {model_name}"
        
        # Generate the requested plot type
        if plot_type == 'summary':
            shap.summary_plot(shap_values, X_test_df, show=False)
        elif plot_type == 'beeswarm':
            shap.plots.beeswarm(shap_values, show=False)
        elif plot_type == 'dependence':
            if not feature:
                raise ValueError("A 'feature' argument must be provided for dependence plots.")
            shap.dependence_plot(feature, shap_values.values, X_test_df, interaction_index=None, show=False)
        else:
            logging.warning(f"Plot type '{plot_type}' is not supported. Supported types: 'summary', 'beeswarm', 'dependence'.")
        
        plt.title(title)
        plt.tight_layout()
        plt.show()

    # =======================================================================
    #                 NEW & MODIFIED REPORTING METHODS
    # =======================================================================

    def _get_summary_html(self) -> str:
        """Generates the HTML for the summary section of the report."""
        if not self.best_model_info:
            return "<p>No results available to display.</p>"
            
        primary_metric = 'f1_score' if self.task == 'classification' else 'r2_score'
        best_score = self.best_model_info['metrics'].get(primary_metric, 0)
        training_time = self.best_model_info.get('training_time_seconds', 0)
        best_params_str = str(self.best_model_info.get('best_params', 'Default'))
        
        # HTML for the summary grid
        return f"""
        <div class="grid-container">
            <div class="grid-item">
                <h4>Process Summary</h4>
                <p><strong>Task Type:</strong> {self.task.title()}</p>
                <p><strong>Models Tested:</strong> {len([res for res in self.all_results if 'error' not in res])}</p>
                <p><strong>Cross-Validation Strategy:</strong> {self.cv_strategy.title()}</p>
                <p><strong>Hyperparameter Tuning:</strong> {'Enabled' if self.tune_hyperparameters else 'Disabled'}</p>
            </div>
            <div class="grid-item score-card">
                <h4>🏆 Best Model</h4>
                <p class="model-name">{self.best_model_info['model_name'].upper()}</p>
                <p class="metric-score">{primary_metric.replace('_', ' ').title()}: {best_score:.4f}</p>
            </div>
            <div class="grid-item">
                <h4>Best Model Details</h4>
                <p><strong>Training Time:</strong> {training_time:.2f} seconds</p>
                <p><strong>Best Parameters:</strong></p>
                <pre class="params-box">{best_params_str}</pre>
            </div>
        </div>
        """

>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
    def _get_comparison_table_html(self) -> str:
        """Generates the HTML for the model comparison table."""
        df_results = self.get_results_dataframe()
        if df_results.empty:
            return "<p>No model comparison data available.</p>"
        # Convert DataFrame to a styled HTML table
        return df_results.to_html(classes='styled-table', border=0, float_format='{:.4f}'.format)

<<<<<<< HEAD
    def _get_plots_html(self) -> str:
        # ... (Kode ini hampir tidak berubah, hanya memastikan variabelnya benar) ...
        # Contohnya:
        try:
            # ... (kode plot perbandingan metrik dari kode sebelumnya) ...
            # ... (kode plot SHAP dari kode sebelumnya) ...
            return "..."
        except Exception as e:
            return f"<p>Gagal membuat plot: {e}</p>"

    def display_report(self):
        logging.info("Mempersiapkan laporan untuk ditampilkan di output...")
        html_content = self.generate_html_report() 
        display(HTML(html_content))
        logging.info("✅ Laporan berhasil ditampilkan.")

    def _null_context_manager(self):
        # Helper untuk blok 'with' saat mlflow dinonaktifkan
        from contextlib import contextmanager
        @contextmanager
        def null_manager():
            yield None
        return null_manager()
=======
    def _create_metric_plot_for_html(self) -> plt.Figure:
        """Creates the metric comparison plot and returns the figure object."""
        metric = 'f1_score' if self.task == 'classification' else 'r2_score'
        df_results = self.get_results_dataframe().reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(df_results['model'], df_results[metric], color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_title(f'Model Comparison - {metric.upper()}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add labels to bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + (df_results[metric].max()*0.01), f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Highlight the best bar
        best_val = df_results[metric].max()
        best_bar_index = df_results[df_results[metric] == best_val].index[0]
        bars[best_bar_index].set_color('gold')
        bars[best_bar_index].set_edgecolor('darkorange')
        bars[best_bar_index].set_linewidth(2)
        
        plt.tight_layout()
        return fig

    def _get_plots_html(self) -> str:
        """Generates HTML for the plots, embedding them as Base64 strings."""
        if self.X_test_final is None:
            return "<p>Plots cannot be generated because the pipeline has not been run.</p>"
        
        plots_html = ""
        # Plot 1: Metric Comparison
        try:
            fig_metric = self._create_metric_plot_for_html()
            buf = io.BytesIO()
            fig_metric.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plots_html += f'<h4>Model Performance Comparison</h4><div class="plot-container"><img src="data:image/png;base64,{img_base64}" alt="Metric Comparison Plot"></div>'
            plt.close(fig_metric)
        except Exception as e:
            plots_html += f"<p>Failed to create metric comparison plot: {e}</p>"
        
        # Plot 2: SHAP Summary
        try:
            model_to_explain = self.best_model_info.get('model_object')
            model_name = self.best_model_info.get('model_name', 'Model').upper()
            
            X_test_df = pd.DataFrame(self.X_test_final.toarray() if hasattr(self.X_test_final, 'toarray') else self.X_test_final)
            if self.preprocessor:
                try:
                    feature_names = self.preprocessor.get_feature_names_out()
                    X_test_df.columns = feature_names
                except Exception:
                    pass

            explainer = shap.Explainer(model_to_explain, self.X_train_final)
            shap_values = explainer(X_test_df)
            
            shap.summary_plot(shap_values, X_test_df, show=False)
            fig_shap = plt.gcf() # Get the current figure generated by SHAP
            fig_shap.suptitle(f"Feature Importance (SHAP) for {model_name}", fontsize=16)
            
            # Convert SHAP plot to Base64
            buf_shap = io.BytesIO()
            fig_shap.savefig(buf_shap, format='png', bbox_inches='tight')
            buf_shap.seek(0)
            img_base64_shap = base64.b64encode(buf_shap.read()).decode('utf-8')
            plots_html += f'<br><h4>Feature Importance (SHAP)</h4><div class="plot-container"><img src="data:image/png;base64,{img_base64_shap}" alt="SHAP Summary Plot"></div>'
            plt.close(fig_shap)
        except Exception as e:
            plots_html += f"<p>Failed to create SHAP plot: {e}</p>"
            
        return plots_html

    def generate_html_report(self, filepath: Optional[str] = None) -> str:
        """
        Assembles and saves a complete, interactive HTML report.

        Args:
            filepath (Optional[str]): The path to save the HTML file. If None, uses the output_dir.

        Returns:
            str: The full HTML content as a string.
        """
        if not self.best_model_info:
            msg = "Report cannot be generated. Please run the pipeline first using .run_pipeline()."
            logging.error(msg)
            return f"<p>{msg}</p>"

        # Generate each part of the report
        summary_html = self._get_summary_html()
        comparison_table_html = self._get_comparison_table_html()
        plots_html = self._get_plots_html()
        
        # Full HTML template with embedded CSS and JavaScript
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Manual Predictor Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Exo+2:wght@700&display=swap');
        :root {{
            --bg-dark-1: #0D1117; --bg-dark-2: #161B22; --border-color: #30363D;
            --primary-blue: #58A6FF; --primary-orange: #F78166;
            --text-light: #C9D1D9; --text-medium: #8B949E;
        }}
        body {{
            font-family: 'Roboto', sans-serif; background-color: var(--bg-dark-1);
            color: var(--text-light); margin: 0; padding: 20px;
        }}
        .container {{
            max-width: 1200px; margin: auto; background-color: var(--bg-dark-2);
            border: 1px solid var(--border-color); border-radius: 8px; overflow: hidden;
        }}
        header {{
            padding: 30px; background-color: var(--bg-dark-1); text-align: center;
            border-bottom: 1px solid var(--border-color);
        }}
        header h1 {{ font-family: 'Exo 2', sans-serif; color: var(--primary-blue); margin: 0; }}
        .navbar {{ display: flex; background-color: var(--bg-dark-2); border-bottom: 1px solid var(--border-color); }}
        .nav-btn {{
            background: none; border: none; color: var(--text-medium);
            padding: 15px 25px; cursor: pointer; font-size: 16px;
            border-bottom: 3px solid transparent;
        }}
        .nav-btn:hover {{ color: var(--text-light); }}
        .nav-btn.active {{ color: var(--primary-orange); border-bottom-color: var(--primary-orange); }}
        .content-section {{ padding: 30px; display: none; }}
        .grid-container {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px; margin-bottom: 30px;
        }}
        .grid-item {{
            background-color: var(--bg-dark-1); padding: 20px;
            border-radius: 6px; border: 1px solid var(--border-color);
        }}
        .score-card {{
            text-align: center; background: linear-gradient(145deg, #1A2D40, #101820);
        }}
        .score-card .model-name {{
            font-family: 'Exo 2'; font-size: 2.5em;
            color: var(--primary-orange); margin: 10px 0;
        }}
        .score-card .metric-score {{ font-size: 1.5em; color: var(--text-light); margin: 0; }}
        .params-box {{
            background-color: #010409; padding: 10px; border-radius: 4px;
            font-family: monospace; font-size: 0.9em;
            white-space: pre-wrap; word-wrap: break-word;
        }}
        .styled-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        .styled-table th, .styled-table td {{
            padding: 12px 15px; text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        .styled-table thead th {{ background-color: #1A2D40; color: var(--primary-blue); }}
        .styled-table tbody tr:hover {{ background-color: #222b38; }}
        .plot-container {{
            background-color: var(--bg-dark-1); padding: 15px; border-radius: 6px;
            border: 1px solid var(--border-color); margin-top: 20px;
        }}
        .plot-container img {{ max-width: 100%; height: auto; display: block; margin: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Manual Predictor Analysis Report</h1>
            <p>Detailed results from the model training and evaluation pipeline.</p>
        </header>
        <nav class="navbar">
            <button class="nav-btn active" onclick="showTab(event, 'summary')">Summary</button>
            <button class="nav-btn" onclick="showTab(event, 'comparison')">Model Comparison</button>
            <button class="nav-btn" onclick="showTab(event, 'plots')">Visualizations</button>
        </nav>
        <main>
            <section id="summary" class="content-section" style="display: block;">
                <h2>Execution Summary</h2>
                {summary_html}
            </section>
            <section id="comparison" class="content-section">
                <h2>Detailed Metric Comparison</h2>
                {comparison_table_html}
            </section>
            <section id="plots" class="content-section">
                <h2>Result Visualizations</h2>
                {plots_html}
            </section>
        </main>
    </div>
    <script>
        function showTab(event, tabName) {{
            let i, sections, navbuttons;
            sections = document.getElementsByClassName("content-section");
            for (i = 0; i < sections.length; i++) {{
                sections[i].style.display = "none";
            }}
            navbuttons = document.getElementsByClassName("nav-btn");
            for (i = 0; i < navbuttons.length; i++) {{
                navbuttons[i].className = navbuttons[i].className.replace(" active", "");
            }}
            document.getElementById(tabName).style.display = "block";
            event.currentTarget.className += " active";
        }}
        // Ensure the first tab is active on page load
        document.addEventListener("DOMContentLoaded", function() {{
            if (document.querySelector('.nav-btn')) {{
               document.querySelector('.nav-btn').click();
            }}
        }});
    </script>
</body>
</html>
        """
        # Determine save path
        if filepath:
            save_path = filepath
        elif self.output_dir:
            save_path = os.path.join(self.output_dir, 'reports', 'analysis_report.html')
        else:
            # If no path is given, just return the HTML string
            return html_template

        # Save the report to a file
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_template)
            logging.info(f"✅ HTML report successfully saved to: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save HTML report: {e}")

        return html_template

    def display_report(self):
        """Renders the HTML report directly in a Jupyter/Colab output cell."""
        logging.info("Preparing report for display in output cell...")
        try:
            # Generate the report content without saving it to a file
            html_content = self.generate_html_report() 
            # Use IPython's display function to render the HTML
            display(HTML(html_content))
            logging.info("✅ Report displayed successfully.")
        except NameError:
             logging.warning("Cannot display report. Ensure you are running this in a Jupyter/Colab environment and that 'display' and 'HTML' from IPython.display are imported.")
        except Exception as e:
            logging.error(f"Failed to display report: {e}")
>>>>>>> 1c86ed69e078df3f9b492c60f35cc4ce875314c2
