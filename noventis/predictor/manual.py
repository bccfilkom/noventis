import pandas as pd
import numpy as np
import pickle
import time
import logging
import warnings
import os
from datetime import datetime
from typing import Dict, Any, List, Union, Optional

import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

from noventis import data_cleaner 
from noventis.data_cleaner import NoventisDataCleaner

warnings.filterwarnings('ignore')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import optuna
import shap
optuna.logging.set_verbosity(optuna.logging.ERROR)

from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold, 
    train_test_split, KFold
)
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from scipy import stats

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

try:
    from ..data_cleaner import NoventisDataCleaner
except ImportError:
    NoventisDataCleaner = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_rf_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Define Random Forest hyperparameter search space."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
    }


def get_xgb_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Define XGBoost hyperparameter search space."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
    }


def get_lgbm_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Define LightGBM hyperparameter search space."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'is_unbalance': trial.suggest_categorical('is_unbalance', [True, False]),
    }


def get_dt_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Define Decision Tree hyperparameter search space."""
    return {
        'max_depth': trial.suggest_int('max_depth', 3, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
    }


def get_gb_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Define Gradient Boosting hyperparameter search space."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
    }


def get_catboost_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Define CatBoost hyperparameter search space."""
    return {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
    }


def get_rf_reg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Define Random Forest Regression hyperparameter search space."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
    }


def get_xgb_reg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Define XGBoost Regression hyperparameter search space."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }


def get_lgbm_reg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Define LightGBM Regression hyperparameter search space."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
    }


# Model configuration registry
MODEL_CONFIG = {
    'classification': {
        'logistic_regression': {'model': LogisticRegression, 'params': None},
        'decision_tree': {'model': DecisionTreeClassifier, 'params': get_dt_params},
        'random_forest': {'model': RandomForestClassifier, 'params': get_rf_params},
        'gradient_boosting': {'model': GradientBoostingClassifier, 'params': get_gb_params},
        'xgboost': {'model': xgb.XGBClassifier, 'params': get_xgb_params},
        'lightgbm': {'model': lgb.LGBMClassifier, 'params': get_lgbm_params},
        'catboost': {'model': cb.CatBoostClassifier, 'params': get_catboost_params}
    },
    'regression': {
        'linear_regression': {'model': LinearRegression, 'params': None},
        'decision_tree': {'model': DecisionTreeRegressor, 'params': get_dt_params},
        'random_forest': {'model': RandomForestRegressor, 'params': get_rf_reg_params},
        'gradient_boosting': {'model': GradientBoostingRegressor, 'params': get_gb_params},
        'xgboost': {'model': xgb.XGBRegressor, 'params': get_xgb_reg_params},
        'lightgbm': {'model': lgb.LGBMRegressor, 'params': get_lgbm_reg_params},
        'catboost': {'model': cb.CatBoostRegressor, 'params': get_catboost_params}
    }
}



class NoventisManualML:
    """
    Manual machine learning predictor with hyperparameter tuning using Optuna.
    
    Provides fine-grained control over model selection, hyperparameter optimization,
    cross-validation strategies, and comprehensive result visualization.
    
    Key Features:
    - 7 classifiers and 7 regressors support
    - Optuna-based hyperparameter tuning with cross-validation
    - SHAP-based model interpretability
    - Professional HTML reports with dark theme
    - Comprehensive visualization suite
    - Flexible cross-validation strategies
    
    Attributes:
        task_type (str): Task type ('classification' or 'regression')
        model_name (Union[str, List[str]]): Models to train
        results (Dict): All model results storage
        best_model_info (Dict): Best performing model information
    """
    
    # Class constants
    DEFAULT_CLASSIFICATION_METRIC = 'f1_score'
    DEFAULT_REGRESSION_METRIC = 'r2_score'
    
    def __init__(
        self,
        model_name: Union[str, List[str]],
        task: str,
        random_state: int = 42,
        data_cleaner_object: Optional[Any] = None,
        data_cleaner_bool: bool = False,
        tune_hyperparameters: bool = False,
        n_trials: int = 50,
        cv_folds: int = 3,
        enable_feature_engineering: bool = False,
        cv_strategy: str = 'repeated',
        show_tuning_plots: bool = False,
        output_dir: Optional[str] = None
    ) -> None:
        """
        Initialize NoventisManualML instance.
        
        Args:
            model_name (Union[str, List[str]]): 
                Name(s) of model(s) to train. 
                Supported classification models: 'logistic_regression', 'decision_tree', 
                'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'catboost'
                Supported regression models: 'linear_regression', 'decision_tree', 
                'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'catboost'
                Default: None
            
            task (str):
                Type of machine learning task.
                Must be one of: 'classification', 'regression'
                Default: None
            
            random_state (int):
                Random seed for reproducibility.
                Controls randomness in train-test split, cross-validation, and models.
                Default: 42
            
            data_cleaner_object (Optional[Any]):
                External DataCleaner object (e.g., NoventisDataCleaner instance).
                If provided, will be used for data preprocessing before model training.
                Must have either fit_transform() or fit()/transform() methods.
                Default: None
            
            data_cleaner_bool (bool):
                Flag to use built-in NoventisDataCleaner from noventis package.
                Only used if data_cleaner_object is not provided.
                Default: False
            
            tune_hyperparameters (bool):
                Whether to perform Bayesian hyperparameter optimization using Optuna.
                If True, will search for optimal hyperparameters for each model.
                Default: False
            
            n_trials (int):
                Number of trials for Optuna hyperparameter search.
                Higher values may find better parameters but take longer.
                Only used if tune_hyperparameters=True.
                Default: 50
            
            cv_folds (int):
                Number of cross-validation folds.
                Used for both hyperparameter tuning and model evaluation.
                Must be >= 2.
                Default: 3
            
            enable_feature_engineering (bool):
                Whether to enable automated feature engineering (polynomial features, etc).
                Currently disabled for stability reasons.
                Default: False
            
            cv_strategy (str):
                Cross-validation strategy for classification tasks.
                Options: 'repeated' (RepeatedStratifiedKFold), 'standard' (StratifiedKFold)
                Default: 'repeated'
            
            show_tuning_plots (bool):
                Whether to display Optuna optimization visualizations during tuning.
                Includes optimization history and parameter importance plots.
                Default: False
            
            output_dir (Optional[str]):
                Base directory for saving outputs (plots, models, reports).
                If provided, creates timestamped subdirectories.
                Default: None
        
        Raises:
            ValueError: If task is not 'classification' or 'regression'
        
        Returns:
            None
        
        Note:
            The predictor will be in an untrained state after initialization.
            Call fit() method to train models.
        """
        self._report_id = f"report-{id(self)}"
        self.model_name = model_name
        self.task_type = task.lower()
        self.random_state = random_state
        
        self.data_cleaner_object = data_cleaner_object
        self.data_cleaner_bool = data_cleaner_bool
        
        self.tune_hyperparameters = tune_hyperparameters
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.enable_feature_engineering = enable_feature_engineering
        self.cv_strategy = cv_strategy
        self.show_tuning_plots = show_tuning_plots
        self.output_dir = self._setup_output_directory(output_dir)

        self.results = {}
        self.best_model_info: Dict[str, Any] = {}
        self.all_results: List[Dict[str, Any]] = []
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor: Optional[ColumnTransformer] = None
        
        self.data_cleaner_used = False
        self.cleaner_instance = None

        if self.task_type not in ['classification', 'regression']:
            raise ValueError(
                f"Task must be 'classification' or 'regression', got '{task}'"
            )
        
        logging.info(f"NoventisManualML initialized for {self.task_type}")

    def _setup_output_directory(self, output_dir: Optional[str]) -> Optional[str]:
        """
        Create organized directory structure for outputs.
        
        Args:
            output_dir (Optional[str]):
                Base output directory path. If None, no directory structure is created.
                Default: None
        
        Returns:
            Optional[str]:
                Path to created run directory containing subdirectories:
                - plots/: Visualization images
                - models/: Saved model files
                - reports/: HTML and text reports
                Returns None if output_dir is not specified.
        
        Raises:
            OSError: If directory creation fails due to permission issues
        
        Example:
            >>> output_dir = ml._setup_output_directory('/tmp/ml_output')
            >>> # Creates: /tmp/ml_output/run_20240115_143022/
            >>> #          ├── plots/
            >>> #          ├── models/
            >>> #          └── reports/
        """
        if not output_dir:
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(output_dir, f'run_{timestamp}')
        
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(run_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(run_dir, 'reports'), exist_ok=True)
        
        logging.info(f"Output directory created at: {run_dir}")
        return run_dir

    def _load_single_model(self, name: str) -> Any:
        """Initialize a model with default configuration."""
        name = name.lower()
        config = MODEL_CONFIG[self.task_type].get(name)
        
        if config is None:
            available = list(MODEL_CONFIG[self.task_type].keys())
            raise ValueError(
                f"Model '{name}' not recognized for task '{self.task_type}'. "
                f"Available models: {available}"
            )
        
        model_class = config['model']
        params = {}
        
        if 'random_state' in model_class().get_params():
            params['random_state'] = self.random_state
        
        if name == 'catboost':
            params['verbose'] = 0
        elif name == 'xgboost':
            params.update({
                'use_label_encoder': False,
                'eval_metric': 'logloss' if self.task_type == 'classification' else 'rmse',
                'verbosity': 0
            })
        elif name == 'lightgbm':
            params['verbose'] = -1
        
        return model_class(**params)

    def _eval_classification(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with accuracy, precision, recall, and F1 score
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0)
        }

    def _eval_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive regression metrics."""
        mse = mean_squared_error(y_true, y_pred)
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2_score': r2_score(y_true, y_pred)
        }

    def _tune_with_optuna(
        self, 
        model_name: str, 
        X_train: np.ndarray, 
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """Perform Bayesian hyperparameter optimization using Optuna."""
        logging.info(f"Starting hyperparameter tuning for {model_name.upper()}...")
        
        param_func = MODEL_CONFIG[self.task_type][model_name.lower()].get('params')
        if not param_func:
            logging.warning(
                f"No hyperparameter search space defined for '{model_name}'. "
                "Using default parameters."
            )
            return {}

        def objective(trial: optuna.Trial) -> float:
            params = param_func(trial)
            model = self._load_single_model(model_name)
            model.set_params(**params)
            
            if self.task_type == 'classification':
                if self.cv_strategy == 'repeated':
                    cv = RepeatedStratifiedKFold(
                        n_splits=self.cv_folds, 
                        n_repeats=2, 
                        random_state=self.random_state
                    )
                else:
                    cv = StratifiedKFold(
                        n_splits=self.cv_folds, 
                        shuffle=True, 
                        random_state=self.random_state
                    )
            else:
                cv = KFold(
                    n_splits=self.cv_folds, 
                    shuffle=True, 
                    random_state=self.random_state
                )

            scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_train_fold = X_train[train_idx]
                X_val_fold = X_train[val_idx]
                y_train_fold = y_train.iloc[train_idx]
                y_val_fold = y_train.iloc[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                preds = model.predict(X_val_fold)
                
                if self.task_type == 'classification':
                    metrics = self._eval_classification(y_val_fold, preds)
                    metric = self.DEFAULT_CLASSIFICATION_METRIC
                else:
                    metrics = self._eval_regression(y_val_fold, preds)
                    metric = self.DEFAULT_REGRESSION_METRIC
                
                scores.append(metrics[metric])
            
            return np.mean(scores)

        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(direction='maximize', pruner=pruner)
        study.optimize(
            objective, 
            n_trials=self.n_trials, 
            timeout=600, 
            n_jobs=-1, 
            show_progress_bar=False
        )
        
        if self.show_tuning_plots:
            self._show_tuning_insights(study, model_name)
        
        logging.info(f"Tuning complete. Best parameters: {study.best_params}")
        return study.best_params

    def _show_tuning_insights(self, study: optuna.Study, model_name: str) -> None:
        """Display Optuna optimization visualizations."""
        logging.info(f"Displaying tuning visualizations for {model_name.upper()}...")
        
        try:
            fig1 = optuna.visualization.plot_optimization_history(study)
            fig1.update_layout(title=f'Optimization History for {model_name}')
            if self.output_dir:
                save_path = os.path.join(
                    self.output_dir, 'plots', 
                    f'{model_name}_optimization_history.png'
                )
                fig1.write_image(save_path)
            fig1.show()

            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.update_layout(title=f'Parameter Importances for {model_name}')
            if self.output_dir:
                save_path = os.path.join(
                    self.output_dir, 'plots', 
                    f'{model_name}_param_importance.png'
                )
                fig2.write_image(save_path)
            fig2.show()
            
        except (ValueError, ImportError) as e:
            logging.warning(f"Could not create tuning visualizations: {e}")

    def _run_single_model_pipeline(
        self, 
        model_name: str, 
        X_train: np.ndarray, 
        y_train: pd.Series, 
        X_test: np.ndarray, 
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Complete pipeline for training and evaluating a single model."""
        logging.info(f"Processing model: {model_name.upper()}")
        
        best_params = {}
        if self.tune_hyperparameters:
            best_params = self._tune_with_optuna(model_name, X_train, y_train)

        model = self._load_single_model(model_name)
        if best_params:
            model.set_params(**best_params)

        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        logging.info(f"Training finished in {training_time:.2f} seconds.")
        
        predictions = model.predict(X_test)
        y_pred_proba = (
            model.predict_proba(X_test) 
            if hasattr(model, 'predict_proba') 
            else None
        )
        
        if self.task_type == 'classification':
            metrics = self._eval_classification(y_test, predictions)
        else:
            metrics = self._eval_regression(y_test, predictions)
        
        return {
            'model_name': model_name,
            'model': model,
            'predictions': predictions,
            'prediction_proba': y_pred_proba,
            'actual': y_test,
            'metrics': metrics,
            'task_type': self.task_type,
            'training_time_seconds': training_time,
            'best_params': best_params,
            'best_config': best_params,
            'feature_importance': None,
            'best_estimator': model_name
        }

    def fit(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        compare: bool = False,
        explain: bool = False,
        chosen_metric: Optional[str] = None,
        display_report: bool = True,
        use_data_cleaner: bool = False,
        data_cleaner_object: Optional[Any] = None,
        auto_save_model: bool = True  
    ) -> Dict[str, Any]:
        """
        Train machine learning models with optional preprocessing and hyperparameter tuning.
        
        This is the main entry point for the ML pipeline. It handles:
        1. Data splitting (with stratification for classification)
        2. Optional data cleaning (via DataCleaner or built-in)
        3. Internal preprocessing (imputation and encoding)
        4. Hyperparameter tuning (if enabled)
        5. Model training and evaluation
        6. Result reporting
        
        Args:
            df (pd.DataFrame):
                Input DataFrame containing features and target column.
                Must be a valid pandas DataFrame with at least 2 columns.
                All data types supported by sklearn preprocessing are accepted.
                Default: None (required)
            
            target_column (str):
                Name of the target column in df.
                Must exist in df.columns.
                For classification: can be any type (will be label encoded if needed)
                For regression: should be numeric
                Default: None (required)
            
            test_size (float):
                Proportion of data to use for testing.
                Must be between 0.0 and 1.0.
                Training data size = 1 - test_size
                Default: 0.2 (20% test, 80% train)
            
            compare (bool):
                Whether to print formatted comparison table of all models.
                Displays metrics for each trained model.
                Default: False
            
            explain (bool):
                Whether to create and display metric comparison plot.
                Shows bar chart with model performance comparison.
                Default: False
            
            chosen_metric (Optional[str]):
                Specific metric to plot when explain=True.
                If None, uses primary metric (f1_score for classification, r2_score for regression).
                Must be a metric key from results.
                Default: None
            
            display_report (bool):
                Whether to display HTML report in notebook output.
                Only works in Jupyter/Colab environments.
                Default: True
            
            use_data_cleaner (bool):
                Whether to use built-in NoventisDataCleaner from noventis package.
                Ignored if data_cleaner_object is provided.
                Default: False
            
            data_cleaner_object (Optional[Any]):
                External DataCleaner instance to use for preprocessing.
                Takes precedence over use_data_cleaner flag.
                Must have fit_transform() or fit()/transform() methods.
                Default: None
        
        Returns:
            Dict[str, Any]:
                Dictionary containing:
                {
                    'best_model_details': Dict with best model info (model, metrics, etc),
                    'all_model_results': List of results for all trained models
                }
                
                Each model result includes:
                {
                    'model_name': str,
                    'model': trained model object,
                    'predictions': np.ndarray,
                    'actual': pd.Series,
                    'metrics': Dict[str, float],
                    'training_time_seconds': float,
                    'best_params': Dict (if tuning was enabled),
                    'task_type': str
                }
        
        Raises:
            TypeError: If df is not a pandas DataFrame
            ValueError: If target_column not in df.columns
            ValueError: If test_size not in range (0, 1)
            RuntimeError: If no models were trained successfully
            ImportError: If use_data_cleaner=True but noventis package not installed
        
        Example:
            >>> results = ml.fit(
            ...     df=df_titanic,
            ...     target_column='survived',
            ...     test_size=0.2,
            ...     compare=True,
            ...     use_data_cleaner=True,
            ...     display_report=True
            ... )
            >>> print(results['best_model_details']['model_name'])
            'xgboost'
        
        Note:
            - Data is automatically stratified by target for classification tasks
            - Models are evaluated using cross-validation during hyperparameter tuning
            - Best model is selected based on primary metric
            - All results are stored in self.all_results and self.results
        """
        
        logging.info("Starting NoventisManualML Training Pipeline")
        
        # Validate inputs
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be pandas DataFrame, got {type(df)}")
        
        if target_column not in df.columns:
            raise ValueError(f"target_column '{target_column}' not found in DataFrame")
        
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        cleaner_to_use = data_cleaner_object or self.data_cleaner_object
        use_builtin = use_data_cleaner or self.data_cleaner_bool
        
        self.data_cleaner_used = False
        self.cleaner_instance = None
        
        stratify = y if self.task_type == 'classification' else None
        
        # SCENARIO 1: External DataCleaner object provided
        if cleaner_to_use is not None:
            logging.info("External DataCleaner object detected. Applying transformations...")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=self.random_state, 
                stratify=stratify
            )
            
            try:
                if hasattr(cleaner_to_use, 'fit_transform'):
                    X_train = cleaner_to_use.fit_transform(X_train, y_train)
                    X_test = cleaner_to_use.transform(X_test)
                elif hasattr(cleaner_to_use, 'fit') and hasattr(cleaner_to_use, 'transform'):
                    cleaner_to_use.fit(X_train, y_train)
                    X_train = cleaner_to_use.transform(X_train)
                    X_test = cleaner_to_use.transform(X_test)
                else:
                    raise AttributeError("Cleaner must have fit_transform or fit/transform methods")
                
                self.data_cleaner_used = True
                self.cleaner_instance = cleaner_to_use
                logging.info("External DataCleaner processing complete.")
                
            except Exception as e:
                logging.error(f"External DataCleaner failed: {e}")
                logging.info("Falling back to internal preprocessing only.")
                self.data_cleaner_used = False
        
        # SCENARIO 2: Built-in NoventisDataCleaner requested
        elif use_builtin:
            logging.info("Creating built-in NoventisDataCleaner...")
            
            try:
                try:
                    from noventis.data_cleaner import data_cleaner as dc_func
                except ImportError:
                    from noventis import data_cleaner as dc_func
                
                df_with_target = X.copy()
                df_with_target[target_column] = y
                
                df_cleaned, cleaner_instance = dc_func(
                    data=df_with_target,
                    return_instance=True,
                    target_column=target_column,
                    verbose=False
                )
                
                X_cleaned = cleaner_instance._processed_df.drop(columns=[target_column], errors='ignore')
                y_cleaned = df_cleaned[target_column] if target_column in df_cleaned.columns else y.loc[X_cleaned.index]
                
                if len(X_cleaned) != len(y_cleaned):
                    common_idx = X_cleaned.index.intersection(y_cleaned.index)
                    X_cleaned = X_cleaned.loc[common_idx]
                    y_cleaned = y_cleaned.loc[common_idx]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_cleaned, y_cleaned, 
                    test_size=test_size, 
                    random_state=self.random_state, 
                    stratify=y_cleaned if self.task_type == 'classification' else None
                )
                
                self.data_cleaner_used = True
                self.cleaner_instance = cleaner_instance
                logging.info("Built-in DataCleaner processing complete.")
                
            except Exception as e:
                logging.error(f"Built-in DataCleaner failed: {e}")
                logging.info("Falling back to internal preprocessing only.")
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=test_size, 
                    random_state=self.random_state, 
                    stratify=stratify
                )
        
        # SCENARIO 3: No external cleaning
        else:
            logging.info("No external DataCleaner provided. Using standard preprocessing.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=self.random_state, 
                stratify=stratify
            )
        
        logging.info(f"Data split complete: Train={len(X_train)}, Test={len(X_test)}")
        
        logging.info("Applying internal preprocessing (imputation + encoding)...")
        
        numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()
        
        logging.info(f"Numeric features: {len(numeric_features)}, Categorical features: {len(categorical_features)}")
        
        transformers = []
        
        if numeric_features:
            transformers.append(
                ('num', SimpleImputer(strategy='median'), numeric_features)
            )
        
        if categorical_features:
            transformers.append(
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            )
        
        if not transformers:
            logging.warning("No features to transform. Using data as-is.")
            X_train_transformed = X_train.values
            X_test_transformed = X_test.values
        else:
            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough'
            )
            
            X_train_transformed = self.preprocessor.fit_transform(X_train)
            X_test_transformed = self.preprocessor.transform(X_test)
            
            logging.info(f"Preprocessing complete. Final shape: {X_train_transformed.shape}")
        
        if self.enable_feature_engineering:
            logging.warning("Feature engineering is currently disabled for stability reasons.")
        
        self.X_train = X_train_transformed
        self.X_test = X_test_transformed
        self.y_train = y_train
        self.y_test = y_test

        logging.info("Starting model training phase...")
        
        model_list = (
            self.model_name 
            if isinstance(self.model_name, list) 
            else [self.model_name]
        )
        
        self.all_results = []
        
        for idx, name in enumerate(model_list, 1):
            logging.info(f"Training model {idx}/{len(model_list)}: {name.upper()}")
            
            try:
                result = self._run_single_model_pipeline(
                    name, 
                    self.X_train, 
                    y_train, 
                    self.X_test, 
                    y_test
                )
                self.all_results.append(result)
                self.results[name] = result
                
                metrics_str = ", ".join([
                    f"{k}={v:.4f}" 
                    for k, v in result['metrics'].items()
                ])
                logging.info(f"✓ {name.upper()} trained successfully: {metrics_str}")
                
            except Exception as e:
                logging.error(f"✗ Failed to train {name.upper()}: {str(e)}")
                self.all_results.append({
                    'model_name': name, 
                    'metrics': {}, 
                    'error': str(e)
                })
        
        successful_results = [
            res for res in self.all_results 
            if 'error' not in res and res.get('metrics')
        ]
        
        if not successful_results:
            error_details = [
                f"{res['model_name']}: {res.get('error', 'Unknown error')}"
                for res in self.all_results if 'error' in res
            ]
            raise RuntimeError(
                f"No models were trained successfully. Errors:\n" + 
                "\n".join(error_details)
            )
        
        primary_metric = (
            self.DEFAULT_CLASSIFICATION_METRIC 
            if self.task_type == 'classification' 
            else self.DEFAULT_REGRESSION_METRIC
        )
        
        self.best_model_info = max(
            successful_results, 
            key=lambda x: x['metrics'].get(primary_metric, -float('inf'))
        )
        
        best_score = self.best_model_info['metrics'][primary_metric]
        best_name = self.best_model_info['model_name']
        
        logging.info("TRAINING COMPLETE!")
        logging.info(f"Best Model: {best_name.upper()}")
        logging.info(f"Best {primary_metric}: {best_score:.4f}")
        logging.info(f"Successfully trained: {len(successful_results)}/{len(model_list)} models")
        
        if auto_save_model and self.output_dir:
            try:
                self.save_model()
                logging.info(f"✓ Best model auto-saved successfully")
            except Exception as e:
                logging.warning(f"Could not auto-save model: {e}")
        
        if compare:
            self._print_comparison()
        
        if explain:
            self._create_metric_plot(chosen_metric)
        
        if display_report:
            try:
                self.display_report()
            except Exception as e:
                logging.warning(f"Could not display report: {e}")
        
        return {
            'best_model_details': self.best_model_info,
            'all_model_results': self.all_results
        }
    def explain_model(
        self, 
        model_object: Optional[Any] = None, 
        plot_type: str = 'summary', 
        feature: Optional[str] = None
    ) -> None:
        """Generate SHAP-based model explanations."""
        if self.X_test is None:
            raise RuntimeError("The pipeline must be run before explaining the model.")
        
        model_to_explain = model_object or self.best_model_info.get('model')
        if not model_to_explain:
            raise ValueError("No model available to explain.")

        model_name = self.best_model_info.get('model_name', 'Model').upper()
        logging.info(f"Creating SHAP Explainer for {model_name}...")

        try:
            X_test_df = pd.DataFrame(
                self.X_test.toarray() 
                if hasattr(self.X_test, 'toarray') 
                else self.X_test
            )
            
            if self.preprocessor:
                try:
                    feature_names = self.preprocessor.get_feature_names_out()
                    X_test_df.columns = feature_names
                except Exception:
                    logging.warning(
                        "Could not retrieve feature names from preprocessor. "
                        "Plots may use generic names."
                    )
            
            explainer = shap.Explainer(model_to_explain, self.X_train)
            shap_values = explainer(X_test_df)
            
        except Exception as e:
            logging.error(f"Failed to create SHAP explainer: {e}")
            return
        
        logging.info(f"Generating SHAP '{plot_type}' plot...")
        plt.figure()
        title = f"SHAP {plot_type.title()} Plot for {model_name}"
        
        if plot_type == 'summary':
            shap.summary_plot(shap_values, X_test_df, show=False)
        elif plot_type == 'beeswarm':
            shap.plots.beeswarm(shap_values, show=False)
        elif plot_type == 'dependence':
            if not feature:
                raise ValueError(
                    "A 'feature' argument must be provided for dependence plots."
                )
            shap.dependence_plot(
                feature, shap_values.values, X_test_df, 
                interaction_index=None, show=False
            )
        else:
            logging.warning(
                f"Plot type '{plot_type}' is not supported. "
                "Supported types: 'summary', 'beeswarm', 'dependence'."
            )
            return
        
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string for HTML embedding."""
        buf = io.BytesIO()
        fig.savefig(
            buf, 
            format='png', 
            bbox_inches='tight', 
            dpi=150, 
            facecolor='#0D1117'
        )
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return f"data:image/png;base64,{img_str}"

    def _set_plot_style(self) -> None:
        """Configure dark theme matplotlib style."""
        sns.set_style("darkgrid")
        plt.rcParams.update({
            'figure.facecolor': '#0D1117',
            'axes.facecolor': '#161B22',
            'text.color': '#C9D1D9',
            'axes.labelcolor': '#C9D1D9',
            'xtick.color': '#8B949E',
            'ytick.color': '#8B949E',
            'grid.color': '#30363D'
        })

    def _create_classification_plots(self) -> Dict[str, str]:
        """Generate comprehensive classification analysis plots."""
        plots = {}
        self._set_plot_style()
        
        best_result = self.best_model_info
        y_true = best_result['actual']
        y_pred = best_result['predictions']
        y_pred_proba = best_result.get('prediction_proba')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.patch.set_facecolor('#0D1117')
        
        # Panel 1: Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            cbar_kws={'label': 'Count'},
            linewidths=1, linecolor='#30363D'
        )
        axes[0, 0].set_title(
            'Confusion Matrix',
            fontsize=14, fontweight='bold',
            color='#58A6FF', pad=15
        )
        axes[0, 0].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('True Label', fontsize=11, fontweight='bold')
        
        # Panel 2: ROC Curve
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            axes[0, 1].plot(
                fpr, tpr,
                color='#F78166', lw=2.5,
                label=f'ROC Curve (AUC = {roc_auc:.3f})'
            )
            axes[0, 1].plot(
                [0, 1], [0, 1],
                color='#8B949E', lw=2,
                linestyle='--', label='Random Classifier'
            )
            axes[0, 1].set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
            axes[0, 1].set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
            axes[0, 1].set_title(
                'ROC Curve',
                fontsize=14, fontweight='bold',
                color='#58A6FF', pad=15
            )
            axes[0, 1].legend(
                loc='lower right', frameon=True,
                facecolor='#161B22', edgecolor='#30363D'
            )
            axes[0, 1].grid(True, alpha=0.3, linestyle='--', color='#30363D')
        else:
            axes[0, 1].text(
                0.5, 0.5, 'ROC Curve\nNot Available',
                ha='center', va='center',
                fontsize=13, color='#8B949E',
                transform=axes[0, 1].transAxes
            )
            axes[0, 1].set_xticks([])
            axes[0, 1].set_yticks([])
        
        # Panel 3: Precision-Recall Curve
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
            
            axes[1, 0].plot(
                recall, precision,
                color='#58A6FF', lw=2.5, label='PR Curve'
            )
            axes[1, 0].set_xlabel('Recall', fontsize=11, fontweight='bold')
            axes[1, 0].set_ylabel('Precision', fontsize=11, fontweight='bold')
            axes[1, 0].set_title(
                'Precision-Recall Curve',
                fontsize=14, fontweight='bold',
                color='#58A6FF', pad=15
            )
            axes[1, 0].legend(
                loc='best', frameon=True,
                facecolor='#161B22', edgecolor='#30363D'
            )
            axes[1, 0].grid(True, alpha=0.3, linestyle='--', color='#30363D')
        else:
            axes[1, 0].text(
                0.5, 0.5, 'Precision-Recall\nNot Available',
                ha='center', va='center',
                fontsize=13, color='#8B949E',
                transform=axes[1, 0].transAxes
            )
            axes[1, 0].set_xticks([])
            axes[1, 0].set_yticks([])
        
        # Panel 4: Model Comparison
        metrics_data = self.get_results_dataframe().reset_index()
        x_pos = np.arange(len(metrics_data))
        
        colors = [
            '#F78166' if row['model'] == best_result['model_name'] else '#58A6FF'
            for _, row in metrics_data.iterrows()
        ]
        
        bars = axes[1, 1].bar(
            x_pos, metrics_data['f1_score'],
            color=colors, alpha=0.85,
            edgecolor='#30363D', linewidth=1.5
        )
        
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(
            metrics_data['model'],
            rotation=45, ha='right', fontsize=10
        )
        axes[1, 1].set_ylabel('F1 Score', fontsize=11, fontweight='bold')
        axes[1, 1].set_title(
            'Model Performance Comparison',
            fontsize=14, fontweight='bold',
            color='#58A6FF', pad=15
        )
        axes[1, 1].grid(True, axis='y', alpha=0.3, linestyle='--', color='#30363D')
        
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9,
                fontweight='bold', color='#C9D1D9'
            )
        
        plt.tight_layout(pad=3)
        plots['classification_analysis'] = self._fig_to_base64(fig)
        
        return plots

    def _create_regression_plots(self) -> Dict[str, str]:
        """Generate comprehensive regression analysis plots."""
        plots = {}
        self._set_plot_style()
        
        best_result = self.best_model_info
        y_true = best_result['actual']
        y_pred = best_result['predictions']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.patch.set_facecolor('#0D1117')
        
        # Panel 1: Predicted vs Actual
        axes[0, 0].scatter(
            y_true, y_pred,
            alpha=0.6, s=50,
            color='#58A6FF',
            edgecolors='#30363D',
            linewidth=0.5
        )
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 0].plot(
            [min_val, max_val], [min_val, max_val],
            'r--', lw=2.5,
            label='Perfect Prediction',
            color='#F78166'
        )
        
        axes[0, 0].set_xlabel('Actual Values', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
        axes[0, 0].set_title(
            'Predicted vs Actual Values',
            fontsize=14, fontweight='bold',
            color='#58A6FF', pad=15
        )
        axes[0, 0].legend(frameon=True, facecolor='#161B22', edgecolor='#30363D')
        axes[0, 0].grid(True, alpha=0.3, linestyle='--', color='#30363D')
        
        # Panel 2: Residuals Plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(
            y_pred, residuals,
            alpha=0.6, s=50,
            color='#F78166',
            edgecolors='#30363D',
            linewidth=0.5
        )
        axes[0, 1].axhline(
            y=0,
            color='#58A6FF',
            linestyle='--', lw=2.5,
            label='Zero Residual'
        )
        
        axes[0, 1].set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Residuals', fontsize=11, fontweight='bold')
        axes[0, 1].set_title(
            'Residual Plot',
            fontsize=14, fontweight='bold',
            color='#58A6FF', pad=15
        )
        axes[0, 1].legend(frameon=True, facecolor='#161B22', edgecolor='#30363D')
        axes[0, 1].grid(True, alpha=0.3, linestyle='--', color='#30363D')
        
        # Panel 3: Residuals Distribution
        axes[1, 0].hist(
            residuals, bins=30,
            color='#58A6FF', alpha=0.75,
            edgecolor='#30363D', linewidth=1.2
        )
        axes[1, 0].axvline(
            x=0,
            color='#F78166',
            linestyle='--', lw=2.5,
            label='Zero Residual'
        )
        
        axes[1, 0].set_xlabel('Residuals', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1, 0].set_title(
            'Distribution of Residuals',
            fontsize=14, fontweight='bold',
            color='#58A6FF', pad=15
        )
        axes[1, 0].legend(frameon=True, facecolor='#161B22', edgecolor='#30363D')
        axes[1, 0].grid(True, axis='y', alpha=0.3, linestyle='--', color='#30363D')
        
        # Panel 4: Model Comparison
        metrics_data = self.get_results_dataframe().reset_index()
        x_pos = np.arange(len(metrics_data))
        
        colors = [
            '#F78166' if row['model'] == best_result['model_name'] else '#58A6FF'
            for _, row in metrics_data.iterrows()
        ]
        
        bars = axes[1, 1].bar(
            x_pos, metrics_data['r2_score'],
            color=colors, alpha=0.85,
            edgecolor='#30363D', linewidth=1.5
        )
        
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(
            metrics_data['model'],
            rotation=45, ha='right', fontsize=10
        )
        axes[1, 1].set_ylabel('R² Score', fontsize=11, fontweight='bold')
        axes[1, 1].set_title(
            'Model Performance Comparison',
            fontsize=14, fontweight='bold',
            color='#58A6FF', pad=15
        )
        axes[1, 1].grid(True, axis='y', alpha=0.3, linestyle='--', color='#30363D')
        
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9,
                fontweight='bold', color='#C9D1D9'
            )
        
        plt.tight_layout(pad=3)
        plots['regression_analysis'] = self._fig_to_base64(fig)
        
        return plots

    def _create_feature_importance_plot(self) -> str:
        """Create horizontal bar plot of top 20 feature importances."""
        try:
            model = self.best_model_info['model']
            
            if not hasattr(model, 'feature_importances_'):
                return ""
            
            importances = model.feature_importances_
            
            try:
                feature_names = self.preprocessor.get_feature_names_out()
            except:
                feature_names = [f'Feature_{i}' for i in range(len(importances))]
            
            indices = np.argsort(importances)[::-1][:20]
            top_features = [feature_names[i] for i in indices]
            top_importances = importances[indices]
            
            self._set_plot_style()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor('#0D1117')
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_importances)))
            bars = ax.barh(
                range(len(top_importances)), top_importances,
                color=colors,
                edgecolor='#30363D', linewidth=1.2, alpha=0.85
            )
            
            ax.set_yticks(range(len(top_importances)))
            ax.set_yticklabels(top_features, fontsize=10)
            ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
            ax.set_title(
                'Top 20 Feature Importances',
                fontsize=14, fontweight='bold',
                color='#58A6FF', pad=15
            )
            ax.invert_yaxis()
            ax.grid(True, axis='x', alpha=0.3, linestyle='--', color='#30363D')
            
            for i, (bar, val) in enumerate(zip(bars, top_importances)):
                ax.text(
                    val, i, f' {val:.4f}',
                    va='center', fontsize=9,
                    color='#C9D1D9', fontweight='bold'
                )
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logging.warning(f"Could not create feature importance plot: {e}")
            return ""

    def _get_summary_html(self) -> str:
        """Generate HTML for summary section of report."""
        if not self.best_model_info:
            return "<p>No results available to display.</p>"
        
        primary_metric = (
            self.DEFAULT_CLASSIFICATION_METRIC 
            if self.task_type == 'classification' 
            else self.DEFAULT_REGRESSION_METRIC
        )
        best_score = self.best_model_info['metrics'].get(primary_metric, 0)
        training_time = self.best_model_info.get('training_time_seconds', 0)
        best_params_str = str(self.best_model_info.get('best_params', 'Default'))
        
        color_map = {
            'mae': '#FFB86C',
            'mse': '#FF79C6',
            'rmse': '#8BE9FD',
            'r2': '#50FA7B'
        }
        all_metrics_html = ""

        for metric_name, metric_value in self.best_model_info['metrics'].items():
            color = color_map.get(metric_name.lower(), '#FFB86C')
            all_metrics_html += f"""
                <div class='metric-item' style="border-color:{color};">
                    <span class='metric-label' style="color:{color};">
                        {metric_name.replace('_', ' ').title()}
                    </span>
                    <span class='metric-value' style="color:{color};">
                        {metric_value:.4f}
                    </span>
                </div>
            """

        cleaner_status = "Yes ✓" if self.data_cleaner_used else "No"
        cleaner_color = "#3FB950" if self.data_cleaner_used else "#8B949E"
        
        cleaner_info = ""
        if self.data_cleaner_used and self.cleaner_instance:
            cleaner_type = type(self.cleaner_instance).__name__
            try:
                quality_score = self.cleaner_instance.quality_score_.get('final_score', 'N/A')
                cleaner_info = f"""
                    <p><strong>Cleaner Type:</strong> {cleaner_type}</p>
                    <p><strong>Cleaner Quality Score:</strong> {quality_score}</p>
                """
            except:
                cleaner_info = f"<p><strong>Cleaner Type:</strong> {cleaner_type}</p>"
        elif self.data_cleaner_used and not self.cleaner_instance:
            cleaner_info = "<p><strong>Cleaner Type:</strong> External DataCleaner (type unknown)</p>"
        
        return f"""
        <div class="grid-container">
            <div class="grid-item">
                <h4>Process Summary</h4>
                <p><strong>Task Type:</strong> {self.task_type.title()}</p>
                <p><strong>Models Tested:</strong> {len([res for res in self.all_results if 'error' not in res])}</p>
                <p><strong>Cross-Validation Strategy:</strong> {self.cv_strategy.title()}</p>
                <p><strong>CV Folds:</strong> {self.cv_folds}</p>
                <p><strong>Hyperparameter Tuning:</strong> {'Enabled' if self.tune_hyperparameters else 'Disabled'}</p>
                {f"<p><strong>Tuning Trials:</strong> {self.n_trials}</p>" if self.tune_hyperparameters else ""}
                <p><strong>Data Cleaner Used:</strong> <span style="color: {cleaner_color}; font-weight: bold; font-size: 1.1em;">{cleaner_status}</span></p>
                {cleaner_info}
            </div>
            <div class="grid-item score-card">
                <h4>Best Model</h4>
                <p class="model-name">{self.best_model_info['model_name'].upper()}</p>
                <p class="metric-score">{primary_metric.replace('_', ' ').title()}: {best_score:.4f}</p>
                <p class="training-time">⏱ Training Time: {training_time:.2f}s</p>
            </div>
            <div class="grid-item">
                <h4>All Metrics for Best Model</h4>
                <div class="metrics-grid">
                    {all_metrics_html}
                </div>
            </div>
            <div class="grid-item">
                <h4>Best Hyperparameters</h4>
                <pre class="params-box">{best_params_str}</pre>
            </div>
            <div class="grid-item">
                <h4>Dataset Information</h4>
                <p><strong>Training Samples:</strong> {self.X_train.shape[0]:,}</p>
                <p><strong>Test Samples:</strong> {self.X_test.shape[0]:,}</p>
                <p><strong>Number of Features:</strong> {self.X_train.shape[1]}</p>
                <p><strong>Test Split Size:</strong> {(self.X_test.shape[0] / (self.X_train.shape[0] + self.X_test.shape[0]) * 100):.1f}%</p>
                <p style="margin-top: 15px;"><strong>Target Distribution (Train):</strong></p>
                <div class="scrollable-box">
                    <pre>{str(self.y_train.value_counts().to_dict())}</pre>
                </div>
            </div>
            <div class="grid-item">
                <h4>Data Processing Pipeline</h4>
                <p><strong>Preprocessing Method:</strong> {self._get_preprocessor_info()}</p>
                <p><strong>Data Cleaner:</strong> <span style="color: {cleaner_color}; font-weight: bold;">{cleaner_status}</span></p>
                <p><strong>Internal Preprocessing:</strong> Enabled (Imputation + Encoding)</p>
                <p><strong>Feature Engineering:</strong> {'Enabled' if self.enable_feature_engineering else 'Disabled'}</p>
            </div>
        </div>
        """

    def _get_comparison_table_html(self) -> str:
        """Generate HTML comparison table of all trained models."""
        df_results = self.get_results_dataframe()
        if df_results.empty:
            return "<p>No model comparison data available.</p>"
        
        styled_html = df_results.style.format('{:.4f}').set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#1A2D40'),
                ('color', '#58A6FF'),
                ('font-weight', 'bold'),
                ('padding', '12px'),
                ('text-align', 'left')
            ]},
            {'selector': 'td', 'props': [
                ('padding', '10px'),
                ('text-align', 'left')
            ]},
            {'selector': 'tr:hover', 'props': [
                ('background-color', '#222b38')
            ]},
        ]).to_html()
        
        return f'<div class="table-container">{styled_html}</div>'

    def _get_plots_html(self) -> str:
        """Generate HTML section with all visualizations."""
        if self.X_test is None:
            return "<p>Plots cannot be generated because the pipeline has not been run.</p>"
        
        plots_html = ""
        
        if self.task_type == 'classification':
            plots = self._create_classification_plots()
            if 'classification_analysis' in plots:
                plots_html += f"""
                    <h4>Classification Analysis</h4>
                    <div class="plot-container">
                        <img src="{plots['classification_analysis']}" alt="Classification Analysis">
                    </div>
                """
        else:
            plots = self._create_regression_plots()
            if 'regression_analysis' in plots:
                plots_html += f"""
                    <h4>Regression Analysis</h4>
                    <div class="plot-container">
                        <img src="{plots['regression_analysis']}" alt="Regression Analysis">
                    </div>
                """
        
        feature_importance_plot = self._create_feature_importance_plot()
        if feature_importance_plot:
            plots_html += f"""
                <br><h4>Feature Importance Analysis</h4>
                <div class="plot-container">
                    <img src="{feature_importance_plot}" alt="Feature Importance">
                </div>
            """
        
        try:
            model_to_explain = self.best_model_info.get('model')
            model_name = self.best_model_info.get('model_name', 'Model').upper()
            
            X_test_df = pd.DataFrame(
                self.X_test.toarray() 
                if hasattr(self.X_test, 'toarray') 
                else self.X_test
            )
            
            if self.preprocessor:
                try:
                    feature_names = self.preprocessor.get_feature_names_out()
                    X_test_df.columns = feature_names
                except Exception:
                    pass

            explainer = shap.Explainer(model_to_explain, self.X_train)
            shap_values = explainer(X_test_df)
            
            plt.rcParams['figure.facecolor'] = '#0D1117'
            plt.rcParams['axes.facecolor'] = '#161B22'
            shap.summary_plot(shap_values, X_test_df, show=False)
            fig_shap = plt.gcf()
            fig_shap.patch.set_facecolor('#0D1117')
            
            img_base64_shap = self._fig_to_base64(fig_shap)
            plots_html += f"""
                <br><h4>Feature Impact (SHAP Analysis)</h4>
                <div class="plot-container">
                    <img src="{img_base64_shap}" alt="SHAP Summary Plot">
                </div>
            """
        except Exception as e:
            logging.warning(f"Could not create SHAP plot: {e}")
        
        return plots_html

    def generate_html_report(self, filepath: Optional[str] = None) -> str:
        """Generate comprehensive interactive HTML report."""
        if not self.best_model_info:
            msg = "Report cannot be generated. Please run the pipeline first using .fit()."
            logging.error(msg)
            return f"<p>{msg}</p>"
        
        summary_html = self._get_summary_html()
        comparison_table_html = self._get_comparison_table_html()
        plots_html = self._get_plots_html()

        report_id = self._report_id
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Manual Predictor Report</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Exo+2:wght@700&display=swap');
                
                [data-report-id="{report_id}"] {{
                    --bg-dark-1: #0D1117;
                    --bg-dark-2: #161B22;
                    --border-color: #30363D;
                    --primary-blue: #58A6FF;
                    --primary-orange: #F78166;
                    --text-light: #C9D1D9;
                    --text-medium: #8B949E;
                    font-family: 'Roboto', sans-serif;
                    background-color: var(--bg-dark-1);
                    color: var(--text-light);
                    line-height: 1.6;
                }}
                
                [data-report-id="{report_id}"] * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                [data-report-id="{report_id}"] .container {{
                    max-width: 1400px;
                    margin: auto;
                    background-color: var(--bg-dark-2);
                    border: 1px solid var(--border-color);
                    border-radius: 10px;
                    overflow: hidden;
                }}
                
                [data-report-id="{report_id}"] header {{
                    padding: 30px;
                    background: linear-gradient(135deg, #1A2D40 0%, #0D1117 100%);
                    text-align: center;
                    border-bottom: 2px solid var(--border-color);
                }}
                
                [data-report-id="{report_id}"] header h1 {{
                    font-family: 'Exo 2', sans-serif;
                    color: var(--primary-blue);
                    margin: 0;
                    font-size: 2.5rem;
                    text-shadow: 0 2px 10px rgba(88, 166, 255, 0.3);
                }}
                
                [data-report-id="{report_id}"] header p {{
                    margin: 10px 0 0;
                    color: var(--text-medium);
                    font-size: 1.1rem;
                }}
                
                [data-report-id="{report_id}"] .navbar {{
                    display: flex;
                    background-color: var(--bg-dark-2);
                    border-bottom: 1px solid var(--border-color);
                    overflow-x: auto;
                }}
                
                [data-report-id="{report_id}"] .nav-btn {{
                    background: none;
                    border: none;
                    color: var(--text-medium);
                    padding: 15px 25px;
                    cursor: pointer;
                    font-size: 16px;
                    border-bottom: 3px solid transparent;
                    transition: all 0.3s;
                    white-space: nowrap;
                    font-family: 'Roboto', sans-serif;
                }}
                
                [data-report-id="{report_id}"] .nav-btn:hover {{
                    color: var(--text-light);
                    background-color: rgba(88, 166, 255, 0.1);
                }}
                
                [data-report-id="{report_id}"] .nav-btn.active {{
                    color: var(--primary-orange);
                    border-bottom-color: var(--primary-orange);
                    font-weight: 700;
                }}
                
                [data-report-id="{report_id}"] .content-section {{
                    padding: 30px;
                    display: none;
                    animation: fadeIn 0.5s;
                }}
                
                [data-report-id="{report_id}"] .content-section.active {{
                    display: block;
                }}
                
                @keyframes fadeIn {{
                    from {{ opacity: 0; transform: translateY(10px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}
                
                [data-report-id="{report_id}"] h2 {{
                    font-family: 'Exo 2';
                    color: var(--primary-orange);
                    font-size: 2rem;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid var(--border-color);
                }}
                
                [data-report-id="{report_id}"] h4 {{
                    color: var(--primary-blue);
                    margin: 20px 0 15px;
                    font-size: 1.3rem;
                }}
                
                [data-report-id="{report_id}"] .grid-container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                
                [data-report-id="{report_id}"] .grid-item {{
                    background-color: var(--bg-dark-1);
                    padding: 20px;
                    border-radius: 8px;
                    border: 1px solid var(--border-color);
                    transition: transform 0.3s, box-shadow 0.3s;
                }}
                
                [data-report-id="{report_id}"] .grid-item:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 5px 20px rgba(88, 166, 255, 0.2);
                }}
                
                [data-report-id="{report_id}"] .grid-item h4 {{
                    margin-top: 0;
                    color: var(--primary-blue);
                    border-bottom: 1px solid var(--border-color);
                    padding-bottom: 10px;
                }}
                
                [data-report-id="{report_id}"] .score-card {{
                    text-align: center;
                    background: linear-gradient(145deg, #1A2D40, #101820);
                    border: 2px solid var(--primary-orange);
                }}
                
                [data-report-id="{report_id}"] .metric-score {{
                    font-size: 2.8em;               
                    font-weight: 800;               
                    color: #FFB86C;                  
                    text-shadow: 0 0 15px rgba(255, 184, 108, 0.6); 
                    margin: 10px 0;
                    display: block;
                }}

                [data-report-id="{report_id}"] .score-card:hover .metric-score {{
                    transform: scale(1.05);
                    transition: all 0.3s ease;
                    text-shadow: 0 0 25px rgba(255, 184, 108, 0.8);
                }}
                
                [data-report-id="{report_id}"] .plot-container {{
                    background-color: var(--bg-dark-1);
                    padding: 20px;
                    border-radius: 8px;
                    border: 1px solid var(--border-color);
                    margin: 20px 0;
                }}
                
                [data-report-id="{report_id}"] .plot-container img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: auto;
                    border-radius: 4px;
                }}
                
                [data-report-id="{report_id}"] table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                
                [data-report-id="{report_id}"] th {{
                    background-color: #1A2D40;
                    color: var(--primary-blue);
                    padding: 12px;
                    text-align: left;
                    font-weight: bold;
                }}
                
                [data-report-id="{report_id}"] td {{
                    padding: 10px;
                    border-bottom: 1px solid var(--border-color);
                }}

                [data-report-id="{report_id}"] .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }}

                [data-report-id="{report_id}"] .metric-item {{
                    background: linear-gradient(145deg, #1A2D40, #101820);
                    border: 1px solid var(--border-color);
                    border-radius: 10px;
                    text-align: center;
                    padding: 15px 10px;
                    transition: all 0.3s ease;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
                }}

                [data-report-id="{report_id}"] .metric-item:hover {{
                    transform: translateY(-4px);
                    box-shadow: 0 4px 15px rgba(88, 166, 255, 0.3);
                }}

                [data-report-id="{report_id}"] .metric-label {{
                    font-weight: 600;
                    color: var(--primary-blue);
                    font-size: 1rem;
                    text-transform: uppercase;
                    margin-bottom: 5px;
                    display: block;
                }}

                [data-report-id="{report_id}"] .metric-value {{
                    font-size: 1.6em;
                    font-weight: 700;
                    color: #FFB86C;
                    text-shadow: 0 0 10px rgba(255, 184, 108, 0.5);
                }}
                
                [data-report-id="{report_id}"] .params-box {{
                    background-color: var(--bg-dark-1);
                    border: 1px solid var(--border-color);
                    border-radius: 6px;
                    padding: 12px;
                    color: var(--text-light);
                    font-size: 0.9em;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    overflow-wrap: break-word;
                    max-width: 100%;
                    overflow-x: auto;
                }}
                
                [data-report-id="{report_id}"] .scrollable-box {{
                    max-height: 150px;
                    overflow-y: auto;
                    background-color: var(--bg-dark-1);
                    border: 1px solid var(--border-color);
                    border-radius: 6px;
                    padding: 12px;
                    margin-top: 5px;
                }}

                [data-report-id="{report_id}"] .scrollable-box pre {{
                    margin: 0;
                    color: var(--text-light);
                    font-size: 0.9em;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }}

                [data-report-id="{report_id}"] .scrollable-box::-webkit-scrollbar {{
                    width: 8px;
                }}

                [data-report-id="{report_id}"] .scrollable-box::-webkit-scrollbar-track {{
                    background: var(--bg-dark-2);
                    border-radius: 4px;
                }}

                [data-report-id="{report_id}"] .scrollable-box::-webkit-scrollbar-thumb {{
                    background: var(--primary-blue);
                    border-radius: 4px;
                }}

                [data-report-id="{report_id}"] .scrollable-box::-webkit-scrollbar-thumb:hover {{
                    background: var(--primary-orange);
                }}
            </style>
        </head>
        <body>
            <div data-report-id="{report_id}">
                <div class="container">
                    <header>
                        <h1>Manual Predictor Analysis Report</h1>
                        <p>Comprehensive Machine Learning Pipeline Results</p>
                    </header>
                    <nav class="navbar">
                        <button class="nav-btn active" onclick="showTab(event, 'summary', '{report_id}')">Summary</button>
                        <button class="nav-btn" onclick="showTab(event, 'comparison', '{report_id}')">Model Comparison</button>
                        <button class="nav-btn" onclick="showTab(event, 'visualizations', '{report_id}')">Visualizations</button>
                    </nav>
                    <main>
                        <section id="summary-{report_id}" class="content-section active">
                            <h2>Execution Summary</h2>
                            {summary_html}
                        </section>
                        <section id="comparison-{report_id}" class="content-section">
                            <h2>Detailed Metric Comparison</h2>
                            {comparison_table_html}
                        </section>
                        <section id="visualizations-{report_id}" class="content-section">
                            <h2>Result Visualizations</h2>
                            {plots_html}
                        </section>
                    </main>
                </div>
            </div>
            <script>
                function showTab(event, tabName, reportId) {{
                    const reportScope = document.querySelector('[data-report-id="' + reportId + '"]');
                    
                    reportScope.querySelectorAll('.content-section').forEach(section => {{
                        section.classList.remove('active');
                    }});
                    
                    reportScope.querySelectorAll('.nav-btn').forEach(btn => {{
                        btn.classList.remove('active');
                    }});
                    
                    const sectionId = tabName + '-' + reportId;
                    reportScope.querySelector('#' + sectionId).classList.add('active');
                    
                    event.currentTarget.classList.add('active');
                }}
            </script>
        </body>
        </html>
        """
        
        if filepath:
            save_path = filepath
        elif self.output_dir:
            save_path = os.path.join(self.output_dir, 'reports', 'analysis_report.html')
        else:
            return html_template

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_template)
            logging.info(f"HTML report successfully saved to: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save HTML report: {e}")

        return html_template

    def display_report(self) -> None:
        """Display HTML report in Jupyter/Colab notebook output cell."""
        logging.info("Preparing report for display in output cell...")
        try:
            html_content = self.generate_html_report()
            display(HTML(html_content))
            logging.info("Report displayed successfully.")
        except (NameError, ImportError, ModuleNotFoundError) as e:
            logging.warning(
                f"Cannot display report: {e}. "
                "Ensure you are running this in a Jupyter/Colab environment."
            )
        except Exception as e:
            logging.error(f"Failed to display report: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def __repr__(self) -> str:
        """String representation of ManualPredictor instance."""
        status = "Trained" if self.best_model_info else "Not Trained"
        best_model = (
            self.best_model_info.get('model_name', 'None') 
            if self.best_model_info 
            else 'None'
        )
        models_trained = len([r for r in self.all_results if 'error' not in r])
        
        return (
            f"NoventisManualML(\n"
            f"  task='{self.task_type}',\n"
            f"  status='{status}',\n"
            f"  best_model='{best_model}',\n"
            f"  models_trained={models_trained},\n"
            f"  data_cleaner_used={self.data_cleaner_used}\n"
            f")"
        )

    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.__repr__()


