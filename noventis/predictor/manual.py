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



class NoventisManualPredictor:
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
        data_cleaner: Optional[Any] = None,
        tune_hyperparameters: bool = False,
        n_trials: int = 50,
        cv_folds: int = 3,
        enable_feature_engineering: bool = False,
        cv_strategy: str = 'repeated',
        show_tuning_plots: bool = False,
        output_dir: Optional[str] = None
    ):
        """
        Initialize ManualPredictor instance.
        
        Args:
            model_name: Single model name or list of models to train
            task: Task type ('classification' or 'regression')
            random_state: Random seed for reproducibility
            data_cleaner: Optional external data cleaning pipeline
            tune_hyperparameters: Whether to perform Optuna hyperparameter tuning
            n_trials: Number of Optuna trials for tuning
            cv_folds: Number of cross-validation folds
            enable_feature_engineering: Whether to create polynomial features
            cv_strategy: CV strategy ('repeated' or 'standard')
            show_tuning_plots: Whether to display Optuna optimization visualizations
            output_dir: Directory to save results, models, and reports
            
        Raises:
            ValueError: If task is not 'classification' or 'regression'
        """
        self._report_id = f"report-{id(self)}"
        self.model_name = model_name
        self.task_type = task.lower()  # Renamed from self.task to match AutoML
        self.random_state = random_state
        self.data_cleaner = data_cleaner
        self.tune_hyperparameters = tune_hyperparameters
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.enable_feature_engineering = enable_feature_engineering
        self.cv_strategy = cv_strategy
        self.show_tuning_plots = show_tuning_plots
        self.output_dir = self._setup_output_directory(output_dir)

        # Result storage matching AutoML structure
        self.results = {}
        self.best_model_info: Dict[str, Any] = {}
        self.all_results: List[Dict[str, Any]] = []
        
        # Data storage matching AutoML naming
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor: Optional[ColumnTransformer] = None

        # Validate task type
        if self.task_type not in ['classification', 'regression']:
            raise ValueError(
                f"Task must be 'classification' or 'regression', got '{task}'"
            )
        
        logging.info(f"ManualPredictor initialized for {self.task_type}")

    def _setup_output_directory(self, output_dir: Optional[str]) -> Optional[str]:
        """
        Create organized directory structure for outputs.
        
        Creates timestamped run directory with subdirectories:
        - plots/: Visualization images
        - models/: Saved model files
        - reports/: HTML and text reports
        
        Args:
            output_dir: Base output directory path
            
        Returns:
            Path to created run directory, or None if output_dir not specified
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
        """
        Initialize a model with default configuration.
        
        Handles model-specific settings like verbosity and evaluation metrics.
        
        Args:
            name: Model name from MODEL_CONFIG
            
        Returns:
            Initialized model instance
            
        Raises:
            ValueError: If model name not recognized for task type
        """
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
        
        # Add random_state if model supports it
        if 'random_state' in model_class().get_params():
            params['random_state'] = self.random_state
        
        # Model-specific configurations
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
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Dictionary with MAE, MSE, RMSE, and R² score
        """
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
        """
        Perform Bayesian hyperparameter optimization using Optuna.
        
        Uses cross-validation to evaluate hyperparameter combinations
        and MedianPruner to stop unpromising trials early.
        
        Args:
            model_name: Name of model to tune
            X_train: Training features (preprocessed)
            y_train: Training labels
            
        Returns:
            Dictionary of best hyperparameters found
        """
        logging.info(f"Starting hyperparameter tuning for {model_name.upper()}...")
        
        param_func = MODEL_CONFIG[self.task_type][model_name.lower()].get('params')
        if not param_func:
            logging.warning(
                f"No hyperparameter search space defined for '{model_name}'. "
                "Using default parameters."
            )
            return {}

        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function for single trial evaluation."""
            # Get hyperparameters for this trial
            params = param_func(trial)
            model = self._load_single_model(model_name)
            model.set_params(**params)
            
            # Setup cross-validation strategy
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

            # Evaluate on all folds
            scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_train_fold = X_train[train_idx]
                X_val_fold = X_train[val_idx]
                y_train_fold = y_train.iloc[train_idx]
                y_val_fold = y_train.iloc[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                preds = model.predict(X_val_fold)
                
                # Calculate primary metric
                if self.task_type == 'classification':
                    metrics = self._eval_classification(y_val_fold, preds)
                    metric = self.DEFAULT_CLASSIFICATION_METRIC
                else:
                    metrics = self._eval_regression(y_val_fold, preds)
                    metric = self.DEFAULT_REGRESSION_METRIC
                
                scores.append(metrics[metric])
            
            return np.mean(scores)

        # Run Optuna optimization
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(direction='maximize', pruner=pruner)
        study.optimize(
            objective, 
            n_trials=self.n_trials, 
            timeout=600, 
            n_jobs=-1, 
            show_progress_bar=False
        )
        
        # Display optimization visualizations if requested
        if self.show_tuning_plots:
            self._show_tuning_insights(study, model_name)
        
        logging.info(f"Tuning complete. Best parameters: {study.best_params}")
        return study.best_params

    def _show_tuning_insights(self, study: optuna.Study, model_name: str) -> None:
        """
        Display Optuna optimization visualizations.
        
        Creates and optionally saves:
        - Optimization history plot
        - Parameter importance plot
        
        Args:
            study: Completed Optuna study
            model_name: Model name for plot titles
        """
        logging.info(f"Displaying tuning visualizations for {model_name.upper()}...")
        
        try:
            # Optimization history
            fig1 = optuna.visualization.plot_optimization_history(study)
            fig1.update_layout(title=f'Optimization History for {model_name}')
            if self.output_dir:
                save_path = os.path.join(
                    self.output_dir, 'plots', 
                    f'{model_name}_optimization_history.png'
                )
                fig1.write_image(save_path)
            fig1.show()

            # Parameter importance
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
        """
        Complete pipeline for training and evaluating a single model.
        
        Workflow:
        1. Hyperparameter tuning (if enabled)
        2. Model training with best/default parameters
        3. Prediction on test set
        4. Metric calculation
        
        Args:
            model_name: Name of model to train
            X_train: Training features (preprocessed)
            y_train: Training labels
            X_test: Test features (preprocessed)
            y_test: Test labels
            
        Returns:
            Dictionary with model, predictions, metrics, and metadata
        """
        logging.info(f"Processing model: {model_name.upper()}")
        
        # Hyperparameter tuning
        best_params = {}
        if self.tune_hyperparameters:
            best_params = self._tune_with_optuna(model_name, X_train, y_train)

        # Initialize and configure model
        model = self._load_single_model(model_name)
        if best_params:
            model.set_params(**best_params)

        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        logging.info(f"Training finished in {training_time:.2f} seconds.")
        
        # Generate predictions
        predictions = model.predict(X_test)
        y_pred_proba = (
            model.predict_proba(X_test) 
            if hasattr(model, 'predict_proba') 
            else None
        )
        
        # Calculate metrics
        if self.task_type == 'classification':
            metrics = self._eval_classification(y_test, predictions)
        else:
            metrics = self._eval_regression(y_test, predictions)
        
        # Return comprehensive result dictionary matching AutoML structure
        return {
            'model_name': model_name,
            'model': model,  # Changed from 'model_object' to match AutoML
            'predictions': predictions,
            'prediction_proba': y_pred_proba,
            'actual': y_test,
            'metrics': metrics,
            'task_type': self.task_type,  # Added for consistency
            'training_time_seconds': training_time,
            'best_params': best_params,
            'best_config': best_params,  # Alias for AutoML compatibility
            'feature_importance': None,  # Placeholder
            'best_estimator': model_name  # For consistency
        }

    def fit(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        compare: bool = False,
        explain: bool = False,
        chosen_metric: Optional[str] = None,
        display_report: bool = True
    ) -> Dict[str, Any]:
        """
        Main training pipeline (renamed from run_pipeline to match AutoML).
        
        Complete workflow:
        1. Train-test split with stratification
        2. External data cleaning (if provided)
        3. Internal preprocessing (imputation + encoding)
        4. Model training for all specified models
        5. Best model selection
        6. Optional: Comparison table, visualizations, HTML report
        
        Args:
            df: Input DataFrame with features and target
            target_column: Name of target column
            test_size: Proportion of data for testing (0.0-1.0)
            compare: Whether to print comparison table
            explain: Whether to create metric visualization plots
            chosen_metric: Specific metric to visualize (uses primary if None)
            display_report: Whether to display HTML report in notebook
            
        Returns:
            Dictionary with 'best_model_details' and 'all_model_results'
            
        Raises:
            RuntimeError: If no models train successfully
        """
        logging.info("Starting ManualPredictor Training Pipeline")

        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Train-test split with stratification for classification
        stratify = y if self.task_type == 'classification' else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state, 
            stratify=stratify
        )
        
        logging.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        
        # Apply external data cleaner if provided
        if self.data_cleaner and NoventisDataCleaner is not None:
            logging.info("External DataCleaner detected. Applying transformations...")
            self.data_cleaner.fit(X_train, y_train)
            X_train = self.data_cleaner.transform(X_train)
            X_test = self.data_cleaner.transform(X_test)
            logging.info("External DataCleaner processing complete.")
        
        # Internal preprocessing pipeline
        logging.info("Running internal preprocessor...")
        numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )

        self.preprocessor.fit(X_train)
        X_train_transformed = self.preprocessor.transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)
        logging.info("Internal preprocessing complete. All data is now numeric.")
        
        # Feature engineering (currently disabled for stability)
        if self.enable_feature_engineering:
            logging.warning("Feature engineering is currently disabled for stability.")
        
        # Store processed data matching AutoML naming
        self.X_train = X_train_transformed
        self.X_test = X_test_transformed
        self.y_train = y_train
        self.y_test = y_test

        # Train all specified models
        model_list = (
            self.model_name 
            if isinstance(self.model_name, list) 
            else [self.model_name]
        )
        self.all_results = []
        
        for name in model_list:
            try:
                result = self._run_single_model_pipeline(
                    name, self.X_train, y_train, self.X_test, y_test
                )
                self.all_results.append(result)
                self.results[name] = result  # Store in results dict
                
            except Exception as e:
                logging.error(f"Failed to process model {name}: {e}")
                self.all_results.append({
                    'model_name': name, 
                    'metrics': {}, 
                    'error': str(e)
                })

        # Check if any models succeeded
        successful_results = [
            res for res in self.all_results if 'error' not in res
        ]
        if not successful_results:
            raise RuntimeError(
                "No models were trained successfully. "
                "Please check your data or configuration."
            )

        # Select best model based on primary metric
        primary_metric = (
            self.DEFAULT_CLASSIFICATION_METRIC 
            if self.task_type == 'classification' 
            else self.DEFAULT_REGRESSION_METRIC
        )
        
        self.best_model_info = max(
            successful_results, 
            key=lambda x: x['metrics'].get(primary_metric, -1)
        )
        
       
        logging.info("Process Complete!")
        logging.info(
            f"Best Model: {self.best_model_info['model_name'].upper()} "
            f"with {primary_metric} = {self.best_model_info['metrics'][primary_metric]:.4f}"
        )

        # Optional outputs
        if compare:
            self._print_comparison()
        if explain:
            self._create_metric_plot(chosen_metric)
        if display_report:
            self.display_report()

        return {
            'best_model_details': self.best_model_info,
            'all_model_results': self.all_results
        }

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get model comparison results as DataFrame.
        
        Returns DataFrame with models as index and metrics as columns,
        sorted by primary metric (descending for better-is-higher metrics).
        
        Returns:
            DataFrame with model performance comparison
        """
        if not self.all_results:
            logging.warning(
                "No results available. "
                "Please run the pipeline first using .fit()."
            )
            return pd.DataFrame()
        
        # Build records from successful results
        records = [
            {'model': res['model_name'], **res['metrics']}
            for res in self.all_results if 'error' not in res
        ]
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records).set_index('model')
        
        # Sort by primary metric
        primary_metric = (
            self.DEFAULT_CLASSIFICATION_METRIC 
            if self.task_type == 'classification' 
            else self.DEFAULT_REGRESSION_METRIC
        )
        
        is_higher_better = primary_metric in [
            'f1_score', 'r2_score', 'accuracy', 'precision', 'recall'
        ]
        
        return df.sort_values(by=primary_metric, ascending=not is_higher_better)

    def _print_comparison(self) -> None:
        """
        Print formatted comparison table of all trained models.
        """
        if not self.all_results:
            logging.warning("No results to compare.")
            return
        
    
        print("MODEL COMPARISON - ALL METRICS")
        print(self.get_results_dataframe())

    def _create_metric_plot(self, chosen_metric: Optional[str] = None) -> None:
        """
        Create bar chart comparing models by chosen metric.
        
        Highlights the best performing model in gold color.
        
        Args:
            chosen_metric: Metric to plot (uses primary metric if None)
        """
        if not self.all_results:
            logging.warning("No results to plot.")
            return
        
        # Determine metric to plot
        metric = chosen_metric or (
            self.DEFAULT_CLASSIFICATION_METRIC 
            if self.task_type == 'classification' 
            else self.DEFAULT_REGRESSION_METRIC
        )
        
        df_results = self.get_results_dataframe().reset_index()
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(
            df_results['model'], 
            df_results[metric],
            color='steelblue',
            alpha=0.8,
            edgecolor='black',
            linewidth=1.2
        )
        
        ax.set_title(
            f'Model Comparison - {metric.upper()}',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.0,
                yval + (df_results[metric].max() * 0.01),
                f'{yval:.4f}',
                ha='center',
                va='bottom',
                fontweight='bold'
            )
        
        # Highlight best model in gold
        is_higher_better = metric in [
            'accuracy', 'precision', 'recall', 'f1_score', 'r2_score'
        ]
        best_val = (
            df_results[metric].max() 
            if is_higher_better 
            else df_results[metric].min()
        )
        best_bar_index = df_results[df_results[metric] == best_val].index[0]
        
        bars[best_bar_index].set_color('gold')
        bars[best_bar_index].set_edgecolor('darkorange')
        bars[best_bar_index].set_linewidth(2)
        
        # Save plot if output directory specified
        if self.output_dir:
            plot_path = os.path.join(
                self.output_dir, 'plots', 
                f'metric_comparison_{metric}.png'
            )
            fig.savefig(plot_path, bbox_inches='tight', dpi=300)
            logging.info(f"Plot saved to: {plot_path}")
        
        plt.tight_layout()
        plt.show()

    def save_model(self, filepath: Optional[str] = None) -> None:
        """
        Save best trained model to disk using pickle.
        
        Args:
            filepath: Full path to save model. If None, uses output_dir
            
        Raises:
            ValueError: If no model available or no valid save path
        """
        if not self.best_model_info:
            raise ValueError("No best model available to save. Run fit() first.")
        
        model_to_save = self.best_model_info.get('model')
        model_name = self.best_model_info['model_name']
        
        if filepath:
            save_path = filepath
        elif self.output_dir:
            save_path = os.path.join(
                self.output_dir, 'models', 
                f'{model_name}_best_model.pkl'
            )
        else:
            raise ValueError(
                "Please provide a 'filepath' or set an 'output_dir' "
                "during initialization."
            )
        
        logging.info(f"Saving model '{model_name}' to {save_path}...")
        with open(save_path, 'wb') as f:
            pickle.dump(model_to_save, f)
        logging.info("Model saved successfully.")

    @staticmethod
    def load_model(filepath: str) -> Any:
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to saved model file
            
        Returns:
            Loaded model object
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        logging.info(f"Loading model from {filepath}...")
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded successfully.")
        return model

    def predict(
        self, 
        X_new: Union[pd.DataFrame, np.ndarray], 
        model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make predictions on new data (added to match AutoML interface).
        
        Args:
            X_new: New features for prediction
            model_path: Optional path to load specific model
            
        Returns:
            Dictionary with predictions and probabilities (if classification)
        """
        # Load model from path or use trained model
        model = (
            self.load_model(model_path) 
            if model_path 
            else self.best_model_info.get('model')
        )
        
        if model is None:
            raise ValueError(
                "No model available. Train first with fit() or specify model_path"
            )
        
        # Preprocess new data
        if isinstance(X_new, pd.DataFrame):
            X_new_transformed = self.preprocessor.transform(X_new)
        else:
            X_new_transformed = X_new
        
        try:
            predictions = model.predict(X_new_transformed)
            logging.info(f"Prediction successful for {len(X_new)} samples")
            
            result = {'predictions': predictions}
            
            # Add probabilities for classification
            if (self.task_type == "classification" and 
                hasattr(model, 'predict_proba')):
                result['probabilities'] = model.predict_proba(X_new_transformed)
            
            return result
            
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return None

    def explain_model(
        self, 
        model_object: Optional[Any] = None, 
        plot_type: str = 'summary', 
        feature: Optional[str] = None
    ) -> None:
        """
        Generate SHAP-based model explanations.
        
        Creates interpretability visualizations showing how features
        impact model predictions using SHAP values.
        
        Args:
            model_object: Model to explain (uses best model if None)
            plot_type: Type of SHAP plot ('summary', 'beeswarm', 'dependence')
            feature: Feature name (required for dependence plots)
            
        Raises:
            RuntimeError: If pipeline hasn't been run
            ValueError: If feature not provided for dependence plot
        """
        if self.X_test is None:
            raise RuntimeError("The pipeline must be run before explaining the model.")
        
        model_to_explain = model_object or self.best_model_info.get('model')
        if not model_to_explain:
            raise ValueError("No model available to explain.")

        model_name = self.best_model_info.get('model_name', 'Model').upper()
        logging.info(f"Creating SHAP Explainer for {model_name}...")

        try:
            # Prepare test data as DataFrame
            X_test_df = pd.DataFrame(
                self.X_test.toarray() 
                if hasattr(self.X_test, 'toarray') 
                else self.X_test
            )
            
            # Try to get feature names from preprocessor
            if self.preprocessor:
                try:
                    feature_names = self.preprocessor.get_feature_names_out()
                    X_test_df.columns = feature_names
                except Exception:
                    logging.warning(
                        "Could not retrieve feature names from preprocessor. "
                        "Plots may use generic names."
                    )
            
            # Create SHAP explainer and compute values
            explainer = shap.Explainer(model_to_explain, self.X_train)
            shap_values = explainer(X_test_df)
            
        except Exception as e:
            logging.error(f"Failed to create SHAP explainer: {e}")
            return
        
        # Generate requested plot type
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
        """
        Convert matplotlib figure to base64 string for HTML embedding.
        
        Args:
            fig: Matplotlib figure object
            
        Returns:
            Base64 encoded image with data URI prefix
        """
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
        """
        Generate comprehensive classification analysis plots.
        
        Creates 4-panel visualization:
        1. Confusion matrix (normalized)
        2. ROC curve (binary only)
        3. Precision-Recall curve (binary only)
        4. Model comparison bar chart
        
        Returns:
            Dictionary with plot name and base64 encoded image
        """
        plots = {}
        self._set_plot_style()
        
        best_result = self.best_model_info
        y_true = best_result['actual']
        y_pred = best_result['predictions']
        y_pred_proba = best_result.get('prediction_proba')
        
        # Create 2x2 subplot grid
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
        
        # Panel 2: ROC Curve (binary classification only)
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
        
        # Panel 3: Precision-Recall Curve (binary classification only)
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
        
        # Add value labels on bars
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
        """
        Generate comprehensive regression analysis plots.
        
        Creates 4-panel visualization:
        1. Predicted vs Actual scatter plot
        2. Residuals plot
        3. Residuals distribution histogram
        4. Model comparison bar chart
        
        Returns:
            Dictionary with plot name and base64 encoded image
        """
        plots = {}
        self._set_plot_style()
        
        best_result = self.best_model_info
        y_true = best_result['actual']
        y_pred = best_result['predictions']
        
        # Create 2x2 subplot grid
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
        
        # Add value labels on bars
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
        """
        Create horizontal bar plot of top 20 feature importances.
        
        Only works for tree-based models with feature_importances_ attribute.
        
        Returns:
            Base64 encoded image string, or empty string if not available
        """
        try:
            model = self.best_model_info['model']
            
            if not hasattr(model, 'feature_importances_'):
                return ""
            
            importances = model.feature_importances_
            
            # Try to get feature names from preprocessor
            try:
                feature_names = self.preprocessor.get_feature_names_out()
            except:
                feature_names = [f'Feature_{i}' for i in range(len(importances))]
            
            # Get top 20 features
            indices = np.argsort(importances)[::-1][:20]
            top_features = [feature_names[i] for i in indices]
            top_importances = importances[indices]
            
            # Create plot
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
            
            # Add value labels
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
        """
        Generate HTML for summary section of report.
        
        Returns:
            HTML string with process summary and best model information
        """
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
        
        # Generate metrics HTML
        all_metrics_html = ""
        for metric_name, metric_value in self.best_model_info['metrics'].items():
            all_metrics_html += f"""
                <div class='metric-item'>
                    <span class='metric-label'>{metric_name.replace('_', ' ').title()}</span>
                    <span class='metric-value'>{metric_value:.4f}</span>
                </div>
            """
        
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
                <p><strong>Data Cleaner Used:</strong> {'Yes' if self.data_cleaner else 'No'}</p>
            </div>
            <div class="grid-item score-card">
                <h4>Best Model</h4>
                <p class="model-name">{self.best_model_info['model_name'].upper()}</p>
                <p class="metric-score">{primary_metric.replace('_', ' ').title()}: {best_score:.4f}</p>
                <p class="training-time">Training Time: {training_time:.2f}s</p>
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
                <p><strong>Training Samples:</strong> {self.X_train.shape[0]}</p>
                <p><strong>Test Samples:</strong> {self.X_test.shape[0]}</p>
                <p><strong>Number of Features:</strong> {self.X_train.shape[1]}</p>
                <p><strong>Target Distribution (Train):</strong></p>
                <pre class="params-box">{self.y_train.value_counts().to_dict()}</pre>
            </div>
        </div>
        """

    def _get_comparison_table_html(self) -> str:
        """
        Generate HTML comparison table of all trained models.
        
        Returns:
            HTML string with styled DataFrame table
        """
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
        """
        Generate HTML section with all visualizations.
        
        Creates task-specific plots, feature importance, and SHAP analysis.
        
        Returns:
            HTML string with embedded base64 encoded images
        """
        if self.X_test is None:
            return "<p>Plots cannot be generated because the pipeline has not been run.</p>"
        
        plots_html = ""
        
        # Task-specific analysis plots
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
        
        # Feature importance plot
        feature_importance_plot = self._create_feature_importance_plot()
        if feature_importance_plot:
            plots_html += f"""
                <br><h4>Feature Importance Analysis</h4>
                <div class="plot-container">
                    <img src="{feature_importance_plot}" alt="Feature Importance">
                </div>
            """
        
        # SHAP analysis plot
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
        """
        Generate comprehensive interactive HTML report.
        
        Creates professional dark-themed report with:
        - Executive summary with key metrics
        - Detailed model comparison table
        - Comprehensive visualizations
        - Interactive tabbed navigation
        
        Args:
            filepath: Path to save HTML file. If None, returns HTML string only
            
        Returns:
            Complete HTML string
        """
        if not self.best_model_info:
            msg = "Report cannot be generated. Please run the pipeline first using .fit()."
            logging.error(msg)
            return f"<p>{msg}</p>"
        
        # Generate report sections
        summary_html = self._get_summary_html()
        comparison_table_html = self._get_comparison_table_html()
        plots_html = self._get_plots_html()

        report_id = self._report_id
        
        # Complete HTML template with FIXED CSS selectors
        html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Manual Predictor Report</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Exo+2:wght@700&display=swap');
            .{report_id} {{
                --bg-dark-1: #0D1117; --bg-dark-2: #161B22; --border-color: #30363D;
                --primary-blue: #58A6FF; --primary-orange: #F78166;
                --text-light: #C9D1D9; --text-medium: #8B949E;
                font-family: 'Roboto', sans-serif; background-color: var(--bg-dark-1);
                color: var(--text-light); line-height: 1.6;
            }}
            .{report_id} * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            .{report_id} .container {{
                max-width: 1400px; margin: auto; background-color: var(--bg-dark-2);
                border: 1px solid var(--border-color); border-radius: 10px; overflow: hidden;
            }}
            .{report_id} header {{
                padding: 30px; background: linear-gradient(135deg, #1A2D40 0%, #0D1117 100%);
                text-align: center; border-bottom: 2px solid var(--border-color);
            }}
            .{report_id} header h1 {{ font-family: 'Exo 2', sans-serif; color: var(--primary-blue); 
                        margin: 0; font-size: 2.5rem; text-shadow: 0 2px 10px rgba(88, 166, 255, 0.3); }}
            .{report_id} header p {{ margin: 10px 0 0; color: var(--text-medium); font-size: 1.1rem; }}
            .{report_id} .navbar {{ display: flex; background-color: var(--bg-dark-2); 
                    border-bottom: 1px solid var(--border-color); overflow-x: auto; }}
            .{report_id} .nav-btn {{
                background: none; border: none; color: var(--text-medium);
                padding: 15px 25px; cursor: pointer; font-size: 16px;
                border-bottom: 3px solid transparent; transition: all 0.3s;
                white-space: nowrap;
            }}
            .{report_id} .nav-btn:hover {{ color: var(--text-light); background-color: rgba(88, 166, 255, 0.1); }}
            .{report_id} .nav-btn.active {{ color: var(--primary-orange); border-bottom-color: var(--primary-orange); 
                            font-weight: 700; }}
            .{report_id} .content-section {{ padding: 30px; display: none; animation: fadeIn 0.5s; }}
            .{report_id} .content-section.active {{ display: block; }}
            @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} 
                                to {{ opacity: 1; transform: translateY(0); }} }}
            .{report_id} h2 {{ font-family: 'Exo 2'; color: var(--primary-orange); font-size: 2rem; 
                margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid var(--border-color); }}
            .{report_id} h4 {{ color: var(--primary-blue); margin: 20px 0 15px; font-size: 1.3rem; }}
            .{report_id} .grid-container {{
                display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                gap: 20px; margin-bottom: 30px;
            }}
            .{report_id} .grid-item {{
                background-color: var(--bg-dark-1); padding: 20px;
                border-radius: 8px; border: 1px solid var(--border-color);
                transition: transform 0.3s, box-shadow 0.3s;
            }}
            .{report_id} .grid-item:hover {{ transform: translateY(-5px); 
                            box-shadow: 0 5px 20px rgba(88, 166, 255, 0.2); }}
            .{report_id} .grid-item h4 {{ margin-top: 0; color: var(--primary-blue); 
                            border-bottom: 1px solid var(--border-color); padding-bottom: 10px; }}
            .{report_id} .grid-item p {{ color: var(--text-medium); margin: 10px 0; }}
            .{report_id} .grid-item strong {{ color: var(--text-light); }}
            .{report_id} .score-card {{
                text-align: center; background: linear-gradient(145deg, #1A2D40, #101820);
                border: 2px solid var(--primary-orange);
            }}
            .{report_id} .score-card .model-name {{
                font-family: 'Exo 2'; font-size: 2.2em;
                color: var(--primary-orange); margin: 15px 0;
                text-shadow: 0 2px 10px rgba(247, 129, 102, 0.5);
            }}
            .{report_id} .score-card .metric-score {{ font-size: 1.6em; color: var(--primary-blue); 
                                        margin: 10px 0; font-weight: 700; }}
            .{report_id} .score-card .training-time {{ font-size: 1.1em; color: var(--text-medium); margin-top: 10px; }}
            .{report_id} .metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px; }}
            .{report_id} .metric-item {{ display: flex; justify-content: space-between; padding: 8px 12px;
                        background-color: var(--bg-dark-2); border-radius: 4px;
                        border: 1px solid var(--border-color); }}
            .{report_id} .metric-label {{ color: var(--text-medium); font-size: 0.9em; }}
            .{report_id} .metric-value {{ color: var(--primary-blue); font-weight: 700; }}
            .{report_id} .params-box {{
                background-color: #010409; padding: 15px; border-radius: 4px;
                font-family: 'Courier New', monospace; font-size: 0.9em;
                white-space: pre-wrap; word-wrap: break-word; color: var(--text-light);
                border: 1px solid var(--border-color); margin-top: 10px;
                max-height: 200px; overflow-y: auto;
            }}
            .{report_id} .params-box::-webkit-scrollbar {{ width: 8px; }}
            .{report_id} .params-box::-webkit-scrollbar-track {{ background: var(--bg-dark-1); }}
            .{report_id} .params-box::-webkit-scrollbar-thumb {{ background: var(--border-color); border-radius: 4px; }}
            .{report_id} .table-container {{ overflow-x: auto; margin-top: 20px; }}
            .{report_id} .table-container table {{
                width: 100%; border-collapse: collapse; background-color: var(--bg-dark-1);
                border-radius: 8px; overflow: hidden;
            }}
            .{report_id} .table-container th, .{report_id} .table-container td {{
                padding: 12px 15px; text-align: left; border-bottom: 1px solid var(--border-color);
            }}
            .{report_id} .table-container thead {{ background-color: #1A2D40; }}
            .{report_id} .table-container tbody tr:hover {{ background-color: #222b38; }}
            .{report_id} .plot-container {{
                background-color: var(--bg-dark-1); padding: 20px; border-radius: 8px;
                border: 1px solid var(--border-color); margin: 20px 0;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            }}
            .{report_id} .plot-container img {{ max-width: 100%; height: auto; display: block; margin: auto; 
                                border-radius: 4px; }}
            .{report_id} .info-box {{
                background: linear-gradient(135deg, rgba(88, 166, 255, 0.1), rgba(22, 27, 34, 0.5));
                border-left: 4px solid var(--primary-blue); padding: 15px 20px;
                margin: 20px 0; border-radius: 4px;
            }}
            .{report_id} .info-box p {{ margin: 5px 0; color: var(--text-light); }}
            .{report_id} .shap-interpretation {{
                background-color: var(--bg-dark-1); 
                padding: 20px; 
                border-radius: 8px;
                border: 1px solid var(--border-color); 
                margin: 20px 0;
            }}
            .{report_id} .shap-interpretation h5 {{
                margin-top: 0;
                font-size: 1.3em;
            }}
            .{report_id} .shap-interpretation h6 {{
                font-size: 1.1em;
            }}
            .{report_id} .shap-interpretation li {{
                line-height: 1.6;
            }}
            @media (max-width: 768px) {{
                .{report_id} .grid-container {{ grid-template-columns: 1fr; }}
                .{report_id} .navbar {{ flex-direction: column; }}
                .{report_id} .nav-btn {{ width: 100%; text-align: left; }}
            }}
        </style>
    </head>
    <body>
        <div class="{report_id}">
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
                    <section id="summary" class="content-section active">
                        <h2>Execution Summary</h2>
                        {summary_html}
                    </section>
                    <section id="comparison" class="content-section">
                        <h2>Detailed Metric Comparison</h2>
                        <div class="info-box">
                            <p><strong>Interpretation Guide:</strong></p>
                            <p>Higher values indicate better performance for: Accuracy, Precision, Recall, F1 Score, R² Score</p>
                            <p>Lower values indicate better performance for: MAE, MSE, RMSE</p>
                        </div>
                        {comparison_table_html}
                    </section>
                    <section id="visualizations" class="content-section">
                        <h2>Result Visualizations</h2>
                        {plots_html}
                    </section>
                </main>
            </div>
        </div>
        <script>
            function showTab(event, tabName, reportId) {{
                const reportScope = document.querySelector('.' + reportId);
                reportScope.querySelectorAll('.content-section').forEach(section => {{
                    section.style.display = 'none';
                    section.classList.remove('active');
                }});
                reportScope.querySelectorAll('.nav-btn').forEach(btn => {{
                    btn.classList.remove('active');
                }});
                reportScope.querySelector('#' + tabName).style.display = 'block';
                reportScope.querySelector('#' + tabName).classList.add('active');
                event.currentTarget.classList.add('active');
            }}
        </script>
    </body>
    </html>
            """
        
        # Save to file if path provided
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
        """
        Display HTML report in Jupyter/Colab notebook output cell.
        """
        logging.info("Preparing report for display in output cell...")
        try:
            html_content = self.generate_html_report()
            display(HTML(html_content))
            logging.info("Report displayed successfully.")
        except NameError:
            logging.warning(
                "Cannot display report. Ensure you are running this in a "
                "Jupyter/Colab environment and that 'display' and 'HTML' "
                "from IPython.display are imported."
            )
        except Exception as e:
            logging.error(f"Failed to display report: {e}")

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
            f"ManualPredictor(\n"
            f"  task='{self.task_type}',\n"
            f"  status='{status}',\n"
            f"  best_model='{best_model}',\n"
            f"  models_trained={models_trained}\n"
            f")"
        )

    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.__repr__()


# # Module metadata
# __version__ = "2.0.0"
# __author__ = "Noventis Team"
# __all__ = ['ManualPredictor']