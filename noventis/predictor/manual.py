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
from IPython.display import display, HTML
warnings.filterwarnings('ignore') # Suppress warnings for a cleaner output

import optuna
import shap
optuna.logging.set_verbosity(optuna.logging.ERROR) # Show only errors from Optuna


from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
try:
    # Attempt to import a custom data cleaner if available
    from noventis_beta.data_cleaner import NoventisDataCleaner
except ImportError:
    NoventisDataCleaner = None # If not found, set to None to avoid errors

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)



def get_rf_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines the search space for Random Forest Classifier."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
    }

def get_xgb_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines the search space for XGBoost Classifier."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
    }

def get_lgbm_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines the search space for LightGBM Classifier."""
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
    }
    
def get_catboost_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines the search space for CatBoost (both Classifier and Regressor)."""
    return {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
    }

def get_rf_reg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines the search space for Random Forest Regressor."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
    }

def get_xgb_reg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines the search space for XGBoost Regressor."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }

def get_lgbm_reg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines the search space for LightGBM Regressor."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
    }

# Central registry mapping model names to their classes and parameter search spaces.
MODEL_CONFIG = {
    'classification': {
        'logistic_regression': {'model': LogisticRegression, 'params': None}, # No tuning for this one
        'decision_tree': {'model': DecisionTreeClassifier, 'params': get_dt_params},
        'random_forest': {'model': RandomForestClassifier, 'params': get_rf_params},
        'gradient_boosting': {'model': GradientBoostingClassifier, 'params': get_gb_params},
        'xgboost': {'model': xgb.XGBClassifier, 'params': get_xgb_params},
        'lightgbm': {'model': lgb.LGBMClassifier, 'params': get_lgbm_params},
        'catboost': {'model': cb.CatBoostClassifier, 'params': get_catboost_params}
    },
    'regression': {
        'linear_regression': {'model': LinearRegression, 'params': None}, # No tuning for this one
        'decision_tree': {'model': DecisionTreeRegressor, 'params': get_dt_params},
        'random_forest': {'model': RandomForestRegressor, 'params': get_rf_reg_params},
        'gradient_boosting': {'model': GradientBoostingRegressor, 'params': get_gb_params},
        'xgboost': {'model': xgb.XGBRegressor, 'params': get_xgb_reg_params},
        'lightgbm': {'model': lgb.LGBMRegressor, 'params': get_lgbm_reg_params},
        'catboost': {'model': cb.CatBoostRegressor, 'params': get_catboost_params}
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
        data_cleaner: Optional[Any] = None,
        tune_hyperparameters: bool = False,
        n_trials: int = 50,
        cv_folds: int = 3,
        enable_feature_engineering: bool = False,
        cv_strategy: str = 'repeated',
        show_tuning_plots: bool = False,
        output_dir: Optional[str] = None  
    ):
        self.model_name = model_name
        self.task = task.lower()
        self.random_state = random_state
        self.data_cleaner = data_cleaner
        self.tune_hyperparameters = tune_hyperparameters
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.enable_feature_engineering = enable_feature_engineering
        self.cv_strategy = cv_strategy
        self.show_tuning_plots = show_tuning_plots
        
        # Initialize output directory if provided
        self.output_dir = self._setup_output_directory(output_dir)

        # Attributes to store results
        self.best_model_info: Dict[str, Any] = {}
        self.all_results: List[Dict[str, Any]] = []
        self.X_train_final, self.X_test_final, self.y_test_final = None, None, None
        self.preprocessor: Optional[ColumnTransformer] = None # Attribute to store the fitted preprocessor

        if self.task not in ['classification', 'regression']:
            raise ValueError("Task must be 'classification' or 'regression'.")

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
        name = name.lower()
        config = MODEL_CONFIG[self.task].get(name)
        if config is None:
            raise ValueError(f"Model '{name}' is not recognized for task '{self.task}'.")
        
        model_class = config['model']
        
        # Common parameters
        params = {'random_state': self.random_state} if 'random_state' in model_class().get_params() else {}
        
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
        if self.task == 'classification':
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0)
            }
        else: # Regression
            mse = mean_squared_error(y_true, y_pred)
            return {
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2_score': r2_score(y_true, y_pred)
            }

    def _tune_with_optuna(self, model_name: str, X_train, y_train) -> Dict[str, Any]:
        """Performs hyperparameter tuning for a given model using Optuna."""
        logging.info(f"🔬 Starting hyperparameter tuning for {model_name.upper()}...")
        
        param_func = MODEL_CONFIG[self.task][model_name.lower()].get('params')
        if not param_func:
            logging.warning(f"No hyperparameter search space defined for '{model_name}'. Using default parameters.")
            return {}

        def objective(trial: optuna.Trial) -> float:
            """The objective function that Optuna will try to maximize."""
            params = param_func(trial)
            model = self._load_single_model(model_name)
            model.set_params(**params)
            
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
        
        if self.show_tuning_plots:
            self._show_tuning_insights(study, model_name)
        
        logging.info(f"✅ Tuning complete. Best parameters found: {study.best_params}")
        return study.best_params

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

    def run_pipeline(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        compare: bool = False,
        explain: bool = False,
        chosen_metric: Optional[str] = None,
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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify
        )
        
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
        model_list = self.model_name if isinstance(self.model_name, list) else [self.model_name]
        self.all_results = []
        
        for name in model_list:
            try:
                result = self._run_single_model_pipeline(name, self.X_train_final, y_train, self.X_test_final, self.y_test_final)
                self.all_results.append(result)
            except Exception as e:
                logging.error(f"Failed to process model {name}: {e}")
                self.all_results.append({'model_name': name, 'metrics': {}, 'error': str(e)})

        # 7. Identify the best model from successful runs
        successful_results = [res for res in self.all_results if 'error' not in res]
        if not successful_results:
            raise RuntimeError("No models were trained successfully. Please check your data or configuration.")

        primary_metric = 'f1_score' if self.task == 'classification' else 'r2_score'
        self.best_model_info = max(successful_results, key=lambda x: x['metrics'].get(primary_metric, -1))
        
        logging.info(f"\n--- Process Complete ---")
        logging.info(f"🏆 Best Model: {self.best_model_info['model_name'].upper()} with {primary_metric} = {self.best_model_info['metrics'][primary_metric]:.4f}")

        # 8. Display results as requested
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
        """Converts the list of model results into a sorted Pandas DataFrame."""
        if not self.all_results:
            logging.warning("No results available. Please run the pipeline first using .run_pipeline().")
            return pd.DataFrame()
        
        # Create a list of records, one for each successful model run
        records = [
            {'model': res['model_name'], **res['metrics']}
            for res in self.all_results if 'error' not in res
        ]
        df = pd.DataFrame(records).set_index('model')
        
        # Sort by the primary metric in descending order
        primary_metric = 'f1_score' if self.task == 'classification' else 'r2_score'
        is_higher_better = primary_metric in ['f1_score', 'r2_score', 'accuracy', 'precision', 'recall']
        return df.sort_values(by=primary_metric, ascending=not is_higher_better)

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

    def _get_comparison_table_html(self) -> str:
        """Generates the HTML for the model comparison table."""
        df_results = self.get_results_dataframe()
        if df_results.empty:
            return "<p>No model comparison data available.</p>"
        # Convert DataFrame to a styled HTML table
        return df_results.to_html(classes='styled-table', border=0, float_format='{:.4f}'.format)

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