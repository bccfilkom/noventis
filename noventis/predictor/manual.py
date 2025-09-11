import pandas as pd
import numpy as np
import pickle
import time
import logging
from typing import Dict, Any, List, Union
import matplotlib.pyplot as plt

# Model Selection, Ensembling, and Metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

# Model Libraries
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ManualPredictor:
    """
    Class for running robust manual machine learning pipelines.
    Supports single models, multi-model comparison, and ensembling.
    """
    def __init__(self, model_name: Union[str, List[str]], task: str, random_state: int = 42):
        self.model_name = model_name
        self.task = task.lower()
        self.random_state = random_state
        self.best_model_info = {}
        self.all_results = []  # Store all results for comparison

        if self.task not in ['classification', 'regression']:
            raise ValueError("Task must be 'classification' or 'regression'.")

    def _load_single_model(self, name: str) -> Any:
        """Load a single model instance by name."""
        name = name.lower()
        classification_models = {
            'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'decision_tree': DecisionTreeClassifier(random_state=self.random_state),
            'random_forest': RandomForestClassifier(random_state=self.random_state),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state),
            'xgboost': xgb.XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='logloss'),
            'lightgbm': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1),
            'catboost': cb.CatBoostClassifier(random_state=self.random_state, verbose=0)
        }
        regression_models = {
            'linear_regression': LinearRegression(),
            'decision_tree': DecisionTreeRegressor(random_state=self.random_state),
            'random_forest': RandomForestRegressor(random_state=self.random_state),
            'gradient_boosting': GradientBoostingRegressor(random_state=self.random_state),
            'xgboost': xgb.XGBRegressor(random_state=self.random_state),
            'lightgbm': lgb.LGBMRegressor(random_state=self.random_state, verbose=-1),
            'catboost': cb.CatBoostRegressor(random_state=self.random_state, verbose=0)
        }
        models = classification_models if self.task == 'classification' else regression_models
        model = models.get(name)
        if model is None:
            raise ValueError(f"Model '{name}' not recognized for task {self.task}.")
        return model

    def _calculate_all_metrics(self, y_true, y_pred) -> Dict:
        """Calculate all metrics based on task."""
        if self.task == 'classification':
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
        else:  # regression
            mse = mean_squared_error(y_true, y_pred)
            return {
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2_score': r2_score(y_true, y_pred)
            }

    def _run_single_model_pipeline(self, model_name: str, X_train, y_train, X_test, y_test) -> Dict:
        """Run pipeline for a single model."""
        logging.info(f"--- Processing model: {model_name.upper()} ---")
        model = self._load_single_model(model_name)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        logging.info(f"Training completed in {training_time:.2f} seconds.")
        
        predictions = model.predict(X_test)
        metrics = self._calculate_all_metrics(y_test, predictions)
        
        return {
            'model_name': model_name,
            'model_object': model,
            'metrics': metrics,
            'training_time_seconds': training_time
        }

    def _print_comparison(self):
        """Print comparison of all models with all metrics."""
        if not self.all_results:
            logging.warning("No results to compare.")
            return
        
        print("\n" + "="*80)
        print("📊 MODEL COMPARISON - ALL METRICS")
        print("="*80)
        
        if self.task == 'classification':
            print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
            print("-" * 68)
            for result in self.all_results:
                metrics = result['metrics']
                print(f"{result['model_name']:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                      f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}")
        else:  # regression
            print(f"{'Model':<20} {'MAE':<12} {'MSE':<12} {'RMSE':<12} {'R2-Score':<12}")
            print("-" * 68)
            for result in self.all_results:
                metrics = result['metrics']
                print(f"{result['model_name']:<20} {metrics['mae']:<12.4f} {metrics['mse']:<12.4f} "
                      f"{metrics['rmse']:<12.4f} {metrics['r2_score']:<12.4f}")
        
        print("="*80)

    def _create_metric_plot(self, chosen_metric: str = None):
        """Create bar plot for chosen metric."""
        if not self.all_results:
            logging.warning("No results to plot.")
            return
        
        # Set default metric
        if chosen_metric is None:
            chosen_metric = 'f1_score' if self.task == 'classification' else 'rmse'
        
        # Validate chosen metric
        available_metrics = list(self.all_results[0]['metrics'].keys())
        if chosen_metric not in available_metrics:
            logging.error(f"Metric '{chosen_metric}' not available. Choices: {available_metrics}")
            return
        
        # Extract data for plotting
        model_names = [result['model_name'].replace('_', ' ').title() for result in self.all_results]
        metric_values = [result['metrics'][chosen_metric] for result in self.all_results]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        bars = plt.bar(model_names, metric_values, color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Customize the plot
        plt.title(f'Model Comparison - {chosen_metric.upper().replace("_", " ")}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Models', fontsize=12, fontweight='bold')
        plt.ylabel(chosen_metric.upper().replace("_", " "), fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric_values)*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Highlight the best model
        best_idx = np.argmax(metric_values) if chosen_metric in ['accuracy', 'precision', 'recall', 'f1_score', 'r2_score'] else np.argmin(metric_values)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('darkorange')
        bars[best_idx].set_linewidth(2)
        
        plt.tight_layout()
        plt.show()

    def run_pipeline(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2, 
                    compare: bool = False, explain: bool = False, chosen_metric: str = None) -> Dict:
        """
        Run complete pipeline. If model_name is a list, will run comparison and ensembling.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset for training and testing
        target_column : str
            Target column name
        test_size : float, default=0.2
            Proportion of data for testing
        compare : bool, default=False
            If True, will print comparison of all models with all metrics
        explain : bool, default=False
            If True, will display bar plot for chosen metric
        chosen_metric : str, default=None
            Metric for plot. Default: 'f1_score' for classification, 'rmse' for regression
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        stratify = y if self.task == 'classification' else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify
        )

        if isinstance(self.model_name, str):
            logging.info(f"Running pipeline for single model: {self.model_name}")
            self.best_model_info = self._run_single_model_pipeline(self.model_name, X_train, y_train, X_test, y_test)
            self.all_results = [self.best_model_info]  # Store for comparison
            
            # Apply compare and explain for single model
            if compare:
                self._print_comparison()
            if explain:
                self._create_metric_plot(chosen_metric)
                
            return self.best_model_info

        elif isinstance(self.model_name, list):
            logging.info("Starting multi-model comparison and ensembling mode.")
            self.all_results = []
            
            for name in self.model_name:
                try:
                    result = self._run_single_model_pipeline(name, X_train, y_train, X_test, y_test)
                    self.all_results.append(result)
                except Exception as e:
                    logging.error(f"Failed to process model {name}: {e}")

            if not self.all_results:
                raise RuntimeError("No individual models were successfully trained.")

            logging.info("--- Processing model: ENSEMBLE ---")
            estimators = [(res['model_name'], res['model_object']) for res in self.all_results]
            
            ensemble_model = VotingClassifier(estimators=estimators, voting='soft') if self.task == 'classification' else VotingRegressor(estimators=estimators)

            try:
                start_time = time.time()
                ensemble_model.fit(X_train, y_train)
                training_time = time.time() - start_time
                logging.info(f"Ensemble training completed in {training_time:.2f} seconds.")
                
                predictions = ensemble_model.predict(X_test)
                metrics = self._calculate_all_metrics(y_test, predictions)

                self.all_results.append({
                    'model_name': 'ensemble', 'model_object': ensemble_model,
                    'metrics': metrics, 'training_time_seconds': training_time
                })
            except Exception as e:
                logging.error(f"Failed to train ensemble model: {e}")

            # Determine best model based on primary metric
            primary_metric = 'f1_score' if self.task == 'classification' else 'r2_score'
            self.best_model_info = max(self.all_results, key=lambda x: x['metrics'][primary_metric])
            
            logging.info(f"\n--- Comparison Complete ---")
            logging.info(f"🏆 Best Model: {self.best_model_info['model_name'].upper()} with {primary_metric} = {self.best_model_info['metrics'][primary_metric]:.4f}")

            # Apply compare and explain
            if compare:
                self._print_comparison()
            if explain:
                self._create_metric_plot(chosen_metric)

            return {'best_model_details': self.best_model_info, 'all_model_results': self.all_results}
        else:
            raise TypeError("model_name must be a string or list of strings.")

    def save_model(self, filepath: str):
        """Save the BEST trained model to file."""
        if not self.best_model_info:
            raise ValueError("No best model to save. Run .run_pipeline() first.")
        
        model_to_save = self.best_model_info.get('model_object')
        logging.info(f"Saving best model '{self.best_model_info['model_name']}' to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(model_to_save, f)
        logging.info("✅ Model saved successfully.")

    @staticmethod
    def load_model(filepath: str) -> Any:
        """Load model from file."""
        logging.info(f"Loading model from {filepath}...")
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logging.info("✅ Model loaded successfully.")
        return model