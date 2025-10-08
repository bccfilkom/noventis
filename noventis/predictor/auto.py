import pandas as pd
import numpy as np
import os
import pickle
from typing import Union, Optional, Dict, Any, List, Tuple
import warnings
from IPython.display import HTML
from io import BytesIO
from pathlib import Path

# FLAML imports with fallback
try:
    from flaml import AutoML as FLAMLAutoML
except ImportError:
    try:
        from flaml.automl import AutoML as FLAMLAutoML
    except ImportError:
        from flaml.automl.automl import AutoML as FLAMLAutoML

from flaml.automl.data import get_output_from_log
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, r2_score, confusion_matrix, accuracy_score, 
    precision_score, recall_score, mean_squared_error, 
    mean_absolute_error, roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from .manual import NoventisManualML

warnings.filterwarnings('ignore')


class NoventisAutoML:
    """
    Automated Machine Learning framework that combines FLAML AutoML with manual model training.
    
    Supports both classification and regression tasks with automatic model selection,
    hyperparameter tuning, comprehensive visualization, and HTML reporting.
    
    Attributes:
        COLORS (Dict[str, str]): Color scheme for visualizations
        MIN_UNIQUE_FOR_REGRESSION (int): Minimum unique values to consider regression
        UNIQUE_RATIO_THRESHOLD (float): Ratio threshold for regression detection
        MAX_FEATURES_TO_DISPLAY (int): Maximum features to show in importance plot
    """
    
    # Class constants for configuration
    COLORS = {
        'primary_blue': '#58A6FF',
        'primary_orange': '#F78166',
        'success_green': '#28A745',
        'warning_yellow': '#FFC107',
        'bg_dark': '#161B22',
        'text_light': '#C9D1D9',
        'palette': ['#58A6FF', '#F78166', '#28A745', '#FFC107', '#BB86FC', '#03DAC6', '#CF6679']
    }
    
    MIN_UNIQUE_FOR_REGRESSION = 25
    UNIQUE_RATIO_THRESHOLD = 0.05
    MAX_FEATURES_TO_DISPLAY = 20
    DEFAULT_CLASSIFICATION_METRIC = 'macro_f1'
    DEFAULT_REGRESSION_METRIC = 'r2'
    
    def __init__(
        self, 
        data: Union[str, pd.DataFrame], 
        target: str, 
        task: Optional[str] = None, 
        models: Optional[List[str]] = None, 
        explain: bool = True, 
        compare: bool = True, 
        metrics: Optional[str] = None,
        time_budget: int = 60, 
        output_dir: str = 'Noventis_Results', 
        test_size: float = 0.2, 
        random_state: int = 42
    ):
        """
        Initialize NoventisAutoML instance.
        
        Args:
            data: CSV file path or pandas DataFrame
            target: Name of the target column
            task: Task type ('classification' or 'regression'). Auto-detected if None
            models: List of specific models to train. If None, uses AutoML
            explain: Whether to generate visualizations and explanations
            compare: Whether to compare multiple models
            metrics: Primary evaluation metric. Auto-selected if None
            time_budget: Time budget in seconds for AutoML training
            output_dir: Directory to save results and visualizations
            test_size: Proportion of data for testing (0.0 to 1.0)
            random_state: Random seed for reproducibility
        """
        # Core parameters
        self.target_column = target
        self.task_type = task.lower() if task else None
        self.test_size = self._validate_test_size(test_size)
        self.random_state = random_state
        
        # Feature flags
        self.explain = explain
        self.compare = compare
        self.metrics = metrics
        
        # Training configuration
        self.time_budget = time_budget
        self.output_dir = Path(output_dir)
        self.model_list = models
        
        # Determine if using AutoML or manual models
        self.use_automl = True if (compare or self.model_list is None) else False
        
        # Initialize result containers
        self.report_id = f"automl_report_{id(self)}"
        self.flaml_model = None
        self.manual_model = None
        self.results = {}
        
        # Data attributes (initialized in setup)
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Load and prepare data
        self._load_data(data)
        self._setup_data()

    def _validate_test_size(self, test_size: float) -> float:
        """
        Validate test_size parameter.
        
        Args:
            test_size: Proposed test size
            
        Returns:
            Validated test size
            
        Raises:
            ValueError: If test_size is not in valid range
        """
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        return test_size

    def _load_data(self, data: Union[str, pd.DataFrame]) -> None:
        """
        Load data from file or DataFrame.
        
        Args:
            data: CSV file path or pandas DataFrame
            
        Raises:
            TypeError: If data is neither string path nor DataFrame
            FileNotFoundError: If CSV file doesn't exist
        """
        if isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError(f"Data file not found: {data}")
            self.df = pd.read_csv(data)
            print(f"✓ Data loaded from file: {data}")
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
            print("✓ Data loaded from DataFrame")
        else:
            raise TypeError("Input must be CSV path (str) or pandas DataFrame")
        
        print(f"  Shape: {self.df.shape}")
        print(f"  Columns: {list(self.df.columns)}")

    def _detect_task_type(self) -> str:
        """
        Automatically detect whether task is classification or regression.
        
        Uses heuristics based on target column characteristics:
        - Number of unique values
        - Ratio of unique values to total samples
        - Data type
        
        Returns:
            Task type: 'classification' or 'regression'
        """
        y = self.df[self.target_column]
        unique_values = len(y.unique())
        unique_ratio = unique_values / len(y)
        
        # Numeric check with thresholds
        if pd.api.types.is_numeric_dtype(y):
            is_regression = (
                unique_values > self.MIN_UNIQUE_FOR_REGRESSION and 
                unique_ratio >= self.UNIQUE_RATIO_THRESHOLD
            )
            return "regression" if is_regression else "classification"
        
        # Non-numeric data defaults to classification
        return "classification"

    def _setup_data(self) -> None:
        """
        Prepare data for training by splitting into train/test sets.
        
        Performs:
        - Target column validation
        - Feature-target separation
        - Task type detection (if not specified)
        - Train-test split with stratification for classification
        
        Raises:
            ValueError: If target column not found in DataFrame
        """
        # Validate target column
        if self.target_column not in self.df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found. "
                f"Available columns: {list(self.df.columns)}"
            )
        
        # Separate features and target
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        # Auto-detect task type if not specified
        if self.task_type is None:
            self.task_type = self._detect_task_type()
            print(f"✓ Task type auto-detected: {self.task_type}")
        
        # Stratify for classification to maintain class distribution
        stratify = y if self.task_type == "classification" else None
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=stratify
        )
        
        print(f"✓ Data split: Train={len(self.X_train)}, Test={len(self.X_test)}")

    def fit(self, time_budget: Optional[int] = None, metric: Optional[str] = None) -> Dict[str, Any]:
        """
        Train AutoML or manual models and generate comprehensive results.
        
        Main training workflow:
        1. Setup output directory
        2. Train models (AutoML or manual)
        3. Evaluate and store results
        4. Compare models if requested
        5. Save best model
        6. Generate visualizations and reports
        
        Args:
            time_budget: Override time budget for training (seconds)
            metric: Override evaluation metric
            
        Returns:
            Dictionary containing all results, metrics, and model information
        """

        print("Starting NoventisAutoML Training Process")
       
        
        # Use instance time_budget if not overridden
        time_budget = time_budget or self.time_budget
        
        # Convert metric to FLAML format
        flaml_metric = self._convert_metric_to_flaml(self.metrics or metric)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Train models based on configuration
        if self.use_automl:
            self._train_automl(flaml_metric, time_budget)
        else:
            self._train_manual_models()
        
        # Compare models and select best
        if self.compare:
            self._perform_model_comparison()
        else:
            self._save_single_best_model()
        
        # Generate explanations if requested
        if self.explain:
            self._generate_explanations()
        

        print("✓ Training completed successfully!")
        print(f"✓ Results saved to: {self.output_dir}")
  
        
        return self.results

    def _train_automl(self, flaml_metric: str, time_budget: int) -> None:
        """
        Train model using FLAML AutoML.
        
        Args:
            flaml_metric: Metric in FLAML format
            time_budget: Training time budget in seconds
        """
        print(f"\n[AutoML] Starting training...")
        print(f"  Metric: {flaml_metric}")
        print(f"  Time Budget: {time_budget}s")
        
        # Initialize FLAML AutoML
        self.flaml_model = FLAMLAutoML(
            task=self.task_type, 
            metric=flaml_metric, 
            seed=self.random_state, 
            verbose=2
        )
        
        # Train model
        log_path = self.output_dir / 'flaml.log'
        self.flaml_model.fit(
            X_train=self.X_train, 
            y_train=self.y_train,
            log_file_name=str(log_path), 
            time_budget=time_budget
        )
        
        # Generate predictions
        y_pred = self.flaml_model.predict(self.X_test)
        
        # Evaluate based on task type
        if self.task_type == "classification":
            metrics = self._eval_classification(self.y_test, y_pred)
            y_pred_proba = (
                self.flaml_model.predict_proba(self.X_test) 
                if hasattr(self.flaml_model, 'predict_proba') 
                else None
            )
        else:
            metrics = self._eval_regression(self.y_test, y_pred)
            y_pred_proba = None
        
        # Store results
        self.results['AutoML'] = {
            'model': self.flaml_model,
            'predictions': y_pred,
            'prediction_proba': y_pred_proba,
            'actual': self.y_test,
            'metrics': metrics,
            'task_type': self.task_type,
            'feature_importance': self._get_feature_importance(),
            'best_estimator': self.flaml_model.best_estimator,
            'best_config': self.flaml_model.best_config,
            'training_history': self._get_training_history(str(log_path))
        }
        
        print(f"✓ AutoML training completed")
        print(f"  Best estimator: {self.flaml_model.best_estimator}")

    def _train_manual_models(self) -> None:
        """
        Train models using ManualPredictor with specified model list.
        """
        print(f"\n[Manual] Training {len(self.model_list)} model(s)...")
        
        # Initialize manual predictor
        predictor = NoventisManualML(
            model_name=self.model_list, 
            task=self.task_type, 
            random_state=self.random_state,
        )
        
        # Run training pipeline
        result = predictor.fit(
            self.df, 
            target_column=self.target_column, 
            test_size=self.test_size,
            display_report=False
        )
        
        self.manual_model = predictor
        
        # Store results for each model
        for model_result in result['all_model_results']:
            if 'error' in model_result:
                print(f"  ⚠ {model_result.get('model_name', 'Unknown')} failed")
                continue
            
            model_name = model_result['model_name']
            self.results[model_name] = {
                'model': model_name,
                'predictions': model_result['predictions'],
                'prediction_proba': model_result.get('prediction_proba'),
                'actual': self.y_test,
                'metrics': model_result['metrics'],
                'task_type': self.task_type,
                'feature_importance': None,
                'best_estimator': None,
                'best_config': None,
                'training_history': None
            }
            print(f"  ✓ {model_name} completed")
        
        # Save best model
        predictor.save_model(str(self.output_dir / 'best_model.pkl'))

    def _perform_model_comparison(self) -> None:
        """
        Compare all trained models and save the best one.
        """
        print("\n[Comparison] Comparing models...")
        
        comparison_results = self.compare_models(
            output_dir=str(self.output_dir), 
            models_to_compare=self.model_list
        )
        
        self.results['model_comparison'] = comparison_results
        
        best_model_name = comparison_results['rankings'][0]['model']
        best_model_score = comparison_results['rankings'][0]['score']
        
        print(f"  Best model: {best_model_name}")
        print(f"  Score: {best_model_score:.4f}")
        
        # Save best model
        model_path = self.output_dir / 'best_model.pkl'
        
        if self.use_automl:
            automl_score = self.results['AutoML']['metrics'].get(self.metrics, 0)
            if best_model_score > automl_score:
                # Manual model is better
                internal_key = best_model_name.replace(' ', '_').lower()
                self.manual_model.save_model(str(model_path))
                self.results[internal_key]['model_path'] = str(model_path)
            else:
                # AutoML is better
                self._save_model(self.flaml_model, str(model_path))
                self.results['AutoML']['model_path'] = str(model_path)
        else:
            # Only manual models were trained
            self.manual_model.save_model(str(model_path))

    def _save_single_best_model(self) -> None:
        """
        Save the best model when comparison is disabled.
        """
        model_path = self.output_dir / 'best_model.pkl'
        
        if self.use_automl:
            self._save_model(self.flaml_model, str(model_path))
            self.results['AutoML']['model_path'] = str(model_path)
        else:
            # Still need comparison to determine best manual model
            comparison_results = self.compare_models(
                output_dir=str(self.output_dir), 
                models_to_compare=self.model_list
            )
            self.results['model_comparison'] = comparison_results
            self.manual_model.save_model(str(model_path))

    def _generate_explanations(self) -> None:
        """
        Generate visualizations and summary reports.
        """
        print("\n[Explanation] Generating visualizations...")
        
        viz_paths = self._generate_visualizations(self.results, str(self.output_dir))
        self.results['visualization_paths'] = viz_paths
        
        print(f"  ✓ Generated {len(viz_paths)} visualization(s)")
        
        self._generate_model_summary(self.results, str(self.output_dir))
        print("  ✓ Summary report created")

    def compare_models(
        self, 
        models_to_compare: Optional[List[str]] = None, 
        output_dir: str = "Noventis_results"
    ) -> Dict[str, Any]:
        """
        Compare multiple models and rank by performance.
        
        Args:
            models_to_compare: List of model names to compare. Uses defaults if None
            output_dir: Directory to save comparison results
            
        Returns:
            Dictionary with rankings, best model, and primary metric
        """
        os.makedirs(output_dir, exist_ok=True)
        all_results = {}
        
        # Use default model list if not specified
        if models_to_compare is None:
            models_to_compare = self._get_default_models()
            
            # Include AutoML if available
            if 'AutoML' in self.results:
                all_results['AutoML'] = {
                    'metrics': self.results['AutoML']['metrics'],
                    'model_name': 'AutoML',
                    'best_estimator': 'AutoML'
                }
        
        # Train models if not already trained
        if self.manual_model is None and models_to_compare:
            self._train_comparison_models(models_to_compare)
        
        # Collect all model results
        for model_key in self.results:
            if model_key not in ['model_comparison', 'visualization_paths']:
                all_results[model_key] = self.results[model_key]
        
        # Rank models by performance
        ranked_results = self._rank_models(all_results)
        
        # Generate comparison visualizations if enabled
        if self.compare:
            self._visualize_model_comparison(ranked_results, all_results, output_dir)
            self._generate_comparison_report(ranked_results, all_results, output_dir)
        
        return ranked_results

    def _get_default_models(self) -> List[str]:
        """
        Get default model list based on task type.
        
        Returns:
            List of default model names
        """
        if self.task_type == 'classification':
            return [
                'logistic_regression', 'random_forest', 'xgboost', 
                'decision_tree', 'lightgbm', 'catboost', 'gradient_boosting'
            ]
        else:  # regression
            return [
                'linear_regression', 'random_forest', 'xgboost', 
                'gradient_boosting', 'lightgbm', 'catboost'
            ]

    def _train_comparison_models(self, models_to_compare: List[str]) -> None:
        """
        Train models specifically for comparison.
        
        Args:
            models_to_compare: List of model names to train
        """
        predictor = NoventisManualML(
            model_name=models_to_compare, 
            task=self.task_type, 
            random_state=self.random_state
        )
        
        result = predictor.fit(
            self.df, 
            target_column=self.target_column, 
            test_size=self.test_size,
            display_report=False
        )
        
        self.manual_model = predictor
        
        # Store results
        for model_result in result['all_model_results']:
            if 'error' in model_result:
                continue
            
            model_name = model_result['model_name'].replace(' ', '_').lower()
            self.results[model_name] = {
                'model': model_name,
                'predictions': model_result['predictions'],
                'prediction_proba': model_result.get('prediction_proba'),
                'actual': self.y_test,
                'metrics': model_result['metrics'],
                'task_type': self.task_type,
                'feature_importance': None,
                'best_estimator': None,
                'best_config': None,
                'training_history': None
            }

    def _eval_classification(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metric names and values
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(
                y_true, y_pred, average='weighted', zero_division=0
            ),
            'recall': recall_score(
                y_true, y_pred, average='weighted', zero_division=0
            ),
            'f1_score': f1_score(
                y_true, y_pred, average='macro', zero_division=0
            )
        }

    def _eval_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metric names and values
        """
        mse = mean_squared_error(y_true, y_pred)
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2_score': r2_score(y_true, y_pred)
        }

    def _convert_metric_to_flaml(self, metric: Optional[str]) -> str:
        """
        Convert metric name to FLAML format.
        
        Args:
            metric: Metric name
            
        Returns:
            FLAML-compatible metric name
        """
        if metric is None:
            return (
                self.DEFAULT_CLASSIFICATION_METRIC 
                if self.task_type == "classification" 
                else self.DEFAULT_REGRESSION_METRIC
            )
        
        # Map custom metrics to FLAML equivalents
        metric_map = {
            'f1_score': 'macro_f1',
            'r2_score': 'r2'
        }
        
        return metric_map.get(metric, metric)

    def _get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Extract feature importance from trained AutoML model.
        
        Returns:
            DataFrame with features and importance scores, or None if unavailable
        """
        if not self.use_automl or self.flaml_model is None:
            return None
        
        try:
            if hasattr(self.flaml_model, 'feature_importances_'):
                return pd.DataFrame({
                    'feature': self.X_train.columns,
                    'importance': self.flaml_model.feature_importances_
                }).sort_values('importance', ascending=False)
        except Exception as e:
            print(f"  Warning: Could not extract feature importance: {e}")
        
        return None

    def _get_training_history(self, log_file: str) -> Optional[pd.DataFrame]:
        """
        Parse FLAML training log to extract history.
        
        Args:
            log_file: Path to FLAML log file
            
        Returns:
            DataFrame with training time and loss history, or None if unavailable
        """
        if not self.use_automl:
            return None
        
        try:
            if os.path.exists(log_file):
                time_h, loss_h, _, _, _ = get_output_from_log(
                    filename=log_file, 
                    time_budget=float('inf')
                )
                return pd.DataFrame({
                    'time_seconds': time_h,
                    'best_validation_loss': loss_h
                })
        except Exception as e:
            print(f"  Warning: Could not parse training history: {e}")
        
        return None

    def _save_model(self, model: Any, path: str) -> None:
        """
        Save model to disk using pickle.
        
        Args:
            model: Model object to save
            path: File path for saving
        """
        try:
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            print(f"  ✓ Model saved: {path}")
        except Exception as e:
            print(f"  ✗ Error saving model: {e}")

    def _set_plot_style(self) -> None:
        """
        Configure matplotlib style for dark-themed visualizations.
        """
        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.facecolor': self.COLORS['bg_dark'],
            'axes.facecolor': self.COLORS['bg_dark'],
            'text.color': self.COLORS['text_light'],
            'axes.labelcolor': self.COLORS['text_light'],
            'xtick.color': self.COLORS['text_light'],
            'ytick.color': self.COLORS['text_light'],
            'grid.alpha': 0.2,
            'grid.color': self.COLORS['text_light']
        })

    def _generate_visualizations(
        self, 
        results: Dict[str, Any], 
        output_dir: str
    ) -> List[str]:
        """
        Generate all visualization plots based on task type.
        
        Args:
            results: Dictionary of model results
            output_dir: Directory to save plots
            
        Returns:
            List of file paths to generated plots
        """
        paths = []
        self._set_plot_style()
        
        # Determine best model result to visualize
        if self.use_automl and not self.compare:
            best_model_res = results.get('AutoML', {})
        elif 'model_comparison' in results:
            best_model_name = results['model_comparison']['rankings'][0]['model']
            best_model_res = self.results.get(
            best_model_name.replace(' ', '_').lower(),
            self.results.get('AutoML', {}))
        else:
            return paths  # No results available
        
        try:
            # Feature importance plot (AutoML only)
            if self.use_automl and best_model_res.get('feature_importance') is not None:
                path = self._plot_feature_importance(
                    best_model_res['feature_importance'], 
                    output_dir
                )
                if path:
                    paths.append(path)
            
            # Training history plot (AutoML only)
            if self.use_automl and best_model_res.get('training_history') is not None:
                path = self._plot_training_history(
                    best_model_res['training_history'], 
                    output_dir
                )
                if path:
                    paths.append(path)
            
            # Task-specific plots
            if self.task_type == 'classification':
                paths.extend(
                    self._generate_classification_plots(best_model_res, output_dir)
                )
            else:
                paths.extend(
                    self._generate_regression_plots(best_model_res, output_dir)
                )
        
        except Exception as e:
            print(f"  ✗ Error creating visualizations: {e}")
        
        return paths

    def _plot_feature_importance(
        self, 
        fi_df: pd.DataFrame, 
        output_dir: str
    ) -> Optional[str]:
        """
        Create horizontal bar plot of feature importance.
        
        Args:
            fi_df: DataFrame with 'feature' and 'importance' columns
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot file, or None if failed
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            top_features = fi_df.head(self.MAX_FEATURES_TO_DISPLAY)
            
            ax.barh(
                top_features['feature'], 
                top_features['importance'],
                color=self.COLORS['primary_blue'],
                edgecolor=self.COLORS['primary_orange'],
                linewidth=1.5
            )
            
            ax.set_xlabel('Importance Score', fontsize=12)
            ax.set_ylabel('Features', fontsize=12)
            ax.set_title(
                f'Top {self.MAX_FEATURES_TO_DISPLAY} Feature Importance',
                fontsize=16,
                fontweight='bold',
                color=self.COLORS['primary_orange']
            )
            ax.grid(axis='x', alpha=0.2)
            
            plt.tight_layout()
            path = os.path.join(output_dir, 'feature_importance.png')
            plt.savefig(path, dpi=300, bbox_inches='tight', 
                       facecolor=self.COLORS['bg_dark'])
            plt.close(fig)
            
            return path
        
        except Exception as e:
            print(f"  Warning: Failed to plot feature importance: {e}")
            plt.close('all')
            return None

    def _plot_training_history(
        self, 
        history: pd.DataFrame, 
        output_dir: str
    ) -> Optional[str]:
        """
        Create line plot of AutoML training progress over time.
        
        Args:
            history: DataFrame with 'time_seconds' and 'best_validation_loss'
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot file, or None if failed
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.plot(
                history['time_seconds'],
                history['best_validation_loss'],
                marker='o',
                linestyle='-',
                color=self.COLORS['primary_blue'],
                linewidth=2.5,
                markersize=6
            )
            
            ax.fill_between(
                history['time_seconds'],
                history['best_validation_loss'],
                alpha=0.3,
                color=self.COLORS['primary_blue']
            )
            
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Best Validation Loss', fontsize=12)
            ax.set_title(
                'AutoML Training Progress',
                fontsize=16,
                fontweight='bold',
                color=self.COLORS['primary_orange']
            )
            ax.grid(True, alpha=0.2)
            
            plt.tight_layout()
            path = os.path.join(output_dir, 'training_history.png')
            plt.savefig(path, dpi=300, bbox_inches='tight',
                       facecolor=self.COLORS['bg_dark'])
            plt.close(fig)
            
            return path
        
        except Exception as e:
            print(f"  Warning: Failed to plot training history: {e}")
            plt.close('all')
            return None

    def _generate_classification_plots(
        self, 
        results: Dict[str, Any], 
        output_dir: str
    ) -> List[str]:
        """
        Generate comprehensive classification visualizations.
        
        Creates:
        - Confusion matrix (normalized)
        - Metrics bar chart
        - ROC and PR curves (binary classification)
        - Class distribution comparison
        
        Args:
            results: Model results dictionary
            output_dir: Directory to save plots
            
        Returns:
            List of paths to saved plot files
        """
        paths = []
        
        # 1. Confusion Matrix
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            cm = confusion_matrix(results['actual'], results['predictions'])
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                ax=ax,
                cbar_kws={'label': 'Percentage'}
            )
            
            ax.set_title(
                'Confusion Matrix (Normalized)',
                fontsize=16,
                fontweight='bold',
                color=self.COLORS['primary_orange']
            )
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('Actual', fontsize=12)
            
            plt.tight_layout()
            path = os.path.join(output_dir, 'confusion_matrix.png')
            plt.savefig(path, dpi=300, bbox_inches='tight',
                       facecolor=self.COLORS['bg_dark'])
            paths.append(path)
            plt.close(fig)
        except Exception as e:
            print(f"  Warning: Failed to create confusion matrix: {e}")
            plt.close('all')
        
        # 2. Classification Metrics Bar Chart
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics = results['metrics']
            
            bars = ax.bar(
                metrics.keys(),
                metrics.values(),
                color=self.COLORS['palette'][:len(metrics)],
                edgecolor=self.COLORS['primary_orange'],
                linewidth=2
            )
            
            ax.set_ylabel('Score', fontsize=12)
            ax.set_ylim(0, 1.1)
            ax.set_title(
                'Classification Metrics',
                fontsize=16,
                fontweight='bold',
                color=self.COLORS['primary_orange']
            )
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics.values()):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{value:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )
            
            ax.grid(axis='y', alpha=0.2)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            path = os.path.join(output_dir, 'classification_metrics.png')
            plt.savefig(path, dpi=300, bbox_inches='tight',
                       facecolor=self.COLORS['bg_dark'])
            paths.append(path)
            plt.close(fig)
        except Exception as e:
            print(f"  Warning: Failed to create metrics chart: {e}")
            plt.close('all')
        
        # 3. ROC and PR Curves (binary classification only)
        if (results.get('prediction_proba') is not None and 
            len(np.unique(results['actual'])) == 2):
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                y_true = results['actual']
                y_proba = results['prediction_proba'][:, 1]
                
                # ROC Curve
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)
                
                ax1.plot(
                    fpr, tpr,
                    color=self.COLORS['primary_blue'],
                    lw=3,
                    label=f'ROC (AUC = {roc_auc:.3f})'
                )
                ax1.plot(
                    [0, 1], [0, 1],
                    color=self.COLORS['primary_orange'],
                    lw=2,
                    linestyle='--',
                    label='Random'
                )
                ax1.fill_between(fpr, tpr, alpha=0.3, color=self.COLORS['primary_blue'])
                
                ax1.set_xlabel('False Positive Rate', fontsize=12)
                ax1.set_ylabel('True Positive Rate', fontsize=12)
                ax1.set_title(
                    'ROC Curve',
                    fontsize=14,
                    fontweight='bold',
                    color=self.COLORS['primary_orange']
                )
                ax1.legend(loc='lower right')
                ax1.grid(alpha=0.2)
                
                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_true, y_proba)
                
                ax2.plot(
                    recall, precision,
                    color=self.COLORS['success_green'],
                    lw=3,
                    label='PR Curve'
                )
                ax2.fill_between(
                    recall, precision,
                    alpha=0.3,
                    color=self.COLORS['success_green']
                )
                
                ax2.set_xlabel('Recall', fontsize=12)
                ax2.set_ylabel('Precision', fontsize=12)
                ax2.set_title(
                    'Precision-Recall Curve',
                    fontsize=14,
                    fontweight='bold',
                    color=self.COLORS['primary_orange']
                )
                ax2.legend(loc='lower left')
                ax2.grid(alpha=0.2)
                
                plt.tight_layout()
                path = os.path.join(output_dir, 'roc_pr_curves.png')
                plt.savefig(path, dpi=300, bbox_inches='tight',
                           facecolor=self.COLORS['bg_dark'])
                paths.append(path)
                plt.close(fig)
            except Exception as e:
                print(f"  Warning: Failed to create ROC/PR curves: {e}")
                plt.close('all')
        
        # 4. Class Distribution Comparison
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Actual distribution
            class_counts = pd.Series(results['actual']).value_counts()
            axes[0].pie(
                class_counts.values,
                labels=class_counts.index,
                autopct='%1.1f%%',
                colors=self.COLORS['palette'],
                startangle=90,
                textprops={'fontsize': 11, 'weight': 'bold'}
            )
            axes[0].set_title(
                'Actual Distribution',
                fontsize=14,
                fontweight='bold',
                color=self.COLORS['primary_orange']
            )
            
            # Predicted distribution
            pred_counts = pd.Series(results['predictions']).value_counts()
            axes[1].pie(
                pred_counts.values,
                labels=pred_counts.index,
                autopct='%1.1f%%',
                colors=self.COLORS['palette'],
                startangle=90,
                textprops={'fontsize': 11, 'weight': 'bold'}
            )
            axes[1].set_title(
                'Predicted Distribution',
                fontsize=14,
                fontweight='bold',
                color=self.COLORS['primary_orange']
            )
            
            plt.tight_layout()
            path = os.path.join(output_dir, 'class_distribution.png')
            plt.savefig(path, dpi=300, bbox_inches='tight',
                       facecolor=self.COLORS['bg_dark'])
            paths.append(path)
            plt.close(fig)
        except Exception as e:
            print(f"  Warning: Failed to create class distribution: {e}")
            plt.close('all')
        
        return paths

    def _generate_regression_plots(
        self, 
        results: Dict[str, Any], 
        output_dir: str
    ) -> List[str]:
        """
        Generate comprehensive regression visualizations.
        
        Creates:
        - Predictions vs Actual scatter plot
        - Residuals plot
        - Residuals distribution and Q-Q plot
        - Error distribution over samples
        
        Args:
            results: Model results dictionary
            output_dir: Directory to save plots
            
        Returns:
            List of paths to saved plot files
        """
        paths = []
        
        # 1. Predictions vs Actual
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            ax.scatter(
                results['actual'],
                results['predictions'],
                alpha=0.6,
                c=self.COLORS['primary_blue'],
                edgecolors=self.COLORS['primary_orange'],
                s=80,
                linewidth=1.5
            )
            
            # Perfect prediction line
            min_val = min(min(results['actual']), min(results['predictions']))
            max_val = max(max(results['actual']), max(results['predictions']))
            perfect_line = np.linspace(min_val, max_val, 100)
            
            ax.plot(
                perfect_line, perfect_line,
                color=self.COLORS['primary_orange'],
                linestyle='--',
                linewidth=3,
                label='Perfect Prediction'
            )
            
            ax.set_xlabel('Actual Values', fontsize=12)
            ax.set_ylabel('Predicted Values', fontsize=12)
            ax.set_title(
                'Predictions vs Actual',
                fontsize=16,
                fontweight='bold',
                color=self.COLORS['primary_orange']
            )
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.2)
            
            # Add R² annotation
            r2 = results['metrics'].get('r2_score', 0)
            ax.text(
                0.05, 0.95,
                f'R² = {r2:.4f}',
                transform=ax.transAxes,
                bbox=dict(
                    boxstyle='round',
                    facecolor=self.COLORS['bg_dark'],
                    alpha=0.8,
                    edgecolor=self.COLORS['primary_blue']
                ),
                fontsize=13,
                fontweight='bold',
                verticalalignment='top'
            )
            
            plt.tight_layout()
            path = os.path.join(output_dir, 'predictions_vs_actual.png')
            plt.savefig(path, dpi=300, bbox_inches='tight',
                       facecolor=self.COLORS['bg_dark'])
            paths.append(path)
            plt.close(fig)
        except Exception as e:
            print(f"  Warning: Failed to create predictions plot: {e}")
            plt.close('all')
        
        # 2. Residuals Plot
        try:
            residuals = np.array(results['actual']) - np.array(results['predictions'])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.scatter(
                results['predictions'],
                residuals,
                alpha=0.6,
                c=self.COLORS['primary_blue'],
                edgecolors=self.COLORS['primary_orange'],
                s=80,
                linewidth=1.5
            )
            
            ax.axhline(
                y=0,
                color=self.COLORS['success_green'],
                linestyle='--',
                linewidth=3
            )
            
            ax.set_xlabel('Predicted Values', fontsize=12)
            ax.set_ylabel('Residuals', fontsize=12)
            ax.set_title(
                'Residuals Plot',
                fontsize=16,
                fontweight='bold',
                color=self.COLORS['primary_orange']
            )
            ax.grid(True, alpha=0.2)
            
            plt.tight_layout()
            path = os.path.join(output_dir, 'residuals_plot.png')
            plt.savefig(path, dpi=300, bbox_inches='tight',
                       facecolor=self.COLORS['bg_dark'])
            paths.append(path)
            plt.close(fig)
        except Exception as e:
            print(f"  Warning: Failed to create residuals plot: {e}")
            plt.close('all')
        
        # 3. Regression Analysis (4-panel)
        try:
            residuals = np.array(results['actual']) - np.array(results['predictions'])
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Panel 1: Residuals Distribution
            axes[0, 0].hist(
                residuals,
                bins=30,
                color=self.COLORS['primary_blue'],
                edgecolor=self.COLORS['primary_orange'],
                alpha=0.7,
                linewidth=1.5
            )
            axes[0, 0].set_xlabel('Residuals', fontsize=11)
            axes[0, 0].set_ylabel('Frequency', fontsize=11)
            axes[0, 0].set_title(
                'Residuals Distribution',
                fontsize=13,
                fontweight='bold',
                color=self.COLORS['primary_orange']
            )
            axes[0, 0].grid(alpha=0.2)
            
            # Panel 2: Actual vs Predicted Distribution
            axes[0, 1].hist(
                results['actual'],
                bins=30,
                color=self.COLORS['success_green'],
                edgecolor=self.COLORS['primary_orange'],
                alpha=0.7,
                label='Actual',
                linewidth=1.5
            )
            axes[0, 1].hist(
                results['predictions'],
                bins=30,
                color=self.COLORS['primary_blue'],
                edgecolor=self.COLORS['primary_orange'],
                alpha=0.5,
                label='Predicted',
                linewidth=1.5
            )
            axes[0, 1].set_xlabel('Values', fontsize=11)
            axes[0, 1].set_ylabel('Frequency', fontsize=11)
            axes[0, 1].set_title(
                'Actual vs Predicted Distribution',
                fontsize=13,
                fontweight='bold',
                color=self.COLORS['primary_orange']
            )
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.2)
            
            # Panel 3: Q-Q Plot
            stats.probplot(residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].get_lines()[0].set_markerfacecolor(self.COLORS['primary_blue'])
            axes[1, 0].get_lines()[0].set_markeredgecolor(self.COLORS['primary_orange'])
            axes[1, 0].get_lines()[0].set_markersize(6)
            axes[1, 0].get_lines()[1].set_color(self.COLORS['success_green'])
            axes[1, 0].get_lines()[1].set_linewidth(2)
            axes[1, 0].set_title(
                'Q-Q Plot',
                fontsize=13,
                fontweight='bold',
                color=self.COLORS['primary_orange']
            )
            axes[1, 0].grid(alpha=0.2)
            
            # Panel 4: Metrics Bar Chart
            metrics = results['metrics']
            bars = axes[1, 1].bar(
                metrics.keys(),
                metrics.values(),
                color=self.COLORS['palette'][:len(metrics)],
                edgecolor=self.COLORS['primary_orange'],
                linewidth=2
            )
            axes[1, 1].set_ylabel('Score', fontsize=11)
            axes[1, 1].set_title(
                'Regression Metrics',
                fontsize=13,
                fontweight='bold',
                color=self.COLORS['primary_orange']
            )
            
            # Add value labels
            for bar, value in zip(bars, metrics.values()):
                height = bar.get_height()
                axes[1, 1].text(
                    bar.get_x() + bar.get_width() / 2,
                    height + max(metrics.values()) * 0.02,
                    f'{value:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )
            
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(axis='y', alpha=0.2)
            
            plt.tight_layout()
            path = os.path.join(output_dir, 'regression_analysis.png')
            plt.savefig(path, dpi=300, bbox_inches='tight',
                       facecolor=self.COLORS['bg_dark'])
            paths.append(path)
            plt.close(fig)
        except Exception as e:
            print(f"  Warning: Failed to create regression analysis: {e}")
            plt.close('all')
        
        # 4. Error Distribution
        try:
            residuals = np.array(results['actual']) - np.array(results['predictions'])
            error_percent = np.abs(residuals / results['actual']) * 100
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.scatter(
                range(len(error_percent)),
                error_percent,
                alpha=0.6,
                c=self.COLORS['primary_blue'],
                edgecolors=self.COLORS['primary_orange'],
                s=60,
                linewidth=1.5
            )
            
            mean_error = np.mean(error_percent)
            ax.axhline(
                y=mean_error,
                color=self.COLORS['success_green'],
                linestyle='--',
                linewidth=2,
                label=f'Mean Error: {mean_error:.2f}%'
            )
            
            ax.set_xlabel('Sample Index', fontsize=12)
            ax.set_ylabel('Absolute Percentage Error', fontsize=12)
            ax.set_title(
                'Prediction Error Distribution',
                fontsize=16,
                fontweight='bold',
                color=self.COLORS['primary_orange']
            )
            ax.legend()
            ax.grid(True, alpha=0.2)
            
            plt.tight_layout()
            path = os.path.join(output_dir, 'error_distribution.png')
            plt.savefig(path, dpi=300, bbox_inches='tight',
                       facecolor=self.COLORS['bg_dark'])
            paths.append(path)
            plt.close(fig)
        except Exception as e:
            print(f"  Warning: Failed to create error distribution: {e}")
            plt.close('all')
        
        return paths

    def _generate_model_summary(
        self, 
        results: Dict[str, Any], 
        output_dir: str
    ) -> None:
        """
        Generate text summary of model performance and configuration.
        
        Args:
            results: Dictionary of model results
            output_dir: Directory to save summary file
        """
        try:
            summary_path = os.path.join(output_dir, 'model_summary.txt')
            
            # Determine best model
            if 'model_comparison' in results:
                model_name = results['model_comparison']['rankings'][0]['model']
                model_key = model_name.replace(' ', '_')
            else:
                model_name = 'AutoML'
                model_key = 'AutoML'
            
            best_model_res = results.get(model_key, {})
            
            with open(summary_path, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("         NOVENTIS AutoML - MODEL SUMMARY\n")
                f.write("=" * 70 + "\n\n")
                
                f.write(f"Task Type: {self.task_type.title()}\n")
                f.write(f"Best Model: {model_name}\n")
                f.write(f"Target Column: {self.target_column}\n")
                f.write(f"Dataset Shape: {self.df.shape}\n")
                f.write(f"Train/Test Split: {len(self.X_train)}/{len(self.X_test)}\n\n")
                
                f.write("PERFORMANCE METRICS:\n")
                f.write("-" * 40 + "\n")
                
                if 'metrics' in best_model_res:
                    for metric, value in best_model_res['metrics'].items():
                        f.write(f"  {metric.replace('_', ' ').title()}: {value:.4f}\n")
                
                # Feature importance section
                if best_model_res.get('feature_importance') is not None:
                    f.write(f"\nTOP 10 IMPORTANT FEATURES:\n")
                    f.write("-" * 40 + "\n")
                    fi_df = best_model_res['feature_importance'].head(10)
                    for idx, row in fi_df.iterrows():
                        f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
                
                # Model configuration section
                if best_model_res.get('best_config'):
                    f.write(f"\nMODEL CONFIGURATION:\n")
                    f.write("-" * 40 + "\n")
                    for key, value in best_model_res['best_config'].items():
                        f.write(f"  {key}: {value}\n")
            
            print(f"  ✓ Summary saved: {summary_path}")
        
        except Exception as e:
            print(f"  ✗ Error creating summary: {e}")

    def _rank_models(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rank all models by primary metric performance.
        
        Args:
            results: Dictionary of all model results
            
        Returns:
            Dictionary with rankings, best model, and primary metric
        """
        rankings = []
        
        # Determine primary metric
        if self.metrics is None:
            primary_metric = (
                'f1_score' if self.task_type == "classification" 
                else 'r2_score'
            )
        else:
            primary_metric = self.metrics
        
        self.metrics = primary_metric
        
        # Collect scores from all models
        for name, res in results.items():
            if 'error' in res or 'metrics' not in res:
                continue
            
            score = res['metrics'].get(primary_metric, -1)
            model_display_name = name.replace('_', ' ').title()
            
            rankings.append({
                'model': model_display_name,
                'score': score,
                'metrics': res['metrics']
            })
        
        # Sort by score (descending - higher is better)
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'rankings': rankings,
            'best_model': rankings[0]['model'] if rankings else None,
            'primary_metric': primary_metric
        }

    def _visualize_model_comparison(
        self, 
        ranked_results: Dict[str, Any], 
        all_results: Dict[str, Any], 
        output_dir: str
    ) -> None:
        """
        Create visualization comparing all trained models.
        
        Args:
            ranked_results: Dictionary with model rankings
            all_results: Dictionary of all model results
            output_dir: Directory to save plots
        """
        if not ranked_results['rankings']:
            print("  Warning: No rankings available for comparison")
            return
        
        try:
            self._set_plot_style()
            df_ranks = pd.DataFrame(ranked_results['rankings'])
            
            # Model comparison bar chart
            fig, ax = plt.subplots(figsize=(14, 8))
            
            bars = ax.barh(
                df_ranks['model'],
                df_ranks['score'],
                color=self.COLORS['palette'][:len(df_ranks)],
                edgecolor=self.COLORS['primary_orange'],
                linewidth=2
            )
            
            ax.set_xlabel(
                ranked_results['primary_metric'].replace('_', ' ').title(),
                fontsize=12
            )
            ax.set_ylabel('Model', fontsize=12)
            ax.set_title(
                'Model Performance Comparison',
                fontsize=16,
                fontweight='bold',
                color=self.COLORS['primary_orange']
            )
            
            # Add score labels
            for bar, score in zip(bars, df_ranks['score']):
                ax.text(
                    bar.get_width() + 0.001,
                    bar.get_y() + bar.get_height() / 2,
                    f'{score:.4f}',
                    va='center',
                    ha='left',
                    fontweight='bold',
                    fontsize=11
                )
            
            ax.grid(axis='x', alpha=0.2)
            plt.tight_layout()
            
            path = os.path.join(output_dir, 'model_comparison.png')
            plt.savefig(path, dpi=300, bbox_inches='tight',
                       facecolor=self.COLORS['bg_dark'])
            plt.close(fig)
            
            # Create metrics heatmap
            self._create_metrics_heatmap(df_ranks, output_dir)
        
        except Exception as e:
            print(f"  ✗ Error creating comparison visualization: {e}")
            plt.close('all')

    def _create_metrics_heatmap(
        self, 
        df_ranks: pd.DataFrame, 
        output_dir: str
    ) -> None:
        """
        Create heatmap showing all metrics across all models.
        
        Args:
            df_ranks: DataFrame with model rankings and metrics
            output_dir: Directory to save heatmap
        """
        try:
            # Extract metrics for all models
            all_metrics = {
                ranking['model']: ranking['metrics'] 
                for ranking in df_ranks.to_dict('records')
            }
            
            if not all_metrics:
                return
            
            metrics_df = pd.DataFrame(all_metrics).T
            
            # Only create heatmap if multiple metrics exist
            if len(metrics_df.columns) > 1:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                sns.heatmap(
                    metrics_df,
                    annot=True,
                    fmt='.3f',
                    cmap='RdYlGn',
                    cbar_kws={'label': 'Score'},
                    ax=ax,
                    linewidths=0.5
                )
                
                ax.set_title(
                    'Model Performance Heatmap',
                    fontsize=16,
                    fontweight='bold',
                    color=self.COLORS['primary_orange']
                )
                ax.set_xlabel('Metrics', fontsize=12)
                ax.set_ylabel('Models', fontsize=12)
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                path = os.path.join(output_dir, 'metrics_heatmap.png')
                plt.savefig(path, dpi=300, bbox_inches='tight',
                           facecolor=self.COLORS['bg_dark'])
                plt.close(fig)
        
        except Exception as e:
            print(f"  Warning: Failed to create metrics heatmap: {e}")
            plt.close('all')

    def _generate_comparison_report(
        self, 
        ranked_results: Dict[str, Any], 
        all_results: Dict[str, Any], 
        output_dir: str
    ) -> None:
        """
        Generate text report comparing all models.
        
        Args:
            ranked_results: Dictionary with model rankings
            all_results: Dictionary of all model results
            output_dir: Directory to save report
        """
        try:
            report_path = os.path.join(output_dir, 'model_comparison_report.txt')
            
            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("         NOVENTIS AutoML - MODEL COMPARISON REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Task Type: {self.task_type.title()}\n")
                f.write(f"Primary Metric: {ranked_results['primary_metric']}\n")
                f.write(f"Best Model: {ranked_results['best_model']}\n\n")
                
                f.write("MODEL RANKINGS:\n")
                f.write("-" * 60 + "\n")
                
                for i, ranking in enumerate(ranked_results['rankings'], 1):
                    f.write(f"\n{i}. {ranking['model']}\n")
                    f.write(f"   Primary Score ({ranked_results['primary_metric']}): {ranking['score']:.4f}\n")
                    f.write("   All Metrics:\n")
                    for metric, value in ranking['metrics'].items():
                        f.write(f"     • {metric.replace('_', ' ').title()}: {value:.4f}\n")
            
            print(f"  ✓ Comparison report saved: {report_path}")
        
        except Exception as e:
            print(f"  ✗ Error creating comparison report: {e}")

    def generate_html_report(self, report_height: int = 800) -> HTML:
        """
        Generate interactive HTML report with multiple tabs.
        
        Creates comprehensive dashboard with:
        - Overview of dataset and configuration
        - Performance metrics and evaluation
        - Sample predictions
        - Visualizations gallery
        - Model comparison (if enabled)
        
        Args:
            report_height: Height of report frame in pixels
            
        Returns:
            IPython HTML object for display in notebooks
        """
        if not hasattr(self, 'results') or not self.results:
            return HTML("<p>No results available. Run fit() first.</p>")
        
        # Determine best model
        if 'model_comparison' in self.results:
            best_model_name = self.results['model_comparison']['rankings'][0]['model']
        else:
            best_model_name = 'AutoML'
        
        best_model_res = self.results.get(
            best_model_name.replace(' ', '_'),
            self.results.get('AutoML', {})
        )
        
        # Generate content sections
        overview_html = self._generate_overview_section()
        metrics_html = self._generate_metrics_section(best_model_res, best_model_name)
        predictions_html = self._generate_predictions_section(best_model_res)
        visualizations_html = self._generate_visualizations_section()
        
        # Configure tabs
        tabs_config = [
            {'id': 'overview', 'title': 'Overview', 'content': overview_html},
            {'id': 'metrics', 'title': 'Performance', 'content': metrics_html},
            {'id': 'predictions', 'title': 'Predictions', 'content': predictions_html},
            {'id': 'visualizations', 'title': 'Visualizations', 'content': visualizations_html},
        ]
        
        # Add comparison tab if enabled
        if self.compare:
            comparison_html = self._generate_comparison_section()
            tabs_config.append({
                'id': 'comparison', 
                'title': 'Comparison', 
                'content': comparison_html
            })
        
        # Build navigation and content
        navbar_html = ""
        main_content_html = ""
        
        for i, tab in enumerate(tabs_config):
            active_class = 'active' if i == 0 else ''
            navbar_html += (
                f'<button class="nav-btn {active_class}" '
                f'onclick="showTab(event, \'{tab["id"]}\', \'{self.report_id}\')">'
                f'{tab["title"]}</button>'
            )
            main_content_html += (
                f'<section id="{tab["id"]}-{self.report_id}" '
                f'class="content-section {active_class}">'
                f'<h2>{tab["title"]}</h2>{tab["content"]}</section>'
            )
        
        # Complete HTML template
        html_template = self._build_html_template(
            navbar_html, 
            main_content_html, 
            report_height
        )
        
        return HTML(html_template)

    def _build_html_template(
        self, 
        navbar_html: str, 
        main_content_html: str, 
        report_height: int
    ) -> str:
        """
        Build complete HTML template for report.
        
        Args:
            navbar_html: Navigation bar HTML
            main_content_html: Main content HTML
            report_height: Height of report frame
            
        Returns:
            Complete HTML string
        """
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Noventis AutoML Report</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Exo+2:wght@600;800&display=swap');
                
                :root {{
                    --bg-dark-1: #0D1117;
                    --bg-dark-2: #161B22;
                    --bg-dark-3: #010409;
                    --border-color: #30363D;
                    --text-light: #C9D1D9;
                    --text-muted: #8B949E;
                    --primary-blue: #58A6FF;
                    --primary-orange: #F78166;
                    --success-green: #28A745;
                    --warning-yellow: #FFC107;
                    --font-main: 'Roboto', sans-serif;
                    --font-header: 'Exo 2', sans-serif;
                }}
                
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: var(--font-main);
                    background-color: transparent;
                    color: var(--text-light);
                }}
                
                .report-frame {{
                    height: {report_height}px;
                    width: 100%;
                    border: 1px solid var(--border-color);
                    border-radius: 10px;
                    overflow: hidden;
                    background-color: var(--bg-dark-1);
                }}
                
                .container {{
                    width: 100%;
                    max-width: 1600px;
                    margin: auto;
                    background-color: var(--bg-dark-1);
                    height: 100%;
                    overflow: auto;
                }}
                
                header {{
                    position: sticky;
                    top: 0;
                    z-index: 10;
                    padding: 1.5rem 2.5rem;
                    border-bottom: 1px solid var(--border-color);
                    background: linear-gradient(135deg, #1A2D40 0%, var(--bg-dark-2) 100%);
                    text-align: center;
                }}
                
                header h1 {{
                    font-family: var(--font-header);
                    font-size: 2.5rem;
                    margin: 0;
                    color: var(--primary-blue);
                    text-shadow: 0 2px 10px rgba(88, 166, 255, 0.3);
                }}
                
                header p {{
                    margin: 0.5rem 0 0;
                    color: var(--text-muted);
                    font-size: 1rem;
                }}
                
                .navbar {{
                    position: sticky;
                    top: 118px;
                    z-index: 10;
                    display: flex;
                    flex-wrap: wrap;
                    background-color: var(--bg-dark-2);
                    padding: 0 2.5rem;
                    border-bottom: 1px solid var(--border-color);
                }}
                
                .nav-btn {{
                    background: none;
                    border: none;
                    color: var(--text-muted);
                    padding: 1rem 1.5rem;
                    font-size: 1rem;
                    cursor: pointer;
                    border-bottom: 3px solid transparent;
                    transition: all 0.2s ease-in-out;
                }}
                
                .nav-btn:hover {{
                    color: var(--text-light);
                    background-color: rgba(88, 166, 255, 0.1);
                }}
                
                .nav-btn.active {{
                    color: var(--primary-orange);
                    border-bottom-color: var(--primary-orange);
                    font-weight: 700;
                }}
                
                main {{
                    padding: 2.5rem;
                }}
                
                .content-section {{
                    display: none;
                    animation: fadeIn 0.3s ease-in;
                }}
                
                .content-section.active {{
                    display: block;
                }}
                
                @keyframes fadeIn {{
                    from {{ opacity: 0; transform: translateY(10px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}
                
                h2, h3, h4 {{
                    font-family: var(--font-header);
                }}
                
                h2 {{
                    font-size: 2rem;
                    color: var(--primary-orange);
                    border-bottom: 2px solid var(--border-color);
                    padding-bottom: 0.5rem;
                    margin-top: 0;
                    margin-bottom: 2rem;
                }}
                
                h3 {{
                    color: var(--primary-blue);
                    font-size: 1.5rem;
                    margin-top: 2rem;
                    margin-bottom: 1rem;
                }}
                
                .grid-container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 1.5rem;
                    margin-bottom: 2rem;
                }}
                
                .grid-item {{
                    background-color: var(--bg-dark-2);
                    padding: 1.5rem;
                    border-radius: 8px;
                    border: 1px solid var(--border-color);
                    transition: transform 0.2s, box-shadow 0.2s;
                }}
                
                .grid-item:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(88, 166, 255, 0.2);
                }}
                
                .metric-label {{
                    font-size: 0.9rem;
                    color: var(--text-muted);
                    margin-bottom: 0.5rem;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }}
                
                .metric-value {{
                    font-size: 2rem;
                    font-weight: bold;
                    color: var(--primary-blue);
                }}
                
                .table-scroll-wrapper {{
                    margin-top: 1rem;
                    overflow: auto;
                    max-height: 500px;
                    border-radius: 8px;
                }}
                
                .styled-table {{
                    width: 100%;
                    color: var(--text-light);
                    background-color: var(--bg-dark-2);
                    border-collapse: collapse;
                    border-radius: 8px;
                    overflow: hidden;
                    font-size: 0.9rem;
                }}
                
                .styled-table th,
                .styled-table td {{
                    border-bottom: 1px solid var(--border-color);
                    padding: 0.8rem 1rem;
                    text-align: left;
                }}
                
                .styled-table thead th {{
                    background-color: var(--bg-dark-3);
                    position: sticky;
                    top: 0;
                    z-index: 1;
                    font-weight: 600;
                    color: var(--primary-blue);
                }}
                
                .styled-table tbody tr:hover {{
                    background-color: rgba(88, 166, 255, 0.1);
                }}
                
                .viz-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
                    gap: 2rem;
                    margin-top: 2rem;
                }}
                
                .viz-item {{
                    background-color: var(--bg-dark-2);
                    padding: 1.5rem;
                    border-radius: 8px;
                    border: 1px solid var(--border-color);
                    text-align: center;
                    transition: transform 0.2s;
                }}
                
                .viz-item:hover {{
                    transform: scale(1.02);
                }}
                
                .viz-item img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                }}
                
                .viz-item h4 {{
                    margin-top: 0;
                    margin-bottom: 1rem;
                    color: var(--primary-blue);
                }}
                
                .rank-item {{
                    background-color: var(--bg-dark-2);
                    padding: 1rem 1.5rem;
                    border-radius: 8px;
                    border: 1px solid var(--border-color);
                    margin-bottom: 1rem;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    transition: all 0.2s;
                }}
                
                .rank-item:hover {{
                    border-color: var(--primary-blue);
                    background-color: rgba(88, 166, 255, 0.05);
                }}
                
                .rank-item.best {{
                    border-color: var(--success-green);
                    background-color: rgba(40, 167, 69, 0.1);
                }}
                
                .rank-number {{
                    font-size: 1.5rem;
                    font-weight: bold;
                    color: var(--primary-orange);
                    margin-right: 1rem;
                    min-width: 40px;
                }}
                
                .model-name {{
                    font-size: 1.2rem;
                    font-weight: 600;
                    color: var(--text-light);
                    flex: 1;
                }}
                
                .model-score {{
                    font-size: 1.5rem;
                    font-weight: bold;
                    color: var(--primary-blue);
                }}
                
                .info-box {{
                    background-color: rgba(88, 166, 255, 0.1);
                    border-left: 4px solid var(--primary-blue);
                    padding: 1rem;
                    border-radius: 4px;
                    margin: 1rem 0;
                }}
                
                .warning-box {{
                    background-color: rgba(255, 193, 7, 0.1);
                    border-left: 4px solid var(--warning-yellow);
                    padding: 1rem;
                    border-radius: 4px;
                    margin: 1rem 0;
                }}
                
                ::-webkit-scrollbar {{
                    width: 8px;
                    height: 8px;
                }}
                
                ::-webkit-scrollbar-track {{
                    background: var(--bg-dark-3);
                }}
                
                ::-webkit-scrollbar-thumb {{
                    background: var(--border-color);
                    border-radius: 4px;
                }}
                
                ::-webkit-scrollbar-thumb:hover {{
                    background: var(--primary-blue);
                }}
            </style>
        </head>
        <body>
            <div id="{self.report_id}" class="report-frame">
                <div class="container">
                    <header>
                        <h1>Noventis AutoML Report</h1>
                        <p>Comprehensive machine learning analysis and model comparison</p>
                    </header>
                    <nav class="navbar">{navbar_html}</nav>
                    <main>{main_content_html}</main>
                </div>
            </div>
            <script>
                function showTab(event, tabName, reportId) {{
                    const reportFrame = document.getElementById(reportId);
                    if (!reportFrame) return;
                    
                    // Hide all sections and remove active from buttons
                    reportFrame.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
                    reportFrame.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
                    
                    // Show selected section and activate button
                    const sectionId = `${{tabName}}-${{reportId}}`;
                    const sectionToShow = reportFrame.querySelector(`#${{sectionId}}`);
                    if (sectionToShow) sectionToShow.classList.add('active');
                    event.currentTarget.classList.add('active');
                }}
            </script>
        </body>
        </html>
        """

    def _generate_overview_section(self) -> str:
        """Generate HTML for overview section."""
        # Determine best model
        if 'model_comparison' in self.results:
            best_model_name = self.results['model_comparison']['rankings'][0]['model']
        else:
            best_model_name = 'AutoML'
        
        train_test_ratio = (
            f"{len(self.X_train)/(len(self.X_train)+len(self.X_test))*100:.1f}% / "
            f"{len(self.X_test)/(len(self.X_train)+len(self.X_test))*100:.1f}%"
        )
        
        html = f"""
        <div class="grid-container">
            <div class="grid-item">
                <div class="metric-label">Task Type</div>
                <div class="metric-value">{self.task_type.title()}</div>
            </div>
            <div class="grid-item">
                <div class="metric-label">Best Model</div>
                <div class="metric-value" style="font-size: 1.5rem;">{best_model_name}</div>
            </div>
            <div class="grid-item">
                <div class="metric-label">Training Samples</div>
                <div class="metric-value">{len(self.X_train)}</div>
            </div>
            <div class="grid-item">
                <div class="metric-label">Test Samples</div>
                <div class="metric-value">{len(self.X_test)}</div>
            </div>
            <div class="grid-item">
                <div class="metric-label">Train/Test Split</div>
                <div class="metric-value" style="font-size: 1.2rem;">{train_test_ratio}</div>
            </div>
            <div class="grid-item">
                <div class="metric-label">Total Features</div>
                <div class="metric-value">{self.X_train.shape[1]}</div>
            </div>
        </div>
        
        <h3>Dataset Information</h3>
        <div class="grid-container">
            <div class="grid-item">
                <div class="metric-label">Target Column</div>
                <div class="metric-value" style="font-size: 1.3rem;">{self.target_column}</div>
            </div>
            <div class="grid-item">
                <div class="metric-label">Total Rows</div>
                <div class="metric-value">{self.df.shape[0]}</div>
            </div>
            <div class="grid-item">
                <div class="metric-label">Total Columns</div>
                <div class="metric-value">{self.df.shape[1]}</div>
            </div>
            <div class="grid-item">
                <div class="metric-label">Missing Values</div>
                <div class="metric-value">{self.df.isnull().sum().sum()}</div>
            </div>
        </div>
        """
        
        # Add task-specific statistics
        if self.task_type == "classification":
            class_dist = self.y_train.value_counts().to_dict()
            html += "<h3>Target Distribution</h3><div class='table-scroll-wrapper'><table class='styled-table'>"
            html += "<thead><tr><th>Class</th><th>Count</th><th>Percentage</th></tr></thead><tbody>"
            for cls, count in class_dist.items():
                percentage = (count / len(self.y_train)) * 100
                html += f"<tr><td>{cls}</td><td>{count}</td><td>{percentage:.2f}%</td></tr>"
            html += "</tbody></table></div>"
        else:
            html += f"""
            <h3>Target Statistics</h3>
            <div class='grid-container'>
                <div class='grid-item'>
                    <div class='metric-label'>Mean</div>
                    <div class='metric-value' style='font-size: 1.5rem;'>{self.y_train.mean():.4f}</div>
                </div>
                <div class='grid-item'>
                    <div class='metric-label'>Std Dev</div>
                    <div class='metric-value' style='font-size: 1.5rem;'>{self.y_train.std():.4f}</div>
                </div>
                <div class='grid-item'>
                    <div class='metric-label'>Min</div>
                    <div class='metric-value' style='font-size: 1.5rem;'>{self.y_train.min():.4f}</div>
                </div>
                <div class='grid-item'>
                    <div class='metric-label'>Max</div>
                    <div class='metric-value' style='font-size: 1.5rem;'>{self.y_train.max():.4f}</div>
                </div>
            </div>
            """
        
        # Add best model configuration if available
        best_model_res = self.results.get(
            best_model_name.replace(' ', '_'),
            self.results.get('AutoML', {})
        )
        
        if self.use_automl and best_model_res.get('best_config'):
            html += "<h3>Best Model Configuration</h3><div class='table-scroll-wrapper'>"
            html += "<table class='styled-table'><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>"
            for key, value in best_model_res['best_config'].items():
                html += f"<tr><td>{key}</td><td>{value}</td></tr>"
            html += "</tbody></table></div>"
        
        return html

    def _generate_metrics_section(
        self, 
        best_model_res: Dict[str, Any], 
        best_model_name: str
    ) -> str:
        """Generate HTML for metrics section."""
        metrics = best_model_res.get('metrics', {})
        
        html = f"<h3>Model: {best_model_name}</h3><div class='grid-container'>"
        for metric_name, metric_value in metrics.items():
            html += f"""
            <div class="grid-item">
                <div class="metric-label">{metric_name.replace('_', ' ').title()}</div>
                <div class="metric-value">{metric_value:.4f}</div>
            </div>
            """
        html += "</div>"
        
        # Feature importance table
        if best_model_res.get('feature_importance') is not None:
            fi_df = best_model_res['feature_importance'].head(15)
            html += "<h3>Top 15 Feature Importance</h3><div class='table-scroll-wrapper'>"
            html += "<table class='styled-table'><thead><tr><th>Rank</th><th>Feature</th><th>Importance</th></tr></thead><tbody>"
            for idx, (_, row) in enumerate(fi_df.iterrows(), 1):
                html += f"<tr><td>{idx}</td><td>{row['feature']}</td><td>{row['importance']:.4f}</td></tr>"
            html += "</tbody></table></div>"
        
        return html

    def _generate_predictions_section(self, best_model_res: Dict[str, Any]) -> str:
        """Generate HTML for predictions section."""
        predictions = best_model_res.get('predictions', [])
        actual = best_model_res.get('actual', [])
        
        html = "<h3>Sample Predictions (First 20)</h3><div class='table-scroll-wrapper'>"
        html += "<table class='styled-table'><thead><tr><th>Index</th><th>Actual</th><th>Predicted</th>"
        
        if self.task_type == "classification":
            html += "<th>Correct</th>"
        else:
            html += "<th>Error</th><th>Abs Error</th>"
        html += "</tr></thead><tbody>"
        
        for i in range(min(20, len(predictions))):
            actual_val = actual.iloc[i] if hasattr(actual, 'iloc') else actual[i]
            pred_val = predictions[i]
            
            if self.task_type == "classification":
                correct = "✓ Yes" if actual_val == pred_val else "✗ No"
                html += f"<tr><td>{i+1}</td><td>{actual_val}</td><td>{pred_val}</td><td>{correct}</td></tr>"
            else:
                error = pred_val - actual_val
                abs_error = abs(error)
                html += f"<tr><td>{i+1}</td><td>{actual_val:.4f}</td><td>{pred_val:.4f}</td>"
                html += f"<td>{error:.4f}</td><td>{abs_error:.4f}</td></tr>"
        
        html += "</tbody></table></div>"
        
        # Prediction confidence for classification
        if (self.task_type == "classification" and 
            best_model_res.get('prediction_proba') is not None):
            html += "<h3>Prediction Confidence (First 10)</h3><div class='table-scroll-wrapper'>"
            html += "<table class='styled-table'><thead><tr><th>Index</th><th>Predicted</th><th>Max Probability</th></tr></thead><tbody>"
            proba = best_model_res['prediction_proba']
            for i in range(min(10, len(proba))):
                max_proba = np.max(proba[i])
                html += f"<tr><td>{i+1}</td><td>{predictions[i]}</td><td>{max_proba:.4f}</td></tr>"
            html += "</tbody></table></div>"
        
        return html

    def _generate_visualizations_section(self) -> str:
        """Generate HTML for visualizations section."""
        html = "<div class='viz-grid'>"
        
        if 'visualization_paths' in self.results:
            for viz_path in self.results['visualization_paths']:
                viz_name = (
                    os.path.basename(viz_path)
                    .replace('.png', '')
                    .replace('_', ' ')
                    .title()
                )
                if os.path.exists(viz_path):
                    html += f"""
                    <div class="viz-item">
                        <h4>{viz_name}</h4>
                        <img src="{viz_path}" alt="{viz_name}">
                    </div>
                    """
        else:
            html += "<div class='info-box'><p>No visualizations available. Set explain=True to generate visualizations.</p></div>"
        
        html += "</div>"
        return html

    def _generate_comparison_section(self) -> str:
        """Generate HTML for model comparison section."""
        if 'model_comparison' not in self.results:
            return "<p>No comparison data available.</p>"
        
        rankings = self.results['model_comparison']['rankings']
        primary_metric = self.results['model_comparison']['primary_metric']
        
        html = f"<h3>Model Rankings (by {primary_metric.replace('_', ' ').title()})</h3>"
        html += "<div style='margin-bottom: 2rem;'>"
        
        for i, ranking in enumerate(rankings, 1):
            best_class = " best" if i == 1 else ""
            html += f"""
            <div class="rank-item{best_class}">
                <span class="rank-number">#{i}</span>
                <span class="model-name">{ranking['model']}</span>
                <span class="model-score">{ranking['score']:.4f}</span>
            </div>
            """
        html += "</div>"
        
        # All metrics comparison table
        html += "<h3>All Metrics Comparison</h3><div class='table-scroll-wrapper'>"
        html += "<table class='styled-table'><thead><tr><th>Model</th>"
        
        if rankings:
            for metric in rankings[0]['metrics'].keys():
                html += f"<th>{metric.replace('_', ' ').title()}</th>"
            html += "</tr></thead><tbody>"
            
            for ranking in rankings:
                html += f"<tr><td><strong>{ranking['model']}</strong></td>"
                for metric_value in ranking['metrics'].values():
                    html += f"<td>{metric_value:.4f}</td>"
                html += "</tr>"
        
        html += "</tbody></table></div>"
        
        # Visual comparison chart
        comparison_chart_path = os.path.join(self.output_dir, 'model_comparison.png')
        if os.path.exists(comparison_chart_path):
            html += f"""
            <h3>Visual Comparison</h3>
            <div class="viz-item">
                <img src="{comparison_chart_path}" alt="Model Comparison">
            </div>
            """
        
        return html

    def load_model(self, model_path: str) -> Any:
        """
        Load a saved model from disk.
        
        Args:
            model_path: Path to saved model file
            
        Returns:
            Loaded model object
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            print(f"✓ Model loaded from: {model_path}")
            return loaded_model
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return None

    def predict(
        self, 
        X_new: Union[pd.DataFrame, np.ndarray], 
        model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make predictions on new data using trained model.
        
        Args:
            X_new: Feature data for prediction (DataFrame or array)
            model_path: Optional path to load specific model. Uses trained model if None
            
        Returns:
            Dictionary with predictions and probabilities (for classification)
            
        Raises:
            ValueError: If no model is available
        """
        # Load model if path provided, otherwise use trained model
        model = self.load_model(model_path) if model_path else self.flaml_model
        
        if model is None:
            raise ValueError(
                "No model available. Either train a model first with fit() "
                "or specify model_path"
            )
        
        try:
            predictions = model.predict(X_new)
            print(f"✓ Prediction successful for {len(X_new)} samples")
            
            result = {'predictions': predictions}
            
            # Add probabilities for classification
            if self.task_type == "classification" and hasattr(model, 'predict_proba'):
                result['probabilities'] = model.predict_proba(X_new)
            
            return result
        
        except Exception as e:
            print(f"✗ Error during prediction: {e}")
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary with model details including estimator, config, and metadata
            
        Raises:
            ValueError: If no model has been trained
        """
        if not self.flaml_model:
            return {"error": "Model not trained. Run fit() first"}
        
        return {
            'best_estimator': self.flaml_model.best_estimator,
            'best_config': self.flaml_model.best_config,
            'task_type': self.task_type,
            'training_duration': getattr(
                self.flaml_model, 'training_duration', 'Unknown'
            ),
            'classes_': getattr(self.flaml_model, 'classes_', None),
            'feature_names': (
                list(self.X_train.columns) 
                if hasattr(self, 'X_train') and self.X_train is not None 
                else None
            )
        }

    def export_results_to_csv(self, output_dir: Optional[str] = None) -> None:
        """
        Export predictions, metrics, and feature importance to CSV files.
        
        Args:
            output_dir: Directory to save CSV files. Uses instance output_dir if None
        """
        if not hasattr(self, 'results') or not self.results:
            print("✗ No results to export. Run fit() first")
            return
        
        # Use provided directory or default
        output_dir = output_dir or str(self.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine best model
        if 'model_comparison' in self.results:
            best_model_name = self.results['model_comparison']['rankings'][0]['model']
            best_model_key = best_model_name.replace(' ', '_')
        else:
            best_model_key = 'AutoML'
        
        best_model_res = self.results.get(best_model_key, {})
        
        try:
            # 1. Export predictions
            predictions_df = pd.DataFrame({
                'actual': best_model_res['actual'],
                'predicted': best_model_res['predictions']
            })
            
            # Add probabilities for classification
            if best_model_res.get('prediction_proba') is not None:
                proba = best_model_res['prediction_proba']
                proba_cols = [f'prob_class_{i}' for i in range(proba.shape[1])]
                proba_df = pd.DataFrame(proba, columns=proba_cols)
                predictions_df = pd.concat([predictions_df, proba_df], axis=1)
            
            pred_path = os.path.join(output_dir, 'predictions.csv')
            predictions_df.to_csv(pred_path, index=False)
            print(f"✓ Predictions exported: {pred_path}")
            
            # 2. Export metrics
            metrics_df = pd.DataFrame([best_model_res['metrics']])
            metrics_path = os.path.join(output_dir, 'metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            print(f"✓ Metrics exported: {metrics_path}")
            
            # 3. Export feature importance (if available)
            if best_model_res.get('feature_importance') is not None:
                fi_path = os.path.join(output_dir, 'feature_importance.csv')
                best_model_res['feature_importance'].to_csv(fi_path, index=False)
                print(f"✓ Feature importance exported: {fi_path}")
            
            print(f"\n✓ All results exported to: {output_dir}")
        
        except Exception as e:
            print(f"✗ Error exporting to CSV: {e}")

    def _repr_html_(self) -> str:
        """
        Generate HTML representation for Jupyter notebooks.
        
        Returns:
            HTML string for display
        """
        if hasattr(self, 'results') and self.results:
            return self.generate_html_report()._repr_html_()
        return "<p>NoventisAutoML instance - Run fit() to see interactive dashboard</p>"

    def __repr__(self) -> str:
        """
        String representation of NoventisAutoML instance.
        
        Returns:
            Formatted string with instance information
        """
        # Determine training status
        status = "Trained" if self.flaml_model or self.manual_model else "Not Trained"
        
        # Determine best model
        best_model = "Unknown"
        if hasattr(self, 'results') and 'model_comparison' in self.results:
            best_model = self.results['model_comparison']['rankings'][0]['model']
        elif self.flaml_model:
            best_model = getattr(self.flaml_model, 'best_estimator', 'AutoML')
        
        # Get dataset shape
        shape = getattr(self, 'df', pd.DataFrame()).shape
        
        return (
            f"NoventisAutoML(\n"
            f"  task='{self.task_type}',\n"
            f"  target='{self.target_column}',\n"
            f"  status='{status}',\n"
            f"  best_model='{best_model}',\n"
            f"  dataset_shape={shape}\n"
            f")"
        )

    def __str__(self) -> str:
        """
        User-friendly string representation.
        
        Returns:
            Formatted string for printing
        """
        return self.__repr__()


# Additional utility functions for the module
def compare_multiple_datasets(
    datasets: List[Union[str, pd.DataFrame]],
    target: str,
    task: str,
    output_dir: str = 'multi_dataset_comparison'
) -> Dict[str, Any]:
    """
    Compare model performance across multiple datasets.
    
    Args:
        datasets: List of CSV paths or DataFrames
        target: Target column name (must be same across datasets)
        task: Task type ('classification' or 'regression')
        output_dir: Directory to save comparison results
        
    Returns:
        Dictionary with comparison results for all datasets
    """
    results = {}
    os.makedirs(output_dir, exist_ok=True)
    
    for i, dataset in enumerate(datasets):
        print(f"\n{'='*60}")
        print(f"Processing Dataset {i+1}/{len(datasets)}")
        print(f"{'='*60}")
        
        dataset_dir = os.path.join(output_dir, f'dataset_{i+1}')
        
        # Train model on this dataset
        automl = NoventisAutoML(
            data=dataset,
            target=target,
            task=task,
            output_dir=dataset_dir,
            explain=True,
            compare=True
        )
        
        dataset_results = automl.fit()
        results[f'dataset_{i+1}'] = dataset_results
    
    return results


def load_and_predict(
    model_path: str,
    data: Union[str, pd.DataFrame],
    task_type: str
) -> np.ndarray:
    """
    Convenience function to load model and make predictions.
    
    Args:
        model_path: Path to saved model file
        data: New data for prediction (CSV path or DataFrame)
        task_type: Task type ('classification' or 'regression')
        
    Returns:
        Array of predictions
    """
    # Load data
    if isinstance(data, str):
        X_new = pd.read_csv(data)
    else:
        X_new = data
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions
    predictions = model.predict(X_new)
    
    return predictions


# Module-level constants
__version__ = "2.0.0"
__author__ = "Noventis Team"
__all__ = [
    'NoventisAutoML',
    'compare_multiple_datasets',
    'load_and_predict'
]