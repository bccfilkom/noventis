# ==============================================================================
# ENHANCED NOVENTIS AUTOML - COMPLETE CODE WITH IMPROVED DASHBOARD
# ==============================================================================

import pandas as pd
import numpy as np
import os
import pickle
import io
import base64
import uuid
from typing import Union, Optional, Dict, Any, List
import warnings

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
    mean_absolute_error, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML, display

try:
    from .manual import ManualPredictor
except (ImportError, ModuleNotFoundError):
    print("Warning: 'ManualPredictor' class not found. Manual model comparison features will be limited.")
    class ManualPredictor:
        def __init__(self, *args, **kwargs): pass
        def run_pipeline(self, *args, **kwargs): return {'all_model_results': []}
        def save_model(self, *args, **kwargs): pass

warnings.filterwarnings('ignore')

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding."""
    if fig is None: return ""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

# Modern dark theme for plots
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.titlesize': 12, 'figure.titlesize': 14, 'legend.fontsize': 10,
    'figure.facecolor': '#0f1419', 'axes.facecolor': '#0f1419',
    'text.color': '#e6edf3', 'axes.labelcolor': '#e6edf3',
    'xtick.color': '#7d8590', 'ytick.color': '#7d8590',
    'grid.color': '#21262d', 'patch.edgecolor': '#21262d',
    'figure.edgecolor': '#0f1419',
})

class NoventisAutoML:
    """
    Main class for running AutoML pipeline using FLAML.
    Supports automatic task type detection (classification/regression), model training,
    evaluation, model saving, visualization, and comparison with manual models.
    """

    def __init__(
        self, 
        data: Union[str, pd.DataFrame], 
        target: str, 
        task: Optional[str] = None, 
        models: List[str] = None,
        explain: bool = True,
        compare: bool = True,
        metrics: str = None,
        time_budget: int = 60,
        output_dir: str = 'Noventis_Results',
        test_size: float = 0.2, 
        random_state: int = 42,
    ):
        self.target_column = target
        self.task_type = task.lower() if task else None
        self.test_size = test_size
        self.random_state = random_state
        self.explain = explain
        self.compare = compare
        self.metrics = metrics
        self.time_budget = time_budget
        self.output_dir = output_dir
        self.model_list = models
        self.use_automl = True if (compare or self.model_list is None) else False 

        self.flaml_model = None
        self.manual_model = None
        self.results = {}
        self.report_id = f"report-{uuid.uuid4().hex[:8]}"
        self.visualizations_b64 = {}
        
        self._load_data(data)
        self._setup_data()

    def _load_data(self, data: Union[str, pd.DataFrame]):
        """Load data from CSV or DataFrame"""
        if isinstance(data, str):
            self.df = pd.read_csv(data)
            print(f"Data successfully loaded from file: {data}")
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
            print("Data successfully loaded from DataFrame")
        else:
            raise TypeError("Input 'data' must be a CSV path or pandas DataFrame.")
        
        print(f"Data shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")

    def _detect_task_type(self) -> str:
        """Auto-detect task type based on target variable"""
        y = self.df[self.target_column]
        unique_values = len(y.unique())
        unique_ratio = unique_values / len(y)
        
        if pd.api.types.is_numeric_dtype(y):
            if unique_values > 25 and unique_ratio >= 0.05:
                return "regression"
            else:
                return "classification"
        else:
            return "classification"

    def _setup_data(self):
        """Setup and split data for training and testing"""
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found.")
        
        X, y = self.df.drop(columns=[self.target_column]), self.df[self.target_column]
        
        if self.task_type is None:
            self.task_type = self._detect_task_type()
            print(f"Detected task type: {self.task_type}")
        
        stratify = y if self.task_type == "classification" else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify
        )
        
        print(f"Data split successfully: Train={len(self.X_train)}, Test={len(self.X_test)}")
        print(f"Target distribution: {dict(y.value_counts()) if self.task_type == 'classification' else f'Range: {y.min():.2f} - {y.max():.2f}'}")

    def fit(self, time_budget: int = 60, metric: Optional[str] = None) -> Dict:
        """Train AutoML model with FLAML"""
        print("Starting AutoML process with FLAML...")
        
        flaml_metric = self._convert_metric_to_flaml(self.metrics)
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.use_automl:      
            self.flaml_model = FLAMLAutoML(
                task=self.task_type, 
                metric=flaml_metric, 
                seed=self.random_state, 
                verbose=2,
            )
            
            print(f"Training model (Metric: {flaml_metric}, Time: {time_budget}s)...")
            
            self.flaml_model.fit(
                X_train=self.X_train, 
                y_train=self.y_train,
                log_file_name=f'{self.output_dir}/flaml.log', 
                time_budget=self.time_budget,
            )
            
            y_pred = self.flaml_model.predict(self.X_test)
            
            if self.task_type == "classification":
                metrics = self._eval_classification(self.y_test, y_pred)
                y_pred_proba = self.flaml_model.predict_proba(self.X_test) if hasattr(self.flaml_model, 'predict_proba') else None
            else:
                metrics = self._eval_regression(self.y_test, y_pred)
                y_pred_proba = None
            
            self.results['AutoML'] = {
                'model': self.flaml_model,
                'model_name': 'AutoML',
                'predictions': y_pred,
                'prediction_proba': y_pred_proba,
                'actual': self.y_test,
                'metrics': metrics,
                'task_type': self.task_type,
                'feature_importance': self._get_feature_importance(),
                'best_estimator': self.flaml_model.best_estimator,
                'best_config': self.flaml_model.best_config,
                'training_history': self._get_training_history(f'{self.output_dir}/flaml.log')
            }
        else:
            predictor = ManualPredictor(
                model_name=self.model_list, 
                task=self.task_type, 
                random_state=self.random_state
            )
            result = predictor.run_pipeline(
                self.df, 
                target_column=self.target_column, 
                test_size=self.test_size
            )
            self.manual_model = predictor

            all_model_results = result['all_model_results']
            for model_result in all_model_results:
                if 'error' in model_result:
                    print(f"Warning: Skipping model '{model_result.get('model_name', 'Unknown')}' due to training failure.")
                    continue
                model_name = model_result['model_name']
                metrics = model_result['metrics']
                y_pred = model_result['predictions']
                y_pred_proba = model_result['prediction_proba']
                
                self.results[model_name] = {
                    'model': model_name,
                    'model_name': model_name,
                    'predictions': y_pred,
                    'prediction_proba': y_pred_proba,
                    'actual': self.y_test,
                    'metrics': metrics,
                    'task_type': self.task_type,
                    'feature_importance': None,
                    'best_estimator': None,
                    'best_config': None,
                    'training_history': None
                }
            
            predictor.save_model(f'{self.output_dir}/best_model.pkl')

        model_path = os.path.join(self.output_dir, 'best_automl_model.pkl')
        
        if self.compare:
            print("\nStarting model comparison...")
            comparison_results = self.compare_models(output_dir=self.output_dir, models_to_compare=self.model_list)
            self.results['model_comparison'] = comparison_results

            print('METRICS =', self.metrics)
            best_manual_model_metrics = self.results['model_comparison']['rankings'][0]['score']
            best_manual_model_name = self.results['model_comparison']['rankings'][0]['model'].lower()
            automl_metrics = self.results['AutoML']['metrics'][self.metrics]
            if best_manual_model_metrics > automl_metrics:
                self.manual_model.save_model(f'{self.output_dir}/best_model.pkl')
                self.results[best_manual_model_name]['model_path'] = model_path
            else:
                self._save_model(self.flaml_model, model_path)
                self.results['AutoML']['model_path'] = model_path
                print(f"Best estimator: {self.flaml_model.best_estimator}")
                print(f"Metrics: {metrics}")

            print(f"\nComparison process completed!")
        else:
            if self.use_automl:
                self._save_model(self.flaml_model, model_path)
                self.results['AutoML']['model_path'] = model_path
                print(f"Model saved at: {model_path}")
                print(f"\nProcess completed!")
                print(f"Best estimator: {self.flaml_model.best_estimator}")
                print(f"Metrics: {metrics}")
            else:
                comparison_results = self.compare_models(output_dir=self.output_dir, models_to_compare=self.model_list)
                self.results['model_comparison'] = comparison_results

                print('METRICS =', self.metrics)
                best_manual_model_metrics = self.results['model_comparison']['rankings'][0]['score']
                best_manual_model_name = self.results['model_comparison']['rankings'][0]['model']
                self.manual_model.save_model(f'{self.output_dir}/best_model.pkl')
                self.results[best_manual_model_name]['model_path'] = model_path
                print(f"\nProcess completed!")

        if self.explain:
            print("Creating visualizations...")
            self.results['visualization_paths'] = self._generate_visualizations(self.results, self.output_dir)
            self._generate_model_summary(self.results, self.output_dir)
            print(f"Visualizations created and saved in '{self.output_dir}' directory!")
        
        return self.results

    def compare_models(self, models_to_compare: Optional[List[str]] = None, 
                      output_dir: str = "Noventis_results") -> Dict:
        """Compare AutoML performance with other manual models"""
        os.makedirs(output_dir, exist_ok=True)
        all_results = {}
        
        if models_to_compare is None:
            if self.task_type == 'classification':
                models_to_compare = ['logistic_regression', 'random_forest', 'xgboost', 'decision_tree', 'lightgbm', 'catboost', 'gradient_boosting']
            else:
                models_to_compare = ['linear_regression', 'random_forest', 'xgboost', 'gradient_boosting', 'lightgbm', 'catboost']

            if 'AutoML' in self.results:
                automl_metrics = self.results['AutoML']['metrics']
                all_results['AutoML'] = {
                    'metrics': automl_metrics,
                    'model_name': 'AutoML',
                    'best_estimator': getattr(self.flaml_model, 'best_estimator', 'Unknown') if self.flaml_model else 'Unknown'
                }
        
        if self.manual_model is None:
            predictor = ManualPredictor(
                model_name=models_to_compare, 
                task=self.task_type, 
                random_state=self.random_state
            )
            result = predictor.run_pipeline(
                self.df, 
                target_column=self.target_column, 
                test_size=self.test_size
            )
            self.manual_model = predictor

            all_model_results = result['all_model_results']
            for model_result in all_model_results:
                if 'error' in model_result:
                    print(f"Warning: Skipping model '{model_result.get('model_name', 'Unknown')}' due to training failure.")
                    continue

                model_name = model_result['model_name']
                metrics = model_result['metrics']
                y_pred = model_result['predictions']
                y_pred_proba = model_result['prediction_proba']
                
                self.results[model_name] = {
                    'model': model_name,
                    'model_name': model_name,
                    'predictions': y_pred,
                    'prediction_proba': y_pred_proba,
                    'actual': self.y_test,
                    'metrics': metrics,
                    'task_type': self.task_type,
                    'feature_importance': None,
                    'best_estimator': None,
                    'best_config': None,
                    'training_history': None
                }

        for model in self.results:
            all_results[model] = self.results[model]
        
        ranked_results = self._rank_models(all_results)
        self.results['model_comparison'] = ranked_results

        if self.compare:
            self._visualize_model_comparison(ranked_results, all_results, output_dir)
            self._generate_comparison_report(ranked_results, all_results, output_dir)
            print(f"Model comparison results saved in '{output_dir}' directory.")
        return ranked_results

    def _eval_classification(self, y_true, y_pred) -> Dict:
        """Complete evaluation for classification"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0)
        }

    def _eval_regression(self, y_true, y_pred) -> Dict:
        """Complete evaluation for regression"""
        mse = mean_squared_error(y_true, y_pred)
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2_score': r2_score(y_true, y_pred)
        }

    def _convert_metric_to_flaml(self, metric: Optional[str]) -> str:
        """Convert metric name for FLAML"""
        if metric is None:
            return "macro_f1" if self.task_type == "classification" else "r2"
        elif metric == 'f1_score':
            return 'macro_f1'
        elif metric == 'r2_score':
            return 'r2'
        return metric

    def _get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Extract feature importance if available"""
        if self.use_automl:
            try:
                if hasattr(self.flaml_model, 'model') and hasattr(self.flaml_model.model, 'feature_importances_'):
                    importances = self.flaml_model.model.feature_importances_
                    return pd.DataFrame({
                        'feature': self.X_train.columns, 
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                elif hasattr(self.flaml_model, 'feature_importances_'):
                    return pd.DataFrame({
                        'feature': self.X_train.columns, 
                        'importance': self.flaml_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                else:
                    return None
            except Exception as e:
                print(f"Warning: Cannot extract feature importance: {e}")
                return None
        return None

    def _get_training_history(self, log_file) -> Optional[pd.DataFrame]:
        """Extract training history from FLAML log file"""
        if self.use_automl:
            try:
                if os.path.exists(log_file):
                    time_h, loss_h, _, _, _ = get_output_from_log(filename=log_file, time_budget=float('inf'))
                    return pd.DataFrame({
                        'time_seconds': time_h, 
                        'best_validation_loss': loss_h
                    })
                return None
            except Exception as e:
                print(f"Warning: Cannot read training history: {e}")
                return None

    def _save_model(self, model, path):
        """Save model to pickle file"""
        try:
            with open(path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            print(f"Warning: Error saving model: {e}")

    def _generate_visualizations(self, results, output_dir: str) -> List[str]:
        """Generate comprehensive visualizations"""
        paths = []
        plt.style.use('default')
        if not 'model_comparison' in results:
             best_model_res = results['AutoML']
        else:
             best_model_name = results['model_comparison']['rankings'][0]['model']
             best_model_res = results[best_model_name]
        
        try:
            if self.use_automl:
                if 'feature_importance' in best_model_res and best_model_res['feature_importance'] is not None and not best_model_res['feature_importance'].empty:
                    plt.figure(figsize=(12, max(6, len(best_model_res['feature_importance']) * 0.4)))
                    top_features = best_model_res['feature_importance'].head(20)
                    sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
                    plt.title('Top 20 Feature Importance', fontsize=16, fontweight='bold')
                    plt.xlabel('Importance Score', fontsize=12)
                    plt.ylabel('Features', fontsize=12)
                    plt.tight_layout()
                    path = os.path.join(output_dir, 'feature_importance.png')
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    paths.append(path)
                    plt.close()

                if 'training_history' in best_model_res and best_model_res['training_history'] is not None and not best_model_res['training_history'].empty:
                    plt.figure(figsize=(12, 6))
                    history = best_model_res['training_history']
                    plt.plot(history['time_seconds'], history['best_validation_loss'], marker='o', linestyle='-', color='b', linewidth=2, markersize=4)
                    plt.title('AutoML Training Progress', fontsize=16, fontweight='bold')
                    plt.xlabel('Time (seconds)', fontsize=12)
                    plt.ylabel('Best Validation Loss', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    path = os.path.join(output_dir, 'training_history.png')
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    paths.append(path)
                    plt.close()

            if self.task_type == 'classification':
                paths.extend(self._generate_classification_plots(best_model_res, output_dir))
            else:
                paths.extend(self._generate_regression_plots(best_model_res, output_dir))
                
        except Exception as e:
            print(f"Warning: Error creating visualizations: {e}")
        return paths

    def _generate_classification_plots(self, results, output_dir: str) -> List[str]:
        """Generate classification-specific plots"""
        paths = []
        try:
            # Enhanced Confusion Matrix
            plt.figure(figsize=(12, 8))
            cm = confusion_matrix(results['actual'], results['predictions'])
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create subplots for both normalized and raw confusion matrix
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax1)
            ax1.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Predicted Label')
            ax1.set_ylabel('True Label')
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax2)
            ax2.set_title('Raw Confusion Matrix', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Predicted Label')
            ax2.set_ylabel('True Label')
            
            plt.tight_layout()
            path = os.path.join(output_dir, 'confusion_matrix_enhanced.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            paths.append(path)
            plt.close()
            
            # Classification Metrics Radar Chart
            metrics = results['metrics']
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            values = list(metrics.values())
            angles += angles[:1]  # Complete the circle
            values += values[:1]
            
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
            ax.plot(angles, values, 'o-', linewidth=2, label='Model Performance')
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([k.replace('_', ' ').title() for k in metrics.keys()])
            ax.set_ylim(0, 1)
            ax.set_title('Classification Metrics Radar Chart', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            path = os.path.join(output_dir, 'classification_radar.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            paths.append(path)
            plt.close()
            
            # ROC Curve if probabilities available
            if results.get('prediction_proba') is not None and len(np.unique(results['actual'])) == 2:
                y_proba = results['prediction_proba'][:, 1] if results['prediction_proba'].shape[1] > 1 else results['prediction_proba']
                fpr, tpr, _ = roc_curve(results['actual'], y_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(10, 8))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve Analysis', fontsize=16, fontweight='bold')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                path = os.path.join(output_dir, 'roc_curve.png')
                plt.savefig(path, dpi=300, bbox_inches='tight')
                paths.append(path)
                plt.close()
                
        except Exception as e:
            print(f"Warning: Error creating classification plots: {e}")
        return paths

    def _generate_regression_plots(self, results, output_dir: str) -> List[str]:
        """Generate regression-specific plots"""
        paths = []
        try:
            # Enhanced Prediction vs Actual Plot with density
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Scatter plot with trend line
            ax1.scatter(results['actual'], results['predictions'], alpha=0.6, edgecolors='k', s=50)
            min_val = min(min(results['actual']), min(results['predictions']))
            max_val = max(max(results['actual']), max(results['predictions']))
            perfect_line = np.linspace(min_val, max_val, 100)
            ax1.plot(perfect_line, perfect_line, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
            
            # Add trend line
            z = np.polyfit(results['actual'], results['predictions'], 1)
            p = np.poly1d(z)
            ax1.plot(results['actual'], p(results['actual']), "g--", alpha=0.8, linewidth=2, label='Trend Line')
            
            ax1.set_title('Predictions vs. Actual Values', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Actual Values')
            ax1.set_ylabel('Predicted Values')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            r2 = results['metrics'].get('r2_score', 0)
            ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Residuals Plot
            residuals = np.array(results['actual']) - np.array(results['predictions'])
            ax2.scatter(results['predictions'], residuals, alpha=0.6, edgecolors='k')
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
            ax2.set_title('Residuals Plot')
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.grid(True, alpha=0.3)
            
            # Residuals histogram
            ax3.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            ax3.set_title('Residuals Distribution')
            ax3.set_xlabel('Residuals')
            ax3.set_ylabel('Frequency')
            ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
            
            # Q-Q plot for residual normality
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=ax4)
            ax4.set_title('Q-Q Plot (Residuals Normality)')
            
            plt.tight_layout()
            path = os.path.join(output_dir, 'regression_analysis_enhanced.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            paths.append(path)
            plt.close()
            
            # Metrics visualization
            metrics = results['metrics']
            plt.figure(figsize=(12, 6))
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(metric_names)))
            bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.8)
            
            plt.title('Regression Metrics Overview', fontsize=16, fontweight='bold')
            plt.ylabel('Score')
            for bar, value in zip(bars, metric_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(metric_values) * 0.01), 
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            plt.xticks(rotation=45)
            plt.tight_layout()
            path = os.path.join(output_dir, 'regression_metrics.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            paths.append(path)
            plt.close()
            
        except Exception as e:
            print(f"Warning: Error creating regression plots: {e}")
        return paths

    def _generate_model_summary(self, results, output_dir: str):
        """Generate comprehensive model summary with insights"""
        try:
            summary_path = os.path.join(output_dir, 'model_summary.txt')
            with open(summary_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("                NOVENTIS AUTOML - COMPREHENSIVE MODEL ANALYSIS\n")
                f.write("="*80 + "\n\n")
                f.write(f"Task Type: {self.task_type.upper()}\n")
                f.write(f"Dataset Shape: {self.df.shape}\n")
                f.write(f"Training Duration Budget: {self.time_budget} seconds\n\n")
                
                if 'model_comparison' in results and results['model_comparison']['rankings']:
                    best_model = results['model_comparison']['rankings'][0]
                    model_name = best_model['model']
                    model_results = results[model_name]
                    
                    f.write("BEST MODEL PERFORMANCE SUMMARY\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"Champion Model: {model_name}\n")
                    f.write(f"Primary Metric ({results['model_comparison']['primary_metric']}): {best_model['score']:.4f}\n\n")
                    
                    f.write("DETAILED METRICS:\n")
                    f.write("-" * 30 + "\n")
                    for metric, value in model_results['metrics'].items():
                        status = self._get_metric_status(metric, value, self.task_type)
                        f.write(f"{metric.replace('_', ' ').title():<20}: {value:.4f} {status}\n")
                    
                    # Model insights and recommendations
                    f.write(f"\nMODEL ANALYSIS & INSIGHTS:\n")
                    f.write("-" * 40 + "\n")
                    insights = self._generate_model_insights(model_results, self.task_type)
                    for insight in insights:
                        f.write(f"• {insight}\n")
                    
                    if 'feature_importance' in model_results and model_results['feature_importance'] is not None:
                        f.write(f"\nTOP 15 MOST IMPORTANT FEATURES:\n")
                        f.write("-" * 40 + "\n")
                        for idx, row in model_results['feature_importance'].head(15).iterrows():
                            f.write(f"{row['feature']:<30}: {row['importance']:.6f}\n")
                    
                    # Model comparison summary
                    if len(results['model_comparison']['rankings']) > 1:
                        f.write(f"\nMODEL RANKING SUMMARY:\n")
                        f.write("-" * 30 + "\n")
                        for i, ranking in enumerate(results['model_comparison']['rankings'][:5], 1):
                            f.write(f"{i}. {ranking['model']:<20}: {ranking['score']:.4f}\n")
                    
                else:
                    # Single model summary
                    model_results = results.get('AutoML', {})
                    f.write("SINGLE MODEL ANALYSIS\n")
                    f.write("-" * 30 + "\n")
                    if 'metrics' in model_results:
                        for metric, value in model_results['metrics'].items():
                            status = self._get_metric_status(metric, value, self.task_type)
                            f.write(f"{metric.replace('_', ' ').title():<20}: {value:.4f} {status}\n")
                
                f.write(f"\nRECOMMendations for Model Improvement:\n")
                f.write("-" * 50 + "\n")
                recommendations = self._generate_recommendations(results)
                for rec in recommendations:
                    f.write(f"• {rec}\n")
                    
            print(f"Comprehensive summary report saved at: {summary_path}")
        except Exception as e:
            print(f"Warning: Error creating summary: {e}")

    def _get_metric_status(self, metric: str, value: float, task_type: str) -> str:
        """Get status indicator for metric performance"""
        if task_type == 'classification':
            if metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                if value >= 0.9: return "(Excellent)"
                elif value >= 0.8: return "(Good)"
                elif value >= 0.7: return "(Fair)"
                else: return "(Needs Improvement)"
        else:  # regression
            if metric == 'r2_score':
                if value >= 0.9: return "(Excellent)"
                elif value >= 0.8: return "(Good)"
                elif value >= 0.7: return "(Fair)"
                else: return "(Needs Improvement)"
            elif metric in ['mae', 'mse', 'rmse']:
                return "(Lower is Better)"
        return ""

    def _generate_model_insights(self, model_results: Dict, task_type: str) -> List[str]:
        """Generate actionable insights about the model performance"""
        insights = []
        metrics = model_results.get('metrics', {})
        
        if task_type == 'classification':
            accuracy = metrics.get('accuracy', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            f1 = metrics.get('f1_score', 0)
            
            if accuracy > 0.9:
                insights.append("Model shows excellent accuracy performance, suitable for production deployment")
            elif accuracy < 0.7:
                insights.append("Model accuracy is below optimal threshold, consider feature engineering or data quality improvements")
            
            if precision > recall:
                insights.append("Model is more conservative (high precision, lower recall) - good for minimizing false positives")
            elif recall > precision:
                insights.append("Model is more aggressive (high recall, lower precision) - good for capturing all positive cases")
            
            if f1 < 0.7:
                insights.append("F1-score indicates imbalanced performance between precision and recall")
                
        else:  # regression
            r2 = metrics.get('r2_score', 0)
            mae = metrics.get('mae', float('inf'))
            rmse = metrics.get('rmse', float('inf'))
            
            if r2 > 0.9:
                insights.append("Model explains over 90% of target variance - excellent predictive power")
            elif r2 < 0.7:
                insights.append("Model explains less than 70% of variance - consider additional features or model complexity")
            
            if rmse > mae * 1.5:
                insights.append("RMSE significantly higher than MAE suggests presence of outliers affecting predictions")
                
        if len(insights) == 0:
            insights.append("Model performance is within acceptable ranges for the given task")
            
        return insights

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations for model improvement"""
        recommendations = []
        
        if 'model_comparison' in results:
            rankings = results['model_comparison']['rankings']
            if len(rankings) > 1:
                best_score = rankings[0]['score']
                second_score = rankings[1]['score']
                if abs(best_score - second_score) < 0.05:
                    recommendations.append("Multiple models show similar performance - consider ensemble methods")
        
        recommendations.extend([
            "Increase time_budget parameter for potentially better model discovery",
            "Experiment with feature engineering techniques (scaling, encoding, interactions)",
            "Consider collecting more training data if current dataset is limited",
            "Evaluate model performance on different data splits for robustness",
            "Implement cross-validation for more reliable performance estimates"
        ])
        
        return recommendations

    def _rank_models(self, results) -> Dict:
        """Rank models based on performance with enhanced metadata"""
        rankings = []
        if self.task_type == "classification":
            primary_metric = 'f1_score' if self.metrics is None else self.metrics   
        else:
            primary_metric = 'r2_score' if self.metrics is None else self.metrics
        self.metrics = primary_metric
        
        for name, res in results.items():
            if name == 'model_comparison' or 'error' in res or 'metrics' not in res:
                continue
            score = res['metrics'].get(primary_metric, -1)      
            model_display_name = name.replace('_', ' ')
            rankings.append({
                'model': model_display_name, 
                'score': score, 
                'metrics': res['metrics'],
                'estimator': res.get('best_estimator', 'Unknown')
            })
        
        rankings.sort(key=lambda x: x['score'], reverse=True)
        return {
            'rankings': rankings, 
            'best_model': rankings[0]['model'] if rankings else None,
            'best_model_name': rankings[0]['model'] if rankings else None, 
            'primary_metric': primary_metric,
            'total_models_tested': len(rankings)
        }

    def _visualize_model_comparison(self, ranked_results, all_results, output_dir: str):
        """Create comprehensive model comparison visualizations with enhanced styling"""
        if not ranked_results['rankings']:
            print("No models available for comparison.")
            return
        try:
            df_ranks = pd.DataFrame(ranked_results['rankings'])
            primary_metric = ranked_results['primary_metric']
            metric_display = primary_metric.replace('_', ' ').title()
            
            # Enhanced model comparison chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # Horizontal bar chart with gradient colors
            colors = plt.cm.plasma(np.linspace(0, 1, len(df_ranks)))
            bars = ax1.barh(df_ranks['model'], df_ranks['score'], color=colors)
            ax1.set_title(f'Model Performance Comparison\n({metric_display})', fontsize=16, fontweight='bold')
            ax1.set_xlabel(metric_display, fontsize=12)
            ax1.set_ylabel('Model', fontsize=12)
            
            # Add value labels
            for i, (bar, score) in enumerate(zip(bars, df_ranks['score'])):
                ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{score:.4f}', va='center', ha='left', fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
            
            # Performance gap analysis
            if len(df_ranks) > 1:
                scores = df_ranks['score'].values
                gaps = np.diff(scores)
                ax2.bar(range(1, len(gaps) + 1), gaps, color='coral', alpha=0.7)
                ax2.set_title('Performance Gaps Between Models', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Model Rank Difference')
                ax2.set_ylabel('Score Difference')
                ax2.grid(True, alpha=0.3)
                
                # Add gap values
                for i, gap in enumerate(gaps):
                    ax2.text(i + 1, gap + max(gaps) * 0.01, f'{gap:.4f}', 
                            ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            path = os.path.join(output_dir, 'model_comparison_enhanced.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._create_metrics_heatmap(df_ranks, output_dir)
            
        except Exception as e:
            print(f"Warning: Error creating comparison visualizations: {e}")

    def _create_metrics_heatmap(self, df_ranks, output_dir: str):
        """Create enhanced heatmap of all metrics across models"""
        try:
            all_metrics = {}
            for ranking in df_ranks.to_dict('records'):
                model_name = ranking['model']
                metrics = ranking['metrics']
                all_metrics[model_name] = metrics
            
            if not all_metrics or len(list(all_metrics.values())[0]) <= 1:
                return
                
            metrics_df = pd.DataFrame(all_metrics).T
            
            # Create enhanced heatmap with annotations
            plt.figure(figsize=(14, max(8, len(metrics_df) * 0.8)))
            mask = metrics_df.isnull()
            
            sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                       cbar_kws={'label': 'Score'}, mask=mask,
                       linewidths=0.5, square=True)
            
            plt.title('Comprehensive Model Performance Heatmap\nAll Metrics Comparison', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('Metrics', fontsize=12)
            plt.ylabel('Models', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            path = os.path.join(output_dir, 'metrics_heatmap_enhanced.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create metric distribution plot
            if len(metrics_df.columns) > 2:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.ravel()
                
                for i, metric in enumerate(metrics_df.columns[:4]):
                    if i < len(axes):
                        metrics_df[metric].plot(kind='bar', ax=axes[i], color=plt.cm.Set3(i))
                        axes[i].set_title(f'{metric.replace("_", " ").title()} Distribution')
                        axes[i].tick_params(axis='x', rotation=45)
                        axes[i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                path = os.path.join(output_dir, 'metrics_distribution.png')
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"Warning: Error creating metrics heatmap: {e}")

    def _generate_comparison_report(self, ranked_results, all_results, output_dir: str):
        """Generate detailed comparison report with insights"""
        try:
            report_path = os.path.join(output_dir, 'model_comparison_report.txt')
            with open(report_path, 'w') as f:
                f.write("="*90 + "\n")
                f.write("                    NOVENTIS AUTOML - COMPREHENSIVE MODEL COMPARISON REPORT\n")
                f.write("="*90 + "\n\n")
                f.write(f"Task Type: {self.task_type.title()}\n")
                f.write(f"Primary Evaluation Metric: {ranked_results['primary_metric']}\n")
                f.write(f"Total Models Evaluated: {ranked_results['total_models_tested']}\n")
                f.write(f"Champion Model: {ranked_results['best_model']}\n")
                f.write(f"Dataset Size: {self.df.shape[0]} samples, {self.df.shape[1]-1} features\n\n")
                
                f.write("DETAILED MODEL RANKINGS:\n")
                f.write("-" * 70 + "\n")
                for i, ranking in enumerate(ranked_results['rankings'], 1):
                    f.write(f"{i}. {ranking['model']}\n")
                    f.write(f"   Primary Score ({ranked_results['primary_metric']}): {ranking['score']:.6f}\n")
                    f.write(f"   Estimator: {ranking.get('estimator', 'N/A')}\n")
                    f.write(f"   Complete Metrics:\n")
                    for metric, value in ranking['metrics'].items():
                        f.write(f"     • {metric.replace('_', ' ').title():<18}: {value:.6f}\n")
                    f.write("\n")
                
                # Performance analysis
                if len(ranked_results['rankings']) > 1:
                    best_score = ranked_results['rankings'][0]['score']
                    worst_score = ranked_results['rankings'][-1]['score']
                    score_range = best_score - worst_score
                    
                    f.write("PERFORMANCE ANALYSIS:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Performance Range: {score_range:.6f}\n")
                    f.write(f"Best Model Advantage: {(score_range/worst_score*100):.2f}%\n\n")
                    
                    # Identify close competitors
                    close_competitors = []
                    for i in range(1, min(4, len(ranked_results['rankings']))):
                        diff = best_score - ranked_results['rankings'][i]['score']
                        if diff < 0.05:  # Within 5% difference
                            close_competitors.append(ranked_results['rankings'][i]['model'])
                    
                    if close_competitors:
                        f.write("CLOSE COMPETITORS (within 5% of best model):\n")
                        f.write("-" * 50 + "\n")
                        for comp in close_competitors:
                            f.write(f"• {comp}\n")
                        f.write("\nConsider ensemble methods combining these high-performing models.\n\n")
                
                # Failed models section
                failed_models = [name for name, res in all_results.items() if 'error' in res]
                if failed_models:
                    f.write("FAILED MODELS:\n")
                    f.write("-" * 20 + "\n")
                    for model_name in failed_models:
                        error_msg = all_results[model_name].get('error', 'Unknown error')
                        f.write(f"• {model_name.replace('_', ' ').title()}: {error_msg}\n")
                    f.write("\n")
                
                # Final recommendations
                f.write("DEPLOYMENT RECOMMENDATIONS:\n")
                f.write("-" * 40 + "\n")
                recommendations = self._generate_deployment_recommendations(ranked_results)
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")
                    
            print(f"Comprehensive comparison report saved at: {report_path}")
        except Exception as e:
            print(f"Warning: Error creating comparison report: {e}")

    def _generate_deployment_recommendations(self, ranked_results: Dict) -> List[str]:
        """Generate deployment-specific recommendations"""
        recommendations = []
        
        if ranked_results['total_models_tested'] > 1:
            best_score = ranked_results['rankings'][0]['score']
            
            if self.task_type == 'classification' and best_score > 0.9:
                recommendations.append("Model performance exceeds 90% - ready for production deployment")
            elif self.task_type == 'regression' and best_score > 0.8:
                recommendations.append("R² score above 0.8 indicates strong predictive capability")
            else:
                recommendations.append("Consider additional model tuning or feature engineering before deployment")
            
            # Check for model diversity
            unique_estimators = set([r.get('estimator', 'Unknown') for r in ranked_results['rankings'][:3]])
            if len(unique_estimators) > 2:
                recommendations.append("Diverse top-performing algorithms suggest ensemble approach may be beneficial")
        
        recommendations.extend([
            "Implement proper model monitoring and drift detection in production",
            "Set up automated retraining pipeline for model maintenance",
            "Consider A/B testing framework for model performance validation"
        ])
        
        return recommendations

    # [Previous methods continue...]
    def load_model(self, model_path: str):
        """Load saved model from pickle file"""
        try:
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            print(f"Model successfully loaded from: {model_path}")
            return loaded_model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def predict(self, X_new: Union[pd.DataFrame, np.ndarray], model_path: Optional[str] = None):
        """Predict with trained model"""
        if model_path:
            model = self.load_model(model_path)
        elif self.flaml_model:
            model = self.flaml_model
        else:
            raise ValueError("No model available. Train model first or specify model_path.")
        try:
            predictions = model.predict(X_new)
            print(f"Prediction successful for {len(X_new)} samples")
            if self.task_type == "classification" and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_new)
                return {'predictions': predictions, 'probabilities': probabilities}
            else:
                return {'predictions': predictions}
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def get_model_info(self) -> Dict:
        """Get detailed information about trained model"""
        if not self.flaml_model:
            return {"error": "Model not trained yet. Run fit() first."}
        info = {
            'best_estimator': self.flaml_model.best_estimator,
            'best_config': self.flaml_model.best_config,
            'task_type': self.task_type,
            'training_duration': getattr(self.flaml_model, 'training_duration', 'Unknown'),
            'classes_': getattr(self.flaml_model, 'classes_', None),
            'feature_names': list(self.X_train.columns) if hasattr(self, 'X_train') else None
        }
        return info

    def export_results_to_csv(self, output_dir: str = "noventis_output"):
        """Export prediction results and metrics to CSV"""
        if not hasattr(self, 'results') or not self.results:
            print("No results to export. Run fit() first.")
            return
        
        best_model_res_info = self.results['model_comparison']['rankings'][0]
        best_model_name = best_model_res_info['model']
        best_model_data = self.results[best_model_name]
        try:
            os.makedirs(output_dir, exist_ok=True)
            predictions_df = pd.DataFrame({'actual': best_model_data['actual'], 'predicted': best_model_data['predictions']})
            if best_model_data.get('prediction_proba') is not None:
                proba_cols = [f'prob_class_{i}' for i in range(best_model_data['prediction_proba'].shape[1])]
                proba_df = pd.DataFrame(best_model_data['prediction_proba'], columns=proba_cols)
                predictions_df = pd.concat([predictions_df, proba_df], axis=1)
            pred_path = os.path.join(output_dir, 'predictions.csv')
            predictions_df.to_csv(pred_path, index=False)
            
            metrics_df = pd.DataFrame([best_model_data['metrics']])
            metrics_path = os.path.join(output_dir, 'metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            
            if best_model_data.get('feature_importance') is not None:
                fi_path = os.path.join(output_dir, 'feature_importance.csv')
                best_model_data['feature_importance'].to_csv(fi_path, index=False)
            
            print(f"Results successfully exported to directory: {output_dir}")
        except Exception as e:
            print(f"Error exporting to CSV: {e}")

    def get_hyperparameter_suggestions(self, estimator_name: str = None) -> Dict:
        """Get hyperparameter suggestions for specific model"""
        if not estimator_name and self.flaml_model:
            estimator_name = self.flaml_model.best_estimator
        suggestions = { 'lgbm': {}, 'xgboost': {}, 'rf': {} }
        if estimator_name in suggestions:
            return suggestions[estimator_name].get(self.task_type, {})
        else:
            return {"message": f"No hyperparameter suggestions available for {estimator_name}"}

    # Enhanced HTML Report Generation Methods
    def run_and_generate_report(self) -> HTML:
        """
        Main entry point for running COMPLETE pipeline:
        1. Train and evaluate models (using your fit() method).
        2. Create visualizations in base64 format for report.
        3. Generate and display HTML report.
        """
        print("="*60)
        print("Starting Complete NoventisAutoML Pipeline")
        print("="*60)

        print("\n--- Step 1: Running Training and Evaluation ---")
        self.fit(time_budget=self.time_budget, metric=self.metrics)
        
        print("\n--- Step 2: Creating Visualizations for Report ---")
        if self.explain:
            self._generate_visualizations_b64()
        
        print("\n--- Step 3: Generating HTML Report ---")
        report = self.generate_html_report()
        
        print("\nProcess completed! Report displayed below.")
        return report

    def _generate_visualizations_b64(self):
        """Create all visualizations and save as base64 strings for HTML embedding"""
        if not self.results or 'model_comparison' not in self.results:
            print("Warning: No results available for visualization.")
            return

        comp_res = self.results['model_comparison']
        if not comp_res['rankings']:
            print("Warning: No successfully ranked models available for visualization.")
            return

        best_model_name = comp_res['best_model_name']
        best_model_data = self.results.get(best_model_name) or self.results.get(best_model_name.replace(' ', '_').lower())

        # 1. Model Comparison Visualization
        if len(comp_res['rankings']) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            df_ranks = pd.DataFrame(comp_res['rankings'])
            
            # Create gradient color map
            colors = plt.cm.viridis(np.linspace(0, 1, len(df_ranks)))
            bars = ax.barh(df_ranks['model'], df_ranks['score'], color=colors)
            
            ax.set_xlabel(comp_res['primary_metric'].replace('_', ' ').title(), fontsize=14)
            ax.set_ylabel('Model', fontsize=14)
            ax.set_title('Model Performance Ranking', fontsize=16, fontweight='bold')
            
            # Add value labels
            for bar, score in zip(bars, df_ranks['score']):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{score:.4f}', va='center', ha='left', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            self.visualizations_b64['model_comparison'] = plot_to_base64(fig)
        
        # 2. Best Model Task-Specific Visualizations
        if best_model_data:
            if self.task_type == 'classification':
                # Enhanced Confusion Matrix
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                cm = confusion_matrix(best_model_data['actual'], best_model_data['predictions'])
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax1)
                ax1.set_title(f'Normalized Confusion Matrix\n{best_model_name}', fontweight='bold')
                ax1.set_xlabel('Predicted')
                ax1.set_ylabel('Actual')
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax2)
                ax2.set_title(f'Raw Confusion Matrix\n{best_model_name}', fontweight='bold')
                ax2.set_xlabel('Predicted')
                ax2.set_ylabel('Actual')
                
                self.visualizations_b64['confusion_matrix'] = plot_to_base64(fig)
                
                # ROC Curve for binary classification
                if (best_model_data.get('prediction_proba') is not None and 
                    len(np.unique(best_model_data['actual'])) == 2):
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    y_proba = (best_model_data['prediction_proba'][:, 1] 
                             if best_model_data['prediction_proba'].shape[1] > 1 
                             else best_model_data['prediction_proba'])
                    
                    fpr, tpr, _ = roc_curve(best_model_data['actual'], y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    ax.plot(fpr, tpr, color='darkorange', lw=3, 
                           label=f'ROC Curve (AUC = {roc_auc:.3f})')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                           label='Random Classifier')
                    
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate', fontsize=12)
                    ax.set_ylabel('True Positive Rate', fontsize=12)
                    ax.set_title(f'ROC Curve Analysis\n{best_model_name}', fontsize=14, fontweight='bold')
                    ax.legend(loc="lower right")
                    ax.grid(True, alpha=0.3)
                    
                    self.visualizations_b64['roc_curve'] = plot_to_base64(fig)
                
            else:  # Regression
                # Enhanced Predictions vs Actual with residual analysis
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                
                # Main scatter plot
                actual = best_model_data['actual']
                predicted = best_model_data['predictions']
                
                ax1.scatter(actual, predicted, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                min_val = min(min(actual), min(predicted))
                max_val = max(max(actual), max(predicted))
                perfect_line = np.linspace(min_val, max_val, 100)
                ax1.plot(perfect_line, perfect_line, 'r--', lw=2, label='Perfect Prediction')
                
                # Add trend line
                z = np.polyfit(actual, predicted, 1)
                p = np.poly1d(z)
                ax1.plot(actual, p(actual), "g--", alpha=0.8, lw=2, label='Trend Line')
                
                r2 = best_model_data['metrics'].get('r2_score', 0)
                ax1.set_title(f'Predictions vs Actual\n{best_model_name} (R² = {r2:.3f})', fontweight='bold')
                ax1.set_xlabel('Actual Values')
                ax1.set_ylabel('Predicted Values')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Residuals plot
                residuals = np.array(actual) - np.array(predicted)
                ax2.scatter(predicted, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
                ax2.set_title('Residuals Analysis', fontweight='bold')
                ax2.set_xlabel('Predicted Values')
                ax2.set_ylabel('Residuals')
                ax2.grid(True, alpha=0.3)
                
                # Residuals distribution
                ax3.hist(residuals, bins=25, alpha=0.7, color='skyblue', edgecolor='black')
                ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
                ax3.set_title('Residuals Distribution', fontweight='bold')
                ax3.set_xlabel('Residuals')
                ax3.set_ylabel('Frequency')
                
                # Error metrics visualization
                metrics = best_model_data['metrics']
                metric_names = list(metrics.keys())
                metric_values = list(metrics.values())
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(metric_names)))
                bars = ax4.bar(metric_names, metric_values, color=colors, alpha=0.8)
                ax4.set_title('Model Metrics', fontweight='bold')
                ax4.set_ylabel('Score')
                ax4.tick_params(axis='x', rotation=45)
                
                for bar, value in zip(bars, metric_values):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric_values) * 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                self.visualizations_b64['regression_analysis'] = plot_to_base64(fig)
        
        # 3. Feature Importance Visualization (if available)
        if best_model_data and best_model_data.get('feature_importance') is not None:
            fig, ax = plt.subplots(figsize=(12, 10))
            fi_data = best_model_data['feature_importance'].head(20)
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(fi_data)))
            bars = ax.barh(fi_data['feature'], fi_data['importance'], color=colors)
            
            ax.set_title(f'Top 20 Feature Importance\n{best_model_name}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Importance Score', fontsize=12)
            ax.set_ylabel('Features', fontsize=12)
            
            # Add value labels
            for bar, importance in zip(bars, fi_data['importance']):
                ax.text(bar.get_width() + max(fi_data['importance']) * 0.01, 
                       bar.get_y() + bar.get_height()/2, f'{importance:.4f}', 
                       va='center', ha='left', fontweight='bold')
            
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            self.visualizations_b64['feature_importance'] = plot_to_base64(fig)

        # 4. Training History (AutoML only)
        if ('AutoML' in self.results and 
            self.results['AutoML'].get('training_history') is not None):
            
            history = self.results['AutoML']['training_history']
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.plot(history['time_seconds'], history['best_validation_loss'], 
                   marker='o', linestyle='-', linewidth=2, markersize=6, 
                   color='blue', alpha=0.8)
            
            ax.set_title('AutoML Training Progress Over Time', fontsize=16, fontweight='bold')
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Best Validation Loss', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add annotations for key points
            min_loss_idx = history['best_validation_loss'].idxmin()
            min_loss_time = history.loc[min_loss_idx, 'time_seconds']
            min_loss_value = history.loc[min_loss_idx, 'best_validation_loss']
            
            ax.annotate(f'Best: {min_loss_value:.4f}\nat {min_loss_time:.1f}s', 
                       xy=(min_loss_time, min_loss_value),
                       xytext=(min_loss_time + max(history['time_seconds']) * 0.1, 
                              min_loss_value + (max(history['best_validation_loss']) - min_loss_value) * 0.1),
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                       fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            self.visualizations_b64['training_history'] = plot_to_base64(fig)

        # 5. Comprehensive Metrics Comparison (if multiple models)
        if len(comp_res['rankings']) > 1:
            all_metrics = {}
            for ranking in comp_res['rankings'][:5]:  # Top 5 models
                model_name = ranking['model']
                all_metrics[model_name] = ranking['metrics']
            
            metrics_df = pd.DataFrame(all_metrics).T
            
            if len(metrics_df.columns) > 1:
                fig, ax = plt.subplots(figsize=(14, 8))
                
                sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                           cbar_kws={'label': 'Score'}, linewidths=0.5)
                
                ax.set_title('Model Performance Heatmap\nAll Metrics Comparison', 
                            fontsize=16, fontweight='bold')
                ax.set_xlabel('Metrics', fontsize=12)
                ax.set_ylabel('Models', fontsize=12)
                ax.tick_params(axis='x', rotation=45)
                ax.tick_params(axis='y', rotation=0)
                
                plt.tight_layout()
                self.visualizations_b64['metrics_heatmap'] = plot_to_base64(fig)
                
        print("Enhanced visualizations successfully created for dashboard.")

    def _generate_summary_panel(self) -> str:
        """Generate executive summary panel with key insights"""
        if ('model_comparison' not in self.results or 
            not self.results['model_comparison']['rankings']):
            return "<p>Failed to generate summary. No models were successfully trained.</p>"
        
        comp_res = self.results['model_comparison']
        best_model = comp_res['rankings'][0]
        metric_name = comp_res['primary_metric'].replace('_', ' ').title()
        total_models = comp_res['total_models_tested']

        # Key performance indicators
        kpi_html = f"""
        <div class="kpi-grid">
            <div class="kpi-card champion">
                <h4>Champion Model</h4>
                <p class="kpi-value">{best_model['model']}</p>
                <span class="kpi-label">Best Performer</span>
            </div>
            <div class="kpi-card score">
                <h4>Performance Score</h4>
                <p class="kpi-value">{best_model['score']:.4f}</p>
                <span class="kpi-label">{metric_name}</span>
            </div>
            <div class="kpi-card task">
                <h4>Task Type</h4>
                <p class="kpi-value">{self.task_type.title()}</p>
                <span class="kpi-label">ML Problem</span>
            </div>
            <div class="kpi-card models">
                <h4>Models Evaluated</h4>
                <p class="kpi-value">{total_models}</p>
                <span class="kpi-label">Total Tested</span>
            </div>
        </div>
        """
        
        # Detailed metrics table
        best_metrics_df = pd.DataFrame([best_model['metrics']]).T.reset_index()
        best_metrics_df.columns = ['Metric', 'Score']
        best_metrics_df['Status'] = best_metrics_df.apply(
            lambda row: self._get_metric_status(row['Metric'].lower(), row['Score'], self.task_type), 
            axis=1
        )
        
        metrics_table_html = f"""
        <h3>Champion Model - Detailed Performance Metrics</h3>
        <div class='table-scroll-wrapper'>
            {best_metrics_df.to_html(classes='styled-table', index=False, escape=False)}
        </div>
        """
        
        # Performance insights
        insights = self._generate_model_insights(self.results[best_model['model']], self.task_type)
        insights_html = "<h3>Key Performance Insights</h3><div class='insights-container'>"
        for insight in insights:
            insights_html += f"<div class='insight-item'>{insight}</div>"
        insights_html += "</div>"
        
        return kpi_html + metrics_table_html + insights_html

    def _generate_comparison_panel(self) -> str:
        """Generate comprehensive model comparison panel"""
        if len(self.results.get('model_comparison', {}).get('rankings', [])) < 2:
            return "<h4>No model comparison available (only one model trained).</h4>"
        
        # Model comparison chart
        chart_html = ""
        if 'model_comparison' in self.visualizations_b64:
            chart_html = f"""
            <div class='visualization-container'>
                <img src='{self.visualizations_b64.get('model_comparison', '')}' alt='Model Performance Comparison'>
            </div>
            """
        
        # Rankings table with enhanced information
        rankings_df = pd.DataFrame(self.results['model_comparison']['rankings'])
        rankings_df['Rank'] = range(1, len(rankings_df) + 1)
        rankings_df = rankings_df[['Rank', 'model', 'score', 'estimator']].copy()
        rankings_df.columns = ['Rank', 'Model', 'Score', 'Algorithm']
        
        rankings_table_html = f"""
        <h3>Complete Model Rankings</h3>
        <div class='table-scroll-wrapper'>
            {rankings_df.to_html(classes='styled-table', index=False)}
        </div>
        """
        
        # Performance gap analysis
        if len(rankings_df) > 1:
            best_score = rankings_df.iloc[0]['Score']
            worst_score = rankings_df.iloc[-1]['Score']
            score_range = best_score - worst_score
            
            gap_analysis = f"""
            <h3>Performance Analysis</h3>
            <div class='analysis-grid'>
                <div class='analysis-item'>
                    <strong>Performance Range:</strong> {score_range:.4f}
                </div>
                <div class='analysis-item'>
                    <strong>Best Model Advantage:</strong> {(score_range/worst_score*100):.2f}%
                </div>
            </div>
            """
        else:
            gap_analysis = ""
        
        # Metrics heatmap
        heatmap_html = ""
        if 'metrics_heatmap' in self.visualizations_b64:
            heatmap_html = f"""
            <h3>Multi-Metric Performance Heatmap</h3>
            <div class='visualization-container'>
                <img src='{self.visualizations_b64.get('metrics_heatmap', '')}' alt='Metrics Heatmap'>
            </div>
            """
        
        return chart_html + rankings_table_html + gap_analysis + heatmap_html

    def _generate_analysis_panel(self) -> str:
        """Generate detailed model analysis panel"""
        if 'model_comparison' not in self.results:
            return "<h4>Analysis not available.</h4>"
        
        best_model_name = self.results['model_comparison']['best_model_name']
        best_model_data = self.results.get(best_model_name) or self.results.get(best_model_name.replace(' ', '_').lower())
        
        if not best_model_data:
            return "<h4>Best model data not available for analysis.</h4>"
        
        content_sections = []
        
        # Task-specific visualizations
        if self.task_type == 'classification':
            if 'confusion_matrix' in self.visualizations_b64:
                content_sections.append(f"""
                <div class='analysis-section'>
                    <h3>Classification Performance Analysis</h3>
                    <div class='visualization-container'>
                        <img src='{self.visualizations_b64.get('confusion_matrix', '')}' alt='Confusion Matrix Analysis'>
                    </div>
                </div>
                """)
            
            if 'roc_curve' in self.visualizations_b64:
                content_sections.append(f"""
                <div class='analysis-section'>
                    <h3>ROC Curve Analysis</h3>
                    <div class='visualization-container'>
                        <img src='{self.visualizations_b64.get('roc_curve', '')}' alt='ROC Curve'>
                    </div>
                </div>
                """)
        else:  # regression
            if 'regression_analysis' in self.visualizations_b64:
                content_sections.append(f"""
                <div class='analysis-section'>
                    <h3>Regression Performance Analysis</h3>
                    <div class='visualization-container'>
                        <img src='{self.visualizations_b64.get('regression_analysis', '')}' alt='Regression Analysis'>
                    </div>
                </div>
                """)
        
        # Feature importance
        if 'feature_importance' in self.visualizations_b64:
            content_sections.append(f"""
            <div class='analysis-section'>
                <h3>Feature Importance Analysis</h3>
                <div class='visualization-container'>
                    <img src='{self.visualizations_b64.get('feature_importance', '')}' alt='Feature Importance'>
                </div>
            </div>
            """)
        
        return ''.join(content_sections) if content_sections else "<h4>No detailed analysis available.</h4>"

    def _generate_training_insights_panel(self) -> str:
        """Generate training process insights panel"""
        content_sections = []
        
        # Training history visualization
        if 'training_history' in self.visualizations_b64:
            content_sections.append(f"""
            <div class='training-section'>
                <h3>AutoML Training Progress</h3>
                <div class='visualization-container'>
                    <img src='{self.visualizations_b64.get('training_history', '')}' alt='Training History'>
                </div>
            </div>
            """)
        
        # Training insights
        if 'AutoML' in self.results:
            automl_data = self.results['AutoML']
            insights_html = "<div class='training-insights'><h3>Training Configuration & Results</h3>"
            
            if automl_data.get('best_estimator'):
                insights_html += f"<div class='insight-item'><strong>Best Algorithm:</strong> {automl_data['best_estimator']}</div>"
            
            if automl_data.get('best_config'):
                insights_html += f"<div class='insight-item'><strong>Optimal Configuration:</strong> Custom hyperparameters discovered</div>"
            
            insights_html += f"<div class='insight-item'><strong>Training Budget:</strong> {self.time_budget} seconds</div>"
            insights_html += f"<div class='insight-item'><strong>Task Complexity:</strong> {self.task_type.title()} with {self.df.shape[1]-1} features</div>"
            insights_html += "</div>"
            
            content_sections.append(insights_html)
        
        # Recommendations
        recommendations = self._generate_recommendations(self.results)
        if recommendations:
            rec_html = "<div class='recommendations'><h3>Improvement Recommendations</h3>"
            for i, rec in enumerate(recommendations, 1):
                rec_html += f"<div class='recommendation-item'>{i}. {rec}</div>"
            rec_html += "</div>"
            content_sections.append(rec_html)
        
        return ''.join(content_sections) if content_sections else "<h4>Training insights not available (manual models only).</h4>"

    def generate_html_report(self, report_height: int = 900) -> HTML:
        """Generate comprehensive HTML dashboard report"""
        # Configure tabs based on available data
        tabs_config = [
            {'id': 'summary', 'title': 'Executive Summary', 'content_func': self._generate_summary_panel}
        ]
        
        if self.compare and len(self.results.get('model_comparison', {}).get('rankings', [])) > 1:
            tabs_config.append({'id': 'comparison', 'title': 'Model Comparison', 'content_func': self._generate_comparison_panel})
        
        if self.explain:
            tabs_config.append({'id': 'analysis', 'title': 'Performance Analysis', 'content_func': self._generate_analysis_panel})
        
        if self.use_automl:
            tabs_config.append({'id': 'training', 'title': 'Training Insights', 'content_func': self._generate_training_insights_panel})

        # Generate navigation and content
        navbar_html, main_content_html = "", ""
        for i, tab in enumerate(tabs_config):
            active_class = 'active' if i == 0 else ''
            navbar_html += f"""
            <button class="nav-btn {active_class}" onclick="showTab(event, '{tab['id']}', '{self.report_id}')">
                {tab['title']}
            </button>"""
            
            content = tab['content_func']()
            main_content_html += f"""
            <section id="{tab['id']}-{self.report_id}" class="content-section {active_class}">
                <h2>{tab['title']}</h2>
                {content}
            </section>"""
        
        # Enhanced CSS with modern styling
        enhanced_css = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
            
            :root {
                --bg-primary: #0f1419;
                --bg-secondary: #161b22;
                --bg-tertiary: #21262d;
                --border-color: #30363d;
                --text-primary: #e6edf3;
                --text-secondary: #7d8590;
                --accent-blue: #58a6ff;
                --accent-green: #3fb950;
                --accent-orange: #ff7b72;
                --accent-purple: #a5a2ff;
                --gradient-primary: linear-gradient(135deg, #58a6ff 0%, #a5a2ff 100%);
                --gradient-secondary: linear-gradient(135deg, #3fb950 0%, #58a6ff 100%);
                --shadow-sm: 0 1px 3px rgba(0,0,0,0.12);
                --shadow-md: 0 4px 6px rgba(0,0,0,0.16);
                --shadow-lg: 0 10px 15px rgba(0,0,0,0.2);
            }
            
            body {
                font-family: 'Inter', sans-serif;
                background: var(--bg-primary);
                color: var(--text-primary);
                margin: 0;
                padding: 0;
                line-height: 1.6;
            }
            
            .report-frame {
                height: {report_height}px;
                width: 100%;
                border: 1px solid var(--border-color);
                border-radius: 12px;
                overflow: hidden;
                background: var(--bg-primary);
                box-shadow: var(--shadow-lg);
            }
            
            .container {
                width: 100%;
                height: 100%;
                overflow: auto;
                background: var(--bg-primary);
            }
            
            header {
                position: sticky;
                top: 0;
                z-index: 100;
                padding: 2rem 3rem;
                border-bottom: 1px solid var(--border-color);
                background: rgba(15, 20, 25, 0.95);
                backdrop-filter: blur(20px);
            }
            
            header h1 {
                font-family: 'Inter', sans-serif;
                font-size: 2.5rem;
                font-weight: 700;
                margin: 0;
                background: var(--gradient-primary);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            header p {
                margin: 0.5rem 0 0;
                color: var(--text-secondary);
                font-size: 1.1rem;
                font-weight: 400;
            }
            
            .navbar {
                position: sticky;
                top: 120px;
                z-index: 90;
                display: flex;
                flex-wrap: wrap;
                background: var(--bg-secondary);
                padding: 0 3rem;
                border-bottom: 1px solid var(--border-color);
                gap: 0.5rem;
            }
            
            .nav-btn {
                background: none;
                border: none;
                color: var(--text-secondary);
                padding: 1rem 1.5rem;
                font-size: 0.95rem;
                font-weight: 500;
                cursor: pointer;
                border-bottom: 3px solid transparent;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
            }
            
            .nav-btn:hover {
                color: var(--text-primary);
                background: rgba(88, 166, 255, 0.1);
            }
            
            .nav-btn.active {
                color: var(--accent-blue);
                border-bottom-color: var(--accent-blue);
                font-weight: 600;
            }
            
            main {
                padding: 3rem;
                max-width: 1400px;
                margin: 0 auto;
            }
            
            .content-section {
                display: none;
                animation: fadeIn 0.5s ease-in-out;
            }
            
            .content-section.active {
                display: block;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            h2 {
                font-size: 2.2rem;
                font-weight: 600;
                color: var(--text-primary);
                border-bottom: 2px solid var(--accent-blue);
                padding-bottom: 1rem;
                margin: 0 0 2rem 0;
            }
            
            h3 {
                font-size: 1.5rem;
                font-weight: 600;
                color: var(--accent-green);
                margin: 2rem 0 1rem 0;
            }
            
            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 1.5rem;
                margin: 2rem 0;
            }
            
            .kpi-card {
                background: var(--bg-secondary);
                padding: 2rem;
                border-radius: 12px;
                border: 1px solid var(--border-color);
                position: relative;
                overflow: hidden;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            .kpi-card:hover {
                transform: translateY(-2px);
                box-shadow: var(--shadow-md);
            }
            
            .kpi-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: var(--gradient-primary);
            }
            
            .kpi-card h4 {
                margin: 0 0 1rem 0;
                color: var(--text-secondary);
                font-size: 0.9rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            
            .kpi-value {
                font-size: 2.5rem;
                font-weight: 700;
                margin: 0;
                color: var(--text-primary);
                font-family: 'JetBrains Mono', monospace;
            }
            
            .kpi-label {
                color: var(--text-secondary);
                font-size: 0.85rem;
                font-weight: 400;
            }
            
            .visualization-container {
                background: var(--bg-secondary);
                padding: 2rem;
                border-radius: 12px;
                border: 1px solid var(--border-color);
                margin: 1.5rem 0;
                text-align: center;
            }
            
            .visualization-container img {
                max-width: 100%;
                height: auto;
                border-radius: 8px;
            }
            
            .analysis-section {
                margin: 2rem 0;
                padding: 2rem;
                background: var(--bg-secondary);
                border-radius: 12px;
                border: 1px solid var(--border-color);
            }
            
            .insights-container {
                display: grid;
                gap: 1rem;
                margin: 1rem 0;
            }
            
            .insight-item {
                padding: 1rem;
                background: var(--bg-tertiary);
                border-left: 4px solid var(--accent-green);
                border-radius: 4px;
                font-size: 0.95rem;
                line-height: 1.5;
            }
            
            .table-scroll-wrapper {
                margin: 1rem 0;
                overflow-x: auto;
                border-radius: 8px;
                border: 1px solid var(--border-color);
            }
            
            .styled-table {
                width: 100%;
                color: var(--text-primary);
                background: var(--bg-secondary);
                border-collapse: collapse;
                font-size: 0.9rem;
            }
            
            .styled-table th,
            .styled-table td {
                padding: 1rem;
                text-align: left;
                border-bottom: 1px solid var(--border-color);
                white-space: nowrap;
            }
            
            .styled-table thead th {
                background: var(--bg-tertiary);
                font-weight: 600;
                color: var(--accent-blue);
            }
            
            .styled-table tbody tr:hover {
                background: rgba(88, 166, 255, 0.05);
            }
            
            .analysis-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1rem;
                margin: 1rem 0;
            }
            
            .analysis-item {
                padding: 1rem;
                background: var(--bg-tertiary);
                border-radius: 8px;
                border: 1px solid var(--border-color);
            }
            
            .recommendations {
                margin: 2rem 0;
                padding: 2rem;
                background: var(--bg-secondary);
                border-radius: 12px;
                border: 1px solid var(--border-color);
            }
            
            .recommendation-item {
                padding: 0.8rem 0;
                border-bottom: 1px solid var(--border-color);
                font-size: 0.95rem;
                line-height: 1.5;
            }
            
            .recommendation-item:last-child {
                border-bottom: none;
            }
            
            .training-insights {
                margin: 2rem 0;
                padding: 2rem;
                background: var(--bg-secondary);
                border-radius: 12px;
                border: 1px solid var(--border-color);
            }
            
            /* Responsive design */
            @media (max-width: 768px) {
                header { padding: 1.5rem 2rem; }
                header h1 { font-size: 2rem; }
                .navbar { padding: 0 2rem; }
                main { padding: 2rem; }
                .kpi-grid { grid-template-columns: 1fr; }
                .analysis-grid { grid-template-columns: 1fr; }
            }
        </style>
        """
        
        # Complete HTML template
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>NoventisAutoML - Advanced Analytics Dashboard</title>
            {enhanced_css}
        </head>
        <body>
            <div id="{self.report_id}" class="report-frame">
                <div class="container">
                    <header>
                        <h1>NoventisAutoML Analytics Dashboard</h1>
                        <p>Comprehensive machine learning model analysis and performance insights</p>
                    </header>
                    
                    <nav class="navbar">
                        {navbar_html}
                    </nav>
                    
                    <main>
                        {main_content_html}
                    </main>
                </div>
            </div>
            
            <script>
                function showTab(event, tabName, reportId) {{
                    const reportFrame = document.getElementById(reportId);
                    if (!reportFrame) return;
                    
                    // Hide all content sections
                    reportFrame.querySelectorAll('.content-section').forEach(section => {{
                        section.classList.remove('active');
                    }});
                    
                    // Remove active class from all nav buttons
                    reportFrame.querySelectorAll('.nav-btn').forEach(button => {{
                        button.classList.remove('active');
                    }});
                    
                    // Show selected content section
                    const sectionToShow = reportFrame.querySelector(`#${{tabName}}-${{reportId}}`);
                    if (sectionToShow) {{
                        sectionToShow.classList.add('active');
                    }}
                    
                    // Add active class to clicked button
                    event.currentTarget.classList.add('active');
                }}
                
                // Initialize dashboard
                document.addEventListener('DOMContentLoaded', function() {{
                    console.log('NoventisAutoML Dashboard initialized');
                }});
            </script>
        </body>
        </html>
        """
        
        return HTML(html_template)

    def __repr__(self):
        """Enhanced string representation of NoventisAutoML object"""
        status = "Trained" if self.flaml_model else "Not Trained"
        best_model = getattr(self.flaml_model, 'best_estimator', 'None') if self.flaml_model else 'None'
        
        return f"""
NoventisAutoML Enhanced Analytics Platform
==========================================
Configuration:
  • Task Type: {self.task_type}
  • Target Column: {self.target_column}
  • Training Status: {status}
  • Best Algorithm: {best_model}
  • Dataset Shape: {getattr(self, 'df', pd.DataFrame()).shape}
  • Explanation Mode: {self.explain}
  • Comparison Mode: {self.compare}
  • Time Budget: {self.time_budget}s
  • Output Directory: {self.output_dir}

Features:
  ✓ Automatic task detection
  ✓ Advanced model comparison
  ✓ Interactive HTML dashboard
  ✓ Comprehensive visualizations
  ✓ Performance insights & recommendations
  ✓ Feature importance analysis
  ✓ Model export capabilities
        """