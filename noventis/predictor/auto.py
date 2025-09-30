import pandas as pd
import numpy as np
import os
import pickle
from typing import Union, Optional, Dict, Any, List
import warnings
from IPython.display import HTML
import base64
from io import BytesIO

try:
    from flaml import AutoML as FLAMLAutoML
except ImportError:
    try:
        from flaml.automl import AutoML as FLAMLAutoML
    except ImportError:
        from flaml.automl.automl import AutoML as FLAMLAutoML

from flaml.automl.data import get_output_from_log
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, r2_score, confusion_matrix, accuracy_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from .manual import ManualPredictor

warnings.filterwarnings('ignore')

class NoventisAutoML:
    
    COLORS = {
        'primary_blue': '#58A6FF',
        'primary_orange': '#F78166',
        'success_green': '#28A745',
        'warning_yellow': '#FFC107',
        'bg_dark': '#161B22',
        'text_light': '#C9D1D9',
        'palette': ['#58A6FF', '#F78166', '#28A745', '#FFC107', '#BB86FC', '#03DAC6', '#CF6679']
    }
    
    def __init__(self, data: Union[str, pd.DataFrame], target: str, task: Optional[str] = None, 
                 models: List[str]=None, explain: bool=True, compare: bool=True, metrics: str=None,
                 time_budget: int=60, output_dir: str='Noventis_Results', test_size: float = 0.2, 
                 random_state: int = 42):
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
        self.report_id = f"automl_report_{id(self)}"
        self.flaml_model = None
        self.manual_model = None
        self.results = {}
        self._load_data(data)
        self._setup_data()

    def _load_data(self, data: Union[str, pd.DataFrame]):
        if isinstance(data, str):
            self.df = pd.read_csv(data)
            print(f"Data loaded from file: {data}")
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
            print("Data loaded from DataFrame")
        else:
            raise TypeError("Input must be CSV path or pandas DataFrame")
        print(f"Shape: {self.df.shape}, Columns: {list(self.df.columns)}")

    def _detect_task_type(self) -> str:
        y = self.df[self.target_column]
        unique_values = len(y.unique())
        unique_ratio = unique_values / len(y)
        if pd.api.types.is_numeric_dtype(y):
            return "regression" if unique_values > 25 and unique_ratio >= 0.05 else "classification"
        return "classification"

    def _setup_data(self):
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")
        X, y = self.df.drop(columns=[self.target_column]), self.df[self.target_column]
        if self.task_type is None:
            self.task_type = self._detect_task_type()
            print(f"Task type detected: {self.task_type}")
        stratify = y if self.task_type == "classification" else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify)
        print(f"Train={len(self.X_train)}, Test={len(self.X_test)}")

    def fit(self, time_budget: int = 60, metric: Optional[str] = None) -> Dict:
        print("Starting AutoML process...")
        flaml_metric = self._convert_metric_to_flaml(self.metrics)
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.use_automl:      
            self.flaml_model = FLAMLAutoML(task=self.task_type, metric=flaml_metric, 
                                          seed=self.random_state, verbose=2)
            print(f"Training (Metric: {flaml_metric}, Time: {time_budget}s)")
            self.flaml_model.fit(X_train=self.X_train, y_train=self.y_train,
                               log_file_name=f'{self.output_dir}/flaml.log', time_budget=self.time_budget)
            y_pred = self.flaml_model.predict(self.X_test)
            if self.task_type == "classification":
                metrics = self._eval_classification(self.y_test, y_pred)
                y_pred_proba = self.flaml_model.predict_proba(self.X_test) if hasattr(self.flaml_model, 'predict_proba') else None
            else:
                metrics = self._eval_regression(self.y_test, y_pred)
                y_pred_proba = None
            self.results['AutoML'] = {
                'model': self.flaml_model, 'predictions': y_pred, 'prediction_proba': y_pred_proba,
                'actual': self.y_test, 'metrics': metrics, 'task_type': self.task_type,
                'feature_importance': self._get_feature_importance(),
                'best_estimator': self.flaml_model.best_estimator,
                'best_config': self.flaml_model.best_config,
                'training_history': self._get_training_history(f'{self.output_dir}/flaml.log')
            }
        else:
            predictor = ManualPredictor(model_name=self.model_list, task=self.task_type, 
                                       random_state=self.random_state)
            result = predictor.run_pipeline(self.df, target_column=self.target_column, test_size=self.test_size)
            self.manual_model = predictor
            for model_result in result['all_model_results']:
                if 'error' in model_result:
                    continue
                model_name = model_result['model_name']
                self.results[model_name] = {
                    'model': model_name, 'predictions': model_result['predictions'],
                    'prediction_proba': model_result['prediction_proba'], 'actual': self.y_test,
                    'metrics': model_result['metrics'], 'task_type': self.task_type,
                    'feature_importance': None, 'best_estimator': None, 
                    'best_config': None, 'training_history': None
                }
            predictor.save_model(f'{self.output_dir}/best_model.pkl')

        model_path = os.path.join(self.output_dir, 'best_automl_model.pkl')
        
        if self.compare:
            comparison_results = self.compare_models(output_dir=self.output_dir, models_to_compare=self.model_list)
            self.results['model_comparison'] = comparison_results
            best_manual_model_metrics = comparison_results['rankings'][0]['score']
            automl_metrics = self.results['AutoML']['metrics'][self.metrics]
            if best_manual_model_metrics > automl_metrics:
                internal_model_key = comparison_results['rankings'][0]['model'].replace(' ', '_')
                self.manual_model.save_model(f'{self.output_dir}/best_model.pkl')
                self.results[internal_model_key]['model_path'] = model_path
            else:
                self._save_model(self.flaml_model, model_path)
                self.results['AutoML']['model_path'] = model_path
        else:
            if self.use_automl:
                self._save_model(self.flaml_model, model_path)
                self.results['AutoML']['model_path'] = model_path
            else:
                comparison_results = self.compare_models(output_dir=self.output_dir, models_to_compare=self.model_list)
                self.results['model_comparison'] = comparison_results
                self.manual_model.save_model(f'{self.output_dir}/best_model.pkl')

        if self.explain:
            self.results['visualization_paths'] = self._generate_visualizations(self.results, self.output_dir)
            self._generate_model_summary(self.results, self.output_dir)
        
        return self.results

    def compare_models(self, models_to_compare: Optional[List[str]] = None, output_dir: str = "Noventis_results") -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        all_results = {}
        if models_to_compare is None:
            models_to_compare = ['logistic_regression', 'random_forest', 'xgboost', 'decision_tree', 
                               'lightgbm', 'catboost', 'gradient_boosting'] if self.task_type == 'classification' else \
                              ['linear_regression', 'random_forest', 'xgboost', 'gradient_boosting', 'lightgbm', 'catboost']
            all_results['AutoML'] = {'metrics': self.results.get('AutoML', {}).get('metrics', {}), 
                                    'model_name': 'AutoML', 'best_estimator': 'AutoML'}
        
        if self.manual_model is None:
            predictor = ManualPredictor(model_name=models_to_compare, task=self.task_type, random_state=self.random_state)
            result = predictor.run_pipeline(self.df, target_column=self.target_column, test_size=self.test_size)
            self.manual_model = predictor
            for model_result in result['all_model_results']:
                if 'error' in model_result:
                    continue
                model_name = model_result['model_name']
                self.results[model_name] = {
                    'model': model_name, 'predictions': model_result['predictions'],
                    'prediction_proba': model_result['prediction_proba'], 'actual': self.y_test,
                    'metrics': model_result['metrics'], 'task_type': self.task_type,
                    'feature_importance': None, 'best_estimator': None, 'best_config': None, 'training_history': None
                }

        for model in self.results:
            all_results[model] = self.results[model]
        ranked_results = self._rank_models(all_results)
        if self.compare:
            self._visualize_model_comparison(ranked_results, all_results, output_dir)
            self._generate_comparison_report(ranked_results, all_results, output_dir)
        return ranked_results

    def _eval_classification(self, y_true, y_pred) -> Dict:
        return {'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0)}

    def _eval_regression(self, y_true, y_pred) -> Dict:
        mse = mean_squared_error(y_true, y_pred)
        return {'mae': mean_absolute_error(y_true, y_pred), 'mse': mse, 
                'rmse': np.sqrt(mse), 'r2_score': r2_score(y_true, y_pred)}

    def _convert_metric_to_flaml(self, metric: Optional[str]) -> str:
        if metric is None:
            return "macro_f1" if self.task_type == "classification" else "r2"
        return 'macro_f1' if metric == 'f1_score' else ('r2' if metric == 'r2_score' else metric)

    def _get_feature_importance(self) -> Optional[pd.DataFrame]:
        if self.use_automl:
            try:
                if hasattr(self.flaml_model, 'feature_importances_'):
                    return pd.DataFrame({'feature': self.X_train.columns, 
                                       'importance': self.flaml_model.feature_importances_}).sort_values('importance', ascending=False)
            except Exception as e:
                pass
        return None

    def _get_training_history(self, log_file) -> Optional[pd.DataFrame]:
        if self.use_automl:
            try:
                if os.path.exists(log_file):
                    time_h, loss_h, _, _, _ = get_output_from_log(filename=log_file, time_budget=float('inf'))
                    return pd.DataFrame({'time_seconds': time_h, 'best_validation_loss': loss_h})
            except:
                pass
        return None

    def _save_model(self, model, path):
        try:
            with open(path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            print(f"Error saving model: {e}")

    def _set_plot_style(self):
        plt.style.use('dark_background')
        plt.rcParams['figure.facecolor'] = self.COLORS['bg_dark']
        plt.rcParams['axes.facecolor'] = self.COLORS['bg_dark']
        plt.rcParams['text.color'] = self.COLORS['text_light']
        plt.rcParams['axes.labelcolor'] = self.COLORS['text_light']
        plt.rcParams['xtick.color'] = self.COLORS['text_light']
        plt.rcParams['ytick.color'] = self.COLORS['text_light']

    def _generate_visualizations(self, results, output_dir: str) -> List[str]:
        paths = []
        self._set_plot_style()
        
        if self.use_automl and not self.compare:
            best_model_res = results['AutoML']
        else:
            best_model_name = results['model_comparison']['rankings'][0]['model']
            best_model_res = results[best_model_name.replace(' ', '_')]
        
        try:
            if self.use_automl and best_model_res.get('feature_importance') is not None:
                paths.append(self._plot_feature_importance(best_model_res['feature_importance'], output_dir))
                
            if self.use_automl and best_model_res.get('training_history') is not None:
                paths.append(self._plot_training_history(best_model_res['training_history'], output_dir))

            if self.task_type == 'classification':
                paths.extend(self._generate_classification_plots(best_model_res, output_dir))
            else:
                paths.extend(self._generate_regression_plots(best_model_res, output_dir))
        except Exception as e:
            print(f"Error creating visualizations: {e}")
        return paths

    def _plot_feature_importance(self, fi_df, output_dir):
        fig, ax = plt.subplots(figsize=(12, 8))
        top_features = fi_df.head(20)
        bars = ax.barh(top_features['feature'], top_features['importance'], 
                       color=self.COLORS['primary_blue'], edgecolor=self.COLORS['primary_orange'], linewidth=1.5)
        ax.set_xlabel('Importance Score', fontsize=12, color=self.COLORS['text_light'])
        ax.set_ylabel('Features', fontsize=12, color=self.COLORS['text_light'])
        ax.set_title('Top 20 Feature Importance', fontsize=16, fontweight='bold', color=self.COLORS['primary_orange'])
        ax.grid(axis='x', alpha=0.2, color=self.COLORS['text_light'])
        plt.tight_layout()
        path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.COLORS['bg_dark'])
        plt.close()
        return path

    def _plot_training_history(self, history, output_dir):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(history['time_seconds'], history['best_validation_loss'], 
                marker='o', linestyle='-', color=self.COLORS['primary_blue'], linewidth=2.5, markersize=6)
        ax.fill_between(history['time_seconds'], history['best_validation_loss'], 
                        alpha=0.3, color=self.COLORS['primary_blue'])
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Best Validation Loss', fontsize=12)
        ax.set_title('AutoML Training Progress', fontsize=16, fontweight='bold', color=self.COLORS['primary_orange'])
        ax.grid(True, alpha=0.2, color=self.COLORS['text_light'])
        plt.tight_layout()
        path = os.path.join(output_dir, 'training_history.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.COLORS['bg_dark'])
        plt.close()
        return path

    def _generate_classification_plots(self, results, output_dir: str) -> List[str]:
        paths = []
        
        fig, ax = plt.subplots(figsize=(10, 8))
        cm = confusion_matrix(results['actual'], results['predictions'])
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                   cbar_kws={'label': 'Percentage'})
        ax.set_title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', color=self.COLORS['primary_orange'])
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        plt.tight_layout()
        path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.COLORS['bg_dark'])
        paths.append(path)
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = results['metrics']
        bars = ax.bar(metrics.keys(), metrics.values(), color=self.COLORS['palette'][:len(metrics)], 
                     edgecolor=self.COLORS['primary_orange'], linewidth=2)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.set_title('Classification Metrics', fontsize=16, fontweight='bold', color=self.COLORS['primary_orange'])
        for bar, value in zip(bars, metrics.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.2, color=self.COLORS['text_light'])
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = os.path.join(output_dir, 'classification_metrics.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.COLORS['bg_dark'])
        paths.append(path)
        plt.close()
        
        if results.get('prediction_proba') is not None and len(np.unique(results['actual'])) == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            y_true = results['actual']
            y_proba = results['prediction_proba'][:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr, color=self.COLORS['primary_blue'], lw=3, label=f'ROC (AUC = {roc_auc:.3f})')
            ax1.plot([0, 1], [0, 1], color=self.COLORS['primary_orange'], lw=2, linestyle='--', label='Random')
            ax1.fill_between(fpr, tpr, alpha=0.3, color=self.COLORS['primary_blue'])
            ax1.set_xlabel('False Positive Rate', fontsize=12)
            ax1.set_ylabel('True Positive Rate', fontsize=12)
            ax1.set_title('ROC Curve', fontsize=14, fontweight='bold', color=self.COLORS['primary_orange'])
            ax1.legend(loc='lower right')
            ax1.grid(alpha=0.2)
            
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            ax2.plot(recall, precision, color=self.COLORS['success_green'], lw=3, label='PR Curve')
            ax2.fill_between(recall, precision, alpha=0.3, color=self.COLORS['success_green'])
            ax2.set_xlabel('Recall', fontsize=12)
            ax2.set_ylabel('Precision', fontsize=12)
            ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', color=self.COLORS['primary_orange'])
            ax2.legend(loc='lower left')
            ax2.grid(alpha=0.2)
            plt.tight_layout()
            path = os.path.join(output_dir, 'roc_pr_curves.png')
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.COLORS['bg_dark'])
            paths.append(path)
            plt.close()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        class_counts = pd.Series(results['actual']).value_counts()
        axes[0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                   colors=self.COLORS['palette'], startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
        axes[0].set_title('Actual Distribution', fontsize=14, fontweight='bold', color=self.COLORS['primary_orange'])
        
        pred_counts = pd.Series(results['predictions']).value_counts()
        axes[1].pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%',
                   colors=self.COLORS['palette'], startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
        axes[1].set_title('Predicted Distribution', fontsize=14, fontweight='bold', color=self.COLORS['primary_orange'])
        plt.tight_layout()
        path = os.path.join(output_dir, 'class_distribution.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.COLORS['bg_dark'])
        paths.append(path)
        plt.close()
        
        return paths

    def _generate_regression_plots(self, results, output_dir: str) -> List[str]:
        paths = []
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(results['actual'], results['predictions'], alpha=0.6, 
                  c=self.COLORS['primary_blue'], edgecolors=self.COLORS['primary_orange'], s=80, linewidth=1.5)
        min_val = min(min(results['actual']), min(results['predictions']))
        max_val = max(max(results['actual']), max(results['predictions']))
        perfect_line = np.linspace(min_val, max_val, 100)
        ax.plot(perfect_line, perfect_line, color=self.COLORS['primary_orange'], 
               linestyle='--', linewidth=3, label='Perfect Prediction')
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title('Predictions vs Actual', fontsize=16, fontweight='bold', color=self.COLORS['primary_orange'])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.2)
        r2 = results['metrics'].get('r2_score', 0)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor=self.COLORS['bg_dark'], alpha=0.8, edgecolor=self.COLORS['primary_blue']),
               fontsize=13, fontweight='bold', verticalalignment='top')
        plt.tight_layout()
        path = os.path.join(output_dir, 'predictions_vs_actual.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.COLORS['bg_dark'])
        paths.append(path)
        plt.close()
        
        residuals = np.array(results['actual']) - np.array(results['predictions'])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(results['predictions'], residuals, alpha=0.6, 
                  c=self.COLORS['primary_blue'], edgecolors=self.COLORS['primary_orange'], s=80, linewidth=1.5)
        ax.axhline(y=0, color=self.COLORS['success_green'], linestyle='--', linewidth=3)
        ax.set_xlabel('Predicted Values', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.set_title('Residuals Plot', fontsize=16, fontweight='bold', color=self.COLORS['primary_orange'])
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        path = os.path.join(output_dir, 'residuals_plot.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.COLORS['bg_dark'])
        paths.append(path)
        plt.close()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].hist(residuals, bins=30, color=self.COLORS['primary_blue'], 
                       edgecolor=self.COLORS['primary_orange'], alpha=0.7, linewidth=1.5)
        axes[0, 0].set_xlabel('Residuals', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Residuals Distribution', fontsize=13, fontweight='bold', color=self.COLORS['primary_orange'])
        axes[0, 0].grid(alpha=0.2)
        
        axes[0, 1].hist(results['actual'], bins=30, color=self.COLORS['success_green'], 
                       edgecolor=self.COLORS['primary_orange'], alpha=0.7, label='Actual', linewidth=1.5)
        axes[0, 1].hist(results['predictions'], bins=30, color=self.COLORS['primary_blue'], 
                       edgecolor=self.COLORS['primary_orange'], alpha=0.5, label='Predicted', linewidth=1.5)
        axes[0, 1].set_xlabel('Values', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Actual vs Predicted Distribution', fontsize=13, fontweight='bold', color=self.COLORS['primary_orange'])
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.2)
        
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].get_lines()[0].set_markerfacecolor(self.COLORS['primary_blue'])
        axes[1, 0].get_lines()[0].set_markeredgecolor(self.COLORS['primary_orange'])
        axes[1, 0].get_lines()[0].set_markersize(6)
        axes[1, 0].get_lines()[1].set_color(self.COLORS['success_green'])
        axes[1, 0].get_lines()[1].set_linewidth(2)
        axes[1, 0].set_title('Q-Q Plot', fontsize=13, fontweight='bold', color=self.COLORS['primary_orange'])
        axes[1, 0].grid(alpha=0.2)
        
        metrics = results['metrics']
        bars = axes[1, 1].bar(metrics.keys(), metrics.values(), 
                             color=self.COLORS['palette'][:len(metrics)],
                             edgecolor=self.COLORS['primary_orange'], linewidth=2)
        axes[1, 1].set_ylabel('Score', fontsize=11)
        axes[1, 1].set_title('Regression Metrics', fontsize=13, fontweight='bold', color=self.COLORS['primary_orange'])
        for bar, value in zip(bars, metrics.values()):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, height + max(metrics.values())*0.02,
                          f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.2)
        plt.tight_layout()
        path = os.path.join(output_dir, 'regression_analysis.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.COLORS['bg_dark'])
        paths.append(path)
        plt.close()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        error_percent = np.abs(residuals / results['actual']) * 100
        ax.scatter(range(len(error_percent)), error_percent, alpha=0.6,
                  c=self.COLORS['primary_blue'], edgecolors=self.COLORS['primary_orange'], s=60, linewidth=1.5)
        ax.axhline(y=np.mean(error_percent), color=self.COLORS['success_green'], 
                  linestyle='--', linewidth=2, label=f'Mean Error: {np.mean(error_percent):.2f}%')
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Absolute Percentage Error', fontsize=12)
        ax.set_title('Prediction Error Distribution', fontsize=16, fontweight='bold', color=self.COLORS['primary_orange'])
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        path = os.path.join(output_dir, 'error_distribution.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.COLORS['bg_dark'])
        paths.append(path)
        plt.close()
        
        return paths

    def _generate_model_summary(self, results, output_dir: str):
        try:
            summary_path = os.path.join(output_dir, 'model_summary.txt')
            with open(summary_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("         NOVENTIS AutoML - MODEL SUMMARY\n")
                f.write("="*60 + "\n\n")
                f.write(f"Task Type: {self.task_type}\n")
                model_name = results['model_comparison']['rankings'][0]['model']
                f.write("PERFORMANCE METRICS:\n")
                f.write(f"Model: {model_name}\n")
                f.write("-" * 30 + "\n")
                for metric, value in results[model_name.replace(' ', '_')]['metrics'].items():
                    f.write(f"{metric.replace('_', ' ').title()}: {value:.4f}\n")
                if results[model_name.replace(' ', '_')]['feature_importance'] is not None:
                    f.write(f"\nTOP 10 IMPORTANT FEATURES:\n")
                    f.write("-" * 30 + "\n")
                    for idx, row in results[model_name.replace(' ', '_')]['feature_importance'].head(10).iterrows():
                        f.write(f"{row['feature']}: {row['importance']:.4f}\n")
        except Exception as e:
            print(f"Error creating summary: {e}")

    def _rank_models(self, results) -> Dict:
        rankings = []
        primary_metric = 'f1_score' if self.task_type == "classification" and self.metrics is None else \
                        ('r2_score' if self.task_type == "regression" and self.metrics is None else self.metrics)
        self.metrics = primary_metric
        for name, res in results.items():
            if 'error' in res or 'metrics' not in res:
                continue
            score = res['metrics'].get(primary_metric, -1)      
            model_display_name = name.replace('_', ' ')
            rankings.append({'model': model_display_name, 'score': score, 'metrics': res['metrics']})
        rankings.sort(key=lambda x: x['score'], reverse=True)
        return {'rankings': rankings, 'best_model': rankings[0]['model'] if rankings else None, 'primary_metric': primary_metric}

    def _visualize_model_comparison(self, ranked_results, all_results, output_dir: str):
        if not ranked_results['rankings']:
            return
        try:
            self._set_plot_style()
            df_ranks = pd.DataFrame(ranked_results['rankings'])
            fig, ax = plt.subplots(figsize=(14, 8))
            bars = ax.barh(df_ranks['model'], df_ranks['score'], 
                          color=self.COLORS['palette'][:len(df_ranks)],
                          edgecolor=self.COLORS['primary_orange'], linewidth=2)
            ax.set_xlabel(ranked_results['primary_metric'].replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel('Model', fontsize=12)
            ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', color=self.COLORS['primary_orange'])
            for bar, score in zip(bars, df_ranks['score']):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                       f'{score:.4f}', va='center', ha='left', fontweight='bold', fontsize=11)
            ax.grid(axis='x', alpha=0.2)
            plt.tight_layout()
            path = os.path.join(output_dir, 'model_comparison.png')
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.COLORS['bg_dark'])
            plt.close()
            self._create_metrics_heatmap(df_ranks, output_dir)
        except Exception as e:
            print(f"Error creating comparison: {e}")

    def _create_metrics_heatmap(self, df_ranks, output_dir: str):
        try:
            all_metrics = {ranking['model']: ranking['metrics'] for ranking in df_ranks.to_dict('records')}
            if not all_metrics:
                return
            metrics_df = pd.DataFrame(all_metrics).T
            if len(metrics_df.columns) > 1:
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                           cbar_kws={'label': 'Score'}, ax=ax, linewidths=0.5)
                ax.set_title('Model Performance Heatmap', fontsize=16, fontweight='bold', color=self.COLORS['primary_orange'])
                ax.set_xlabel('Metrics', fontsize=12)
                ax.set_ylabel('Models', fontsize=12)
                plt.xticks(rotation=45)
                plt.tight_layout()
                path = os.path.join(output_dir, 'metrics_heatmap.png')
                plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.COLORS['bg_dark'])
                plt.close()
        except Exception as e:
            print(f"Error creating heatmap: {e}")

    def _generate_comparison_report(self, ranked_results, all_results, output_dir: str):
        try:
            report_path = os.path.join(output_dir, 'model_comparison_report.txt')
            with open(report_path, 'w') as f:
                f.write("="*70 + "\n")
                f.write("         NOVENTIS AutoML - MODEL COMPARISON REPORT\n")
                f.write("="*70 + "\n\n")
                f.write(f"Task Type: {self.task_type.title()}\n")
                f.write(f"Primary Metric: {ranked_results['primary_metric']}\n")
                f.write(f"Best Model: {ranked_results['best_model']}\n\n")
                f.write("MODEL RANKINGS:\n")
                f.write("-" * 50 + "\n")
                for i, ranking in enumerate(ranked_results['rankings'], 1):
                    f.write(f"{i}. {ranking['model']}\n")
                    f.write(f"   Primary Score: {ranking['score']:.4f}\n")
                    f.write("   All Metrics:\n")
                    for metric, value in ranking['metrics'].items():
                        f.write(f"     {metric.replace('_', ' ').title()}: {value:.4f}\n")
                    f.write("\n")
        except Exception as e:
            print(f"Error creating report: {e}")

    def generate_html_report(self, report_height: int = 800) -> HTML:
        if not hasattr(self, 'results') or not self.results:
            return HTML("<p>No results available. Run fit() first.</p>")
        
        best_model_name = self.results['model_comparison']['rankings'][0]['model'] if 'model_comparison' in self.results else 'AutoML'
        best_model_res = self.results.get(best_model_name.replace(' ', '_'), self.results.get('AutoML', {}))
        
        overview_html = self._generate_overview_section()
        metrics_html = self._generate_metrics_section(best_model_res, best_model_name)
        visualizations_html = self._generate_visualizations_section()
        comparison_html = self._generate_comparison_section() if self.compare else ""
        predictions_html = self._generate_predictions_section(best_model_res)
        
        tabs_config = [
            {'id': 'overview', 'title': 'Overview', 'content': overview_html},
            {'id': 'metrics', 'title': 'Performance', 'content': metrics_html},
            {'id': 'predictions', 'title': 'Predictions', 'content': predictions_html},
            {'id': 'visualizations', 'title': 'Visualizations', 'content': visualizations_html},
        ]
        
        if self.compare:
            tabs_config.append({'id': 'comparison', 'title': 'Comparison', 'content': comparison_html})
        
        navbar_html = ""
        main_content_html = ""
        
        for i, tab in enumerate(tabs_config):
            active_class = 'active' if i == 0 else ''
            navbar_html += f'<button class="nav-btn {active_class}" onclick="showTab(event, \'{tab["id"]}\', \'{self.report_id}\')">{tab["title"]}</button>'
            main_content_html += f'<section id="{tab["id"]}-{self.report_id}" class="content-section {active_class}"><h2>{tab["title"]}</h2>{tab["content"]}</section>'
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
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
                body {{
                    font-family: var(--font-main);
                    background-color: transparent;
                    color: var(--text-light);
                    margin: 0;
                    padding: 0;
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
                    background-color: var(--bg-dark-2);
                }}
                header h1 {{
                    font-family: var(--font-header);
                    font-size: 2.5rem;
                    margin: 0;
                    color: var(--primary-blue);
                }}
                header p {{
                    margin: 0.25rem 0 0;
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
                }}
                .content-section.active {{
                    display: block;
                }}
                h2, h3, h4 {{
                    font-family: var(--font-header);
                }}
                h2 {{
                    font-size: 2rem;
                    color: var(--primary-orange);
                    border-bottom: 1px solid var(--border-color);
                    padding-bottom: 0.5rem;
                    margin-top: 0;
                }}
                h3 {{
                    color: var(--primary-blue);
                    font-size: 1.5rem;
                    margin-top: 2rem;
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
                }}
                .metric-label {{
                    font-size: 0.9rem;
                    color: var(--text-muted);
                    margin-bottom: 0.5rem;
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
                }}
                .styled-table {{
                    width: 100%;
                    color: var(--text-muted);
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
                .comparison-container {{
                    margin-top: 2rem;
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
                    reportFrame.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
                    reportFrame.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
                    const sectionId = `#${{tabName}}-${{reportId}}`;
                    const sectionToShow = reportFrame.querySelector(sectionId);
                    if (sectionToShow) sectionToShow.classList.add('active');
                    event.currentTarget.classList.add('active');
                }}
            </script>
        </body>
        </html>
        """
        return HTML(html_template)
    
    def _generate_overview_section(self) -> str:
        best_model_name = self.results['model_comparison']['rankings'][0]['model'] if 'model_comparison' in self.results else 'AutoML'
        train_test_ratio = f"{len(self.X_train)/(len(self.X_train)+len(self.X_test))*100:.1f}% / {len(self.X_test)/(len(self.X_train)+len(self.X_test))*100:.1f}%"
        
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
        
        if self.task_type == "classification":
            class_dist = self.y_train.value_counts().to_dict()
            html += "<h3>Target Distribution</h3><div class='table-scroll-wrapper'><table class='styled-table'><thead><tr><th>Class</th><th>Count</th><th>Percentage</th></tr></thead><tbody>"
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
        
        best_model_res = self.results.get(best_model_name.replace(' ', '_'), self.results.get('AutoML', {}))
        if self.use_automl and best_model_res.get('best_config'):
            html += "<h3>Best Model Configuration</h3><div class='table-scroll-wrapper'><table class='styled-table'><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>"
            for key, value in best_model_res['best_config'].items():
                html += f"<tr><td>{key}</td><td>{value}</td></tr>"
            html += "</tbody></table></div>"
        
        return html
    
    def _generate_metrics_section(self, best_model_res, best_model_name) -> str:
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
        
        if best_model_res.get('feature_importance') is not None:
            fi_df = best_model_res['feature_importance'].head(15)
            html += "<h3>Top 15 Feature Importance</h3><div class='table-scroll-wrapper'><table class='styled-table'><thead><tr><th>Rank</th><th>Feature</th><th>Importance</th></tr></thead><tbody>"
            for idx, (_, row) in enumerate(fi_df.iterrows(), 1):
                html += f"<tr><td>{idx}</td><td>{row['feature']}</td><td>{row['importance']:.4f}</td></tr>"
            html += "</tbody></table></div>"
        
        return html
    
    def _generate_predictions_section(self, best_model_res) -> str:
        predictions = best_model_res.get('predictions', [])
        actual = best_model_res.get('actual', [])
        
        html = "<h3>Sample Predictions (First 20)</h3><div class='table-scroll-wrapper'><table class='styled-table'><thead><tr><th>Index</th><th>Actual</th><th>Predicted</th>"
        
        if self.task_type == "classification":
            html += "<th>Correct</th>"
        else:
            html += "<th>Error</th><th>Abs Error</th>"
        html += "</tr></thead><tbody>"
        
        for i in range(min(20, len(predictions))):
            actual_val = actual.iloc[i] if hasattr(actual, 'iloc') else actual[i]
            pred_val = predictions[i]
            
            if self.task_type == "classification":
                correct = "Yes" if actual_val == pred_val else "No"
                html += f"<tr><td>{i+1}</td><td>{actual_val}</td><td>{pred_val}</td><td>{correct}</td></tr>"
            else:
                error = pred_val - actual_val
                abs_error = abs(error)
                html += f"<tr><td>{i+1}</td><td>{actual_val:.4f}</td><td>{pred_val:.4f}</td><td>{error:.4f}</td><td>{abs_error:.4f}</td></tr>"
        
        html += "</tbody></table></div>"
        
        if self.task_type == "classification" and best_model_res.get('prediction_proba') is not None:
            html += "<h3>Prediction Confidence (First 10)</h3><div class='table-scroll-wrapper'><table class='styled-table'><thead><tr><th>Index</th><th>Predicted</th><th>Max Probability</th></tr></thead><tbody>"
            proba = best_model_res['prediction_proba']
            for i in range(min(10, len(proba))):
                max_proba = np.max(proba[i])
                html += f"<tr><td>{i+1}</td><td>{predictions[i]}</td><td>{max_proba:.4f}</td></tr>"
            html += "</tbody></table></div>"
        
        return html
    
    def _generate_visualizations_section(self) -> str:
        html = "<div class='viz-grid'>"
        if 'visualization_paths' in self.results:
            for viz_path in self.results['visualization_paths']:
                viz_name = os.path.basename(viz_path).replace('.png', '').replace('_', ' ').title()
                if os.path.exists(viz_path):
                    html += f"""
                    <div class="viz-item">
                        <h4>{viz_name}</h4>
                        <img src="{viz_path}" alt="{viz_name}">
                    </div>
                    """
        else:
            html += "<p>No visualizations available. Set explain=True.</p>"
        html += "</div>"
        return html
    
    def _generate_comparison_section(self) -> str:
        if 'model_comparison' not in self.results:
            return "<p>No comparison data available.</p>"
        rankings = self.results['model_comparison']['rankings']
        primary_metric = self.results['model_comparison']['primary_metric']
        html = f"<h3>Model Rankings (by {primary_metric.replace('_', ' ').title()})</h3><div class='comparison-container'>"
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
        html += "<h3>All Metrics Comparison</h3><div class='table-scroll-wrapper'><table class='styled-table'><thead><tr><th>Model</th>"
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
        if os.path.exists(os.path.join(self.output_dir, 'model_comparison.png')):
            html += f"""
            <h3>Visual Comparison</h3>
            <div class="viz-item">
                <img src="{os.path.join(self.output_dir, 'model_comparison.png')}" alt="Model Comparison">
            </div>
            """
        return html

    def load_model(self, model_path: str):
        try:
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            print(f"Model loaded from: {model_path}")
            return loaded_model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def predict(self, X_new: Union[pd.DataFrame, np.ndarray], model_path: Optional[str] = None):
        model = self.load_model(model_path) if model_path else self.flaml_model
        if not model:
            raise ValueError("No model available. Train first or specify model_path")
        try:
            predictions = model.predict(X_new)
            print(f"Prediction successful for {len(X_new)} samples")
            if self.task_type == "classification" and hasattr(model, 'predict_proba'):
                return {'predictions': predictions, 'probabilities': model.predict_proba(X_new)}
            return {'predictions': predictions}
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

    def get_model_info(self) -> Dict:
        if not self.flaml_model:
            return {"error": "Model not trained. Run fit() first"}
        return {
            'best_estimator': self.flaml_model.best_estimator,
            'best_config': self.flaml_model.best_config,
            'task_type': self.task_type,
            'training_duration': getattr(self.flaml_model, 'training_duration', 'Unknown'),
            'classes_': getattr(self.flaml_model, 'classes_', None),
            'feature_names': list(self.X_train.columns) if hasattr(self, 'X_train') else None
        }

    def export_results_to_csv(self, output_dir: str = "noventis_output"):
        if not hasattr(self, 'results') or not self.results:
            print("No results to export. Run fit() first")
            return
        best_model_name = self.results['model_comparison']['rankings'][0]['model']
        best_model_key = best_model_name.replace(' ', '_')
        try:
            os.makedirs(output_dir, exist_ok=True)
            predictions_df = pd.DataFrame({
                'actual': self.results[best_model_key]['actual'],
                'predicted': self.results[best_model_key]['predictions']
            })
            if self.results[best_model_key]['prediction_proba'] is not None:
                proba_cols = [f'prob_class_{i}' for i in range(self.results[best_model_key]['prediction_proba'].shape[1])]
                proba_df = pd.DataFrame(self.results[best_model_key]['prediction_proba'], columns=proba_cols)
                predictions_df = pd.concat([predictions_df, proba_df], axis=1)
            pred_path = os.path.join(output_dir, 'predictions.csv')
            predictions_df.to_csv(pred_path, index=False)
            metrics_df = pd.DataFrame([self.results[best_model_key]['metrics']])
            metrics_path = os.path.join(output_dir, 'metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            if self.results[best_model_key]['feature_importance'] is not None:
                fi_path = os.path.join(output_dir, 'feature_importance.csv')
                self.results[best_model_key]['feature_importance'].to_csv(fi_path, index=False)
            print(f"Results exported to: {output_dir}")
        except Exception as e:
            print(f"Error exporting to CSV: {e}")

    def _repr_html_(self):
        if hasattr(self, 'results') and self.results:
            return self.generate_html_report()._repr_html_()
        return f"<p>NoventisAutoML instance - Run fit() to see dashboard</p>"

    def __repr__(self):
        status = "Trained" if self.flaml_model or self.manual_model else "Not Trained"
        best_model = "Unknown"
        if hasattr(self, 'results') and 'model_comparison' in self.results:
            best_model = self.results['model_comparison']['rankings'][0]['model']
        elif self.flaml_model:
            best_model = getattr(self.flaml_model, 'best_estimator', 'AutoML')
        return f"NoventisAutoML(task='{self.task_type}', target='{self.target_column}', status='{status}', best='{best_model}', shape={getattr(self, 'df', pd.DataFrame()).shape})"