import pandas as pd
import numpy as np
import os
import pickle
from typing import Union, Optional, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

from supervised import AutoML

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .manual import (
    load_model_classification, 
    load_model_regression,
    eval_classification,
    eval_regression
)

class NoventisAutoML:
    def __init__(self):
        self.automl = None
        self.model_results = {}
        self.best_model = None
        self.task_type = None
        self.target_column = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def _detect_task_type(self, y: pd.Series) -> str:
        """
        Automatically detect whether the task is classification or regression.
        """

        if pd.api.types.is_numeric_dtype(y):
            unique_ratio = len(y.unique()) / len(y)
            if unique_ratio > 0.05:
                return "regression"
            else:
                return "classification"
        else:
            return "classification"
    
    def _setup_data(self, df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in the dataset")
        
        self.target_column = target
        X = df.drop(columns=[target])
        y = df[target]
    
        if self.task_type is None:
            self.task_type = self._detect_task_type(y)
            print(f"Task type detected: {self.task_type}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, 
            random_state=random_state,
            stratify=y if self.task_type == "classification" else None
        )
        
        print(f"Data split: Train={len(self.X_train)}, Test={len(self.X_test)}")

def automl(
    data: Union[str, pd.DataFrame],
    target: str,
    task: Optional[str] = None,
    model: Optional[str] = None,
    tuning: bool = True,
    use_cleaner: bool = False,
    explain: bool = True,
    compare: bool = False,
    metric: Optional[str] = None,
    time_budget: int = 60,
    output: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for AutoML using MLJAR.
    
    Parameters:
    - data: Path to the CSV file or a pandas DataFrame
    - target: Name of the target column
    - task: "classification" or "regression" (auto-detected if None)
    - model: Specific model to use (for manual selection)
    - tuning: Whether to perform hyperparameter tuning
    - use_cleaner: Whether to use a data cleaner
    - explain: Generate visualizations and a report
    - compare: Compare several models
    - metric: Main evaluation metric
    - time_budget: Maximum training time (in seconds)
    - output: Path to save the model
    - test_size: Proportion of the test data
    - random_state: Random state for reproducibility
    
    Returns:
    - A dict containing model results, metrics, and other information
    """
    
    # Load data
    if isinstance(data, str):
        if not os.path.exists(data):
            raise FileNotFoundError(f"File {data} not found")
        df = pd.read_csv(data)
        print(f"Dataset loaded: {df.shape}")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        print(f"DataFrame loaded: {df.shape}")
    else:
        raise ValueError("Data must be a path string or a pandas DataFrame")
    
    # Initialize AutoML instance
    automl_instance = NoventisAutoML()
    
    # Set task type
    if task:
        automl_instance.task_type = task.lower()
    
    # Setup data
    automl_instance._setup_data(df, target, test_size, random_state)
    
    # Data cleaner (if needed)
    if use_cleaner:
        print("Running data cleaner...")
        # TODO: Import and use noventis.cleaner

        pass
    
    # If the user wants a specific model (manual mode)
    if model and not compare:
        print(f"Manual mode: using model {model}")
        return _run_manual_model(automl_instance, model, tuning, output, metric)
    
    # AutoML mode with MLJAR
    print("Running AutoML with MLJAR...")
    
    # Setup MLJAR AutoML
    mode = "Compete" if tuning else "Perform"
    if not tuning and time_budget < 120:
        mode = "Explain"  
    
    # Set metric for MLJAR
    mljar_metric = _convert_metric_to_mljar(metric, automl_instance.task_type)
    
    # Initialize MLJAR AutoML
    automl_instance.automl = AutoML(
        mode=mode,
        ml_task=automl_instance.task_type,
        eval_metric=mljar_metric,
        total_time_limit=time_budget,
        explain_level=2 if explain else 0,
        random_state=random_state,
        **kwargs
    )
    
    print(f"Training AutoML model (mode: {mode}, time budget: {time_budget}s)...")
    
    # Train model
    automl_instance.automl.fit(automl_instance.X_train, automl_instance.y_train)
    
    # Predictions
    y_pred = automl_instance.automl.predict(automl_instance.X_test)
    
    # Evaluate
    if automl_instance.task_type == "classification":
        metrics = eval_classification(automl_instance.y_test, y_pred)
    else:
        metrics = eval_regression(automl_instance.y_test, y_pred)
    
    # Prepare results
    results = {
        'model': automl_instance.automl,
        'predictions': y_pred,
        'actual': automl_instance.y_test,
        'metrics': metrics,
        'task_type': automl_instance.task_type,
        'feature_importance': _get_feature_importance(automl_instance.automl),
        'leaderboard': _get_leaderboard(automl_instance.automl)
    }
    
    # Save model
    if output:
        _save_model(automl_instance.automl, output)
        results['model_path'] = output
        print(f"Model saved to: {output}")
    
    # Generate report if explain=True
    if explain:
        report_path = _generate_report(automl_instance.automl, output)
        if report_path:
            results['report_path'] = report_path
            print(f"HTML report saved to: {report_path}")
    
    print("AutoML finished!")
    return results

def _run_manual_model(automl_instance: NoventisAutoML, model: str, tuning: bool, 
                     output: Optional[str], metric: Optional[str]) -> Dict[str, Any]:

    if automl_instance.task_type == "classification":
        model_obj = load_model_classification(model)
    else:
        model_obj = load_model_regression(model)
    
    # Training
    model_obj.fit(automl_instance.X_train, automl_instance.y_train)
    predictions = model_obj.predict(automl_instance.X_test)
    
    # Evaluate
    if automl_instance.task_type == "classification":
        metrics = eval_classification(automl_instance.y_test, predictions)
    else:
        metrics = eval_regression(automl_instance.y_test, predictions)
    
    results = {
        'model': model_obj,
        'predictions': predictions,
        'actual': automl_instance.y_test,
        'metrics': metrics,
        'task_type': automl_instance.task_type,
        'model_name': model
    }
    
    if output:
        _save_model(model_obj, output)
        results['model_path'] = output
    
    return results

def select_models(models: List[str], data: Union[str, pd.DataFrame], target: str, 
                 task: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Compare several models selected by the user.
    
    Parameters:
    - models: List of model names to compare
    - data: Dataset
    - target: Target column
    - task: Task type
    
    Returns:
    - A dict containing the model comparison results
    """
    results = {}
    
    for model in models:
        print(f"\n=== Training {model} ===")
        try:
            result = automl(
                data=data, 
                target=target, 
                task=task, 
                model=model, 
                compare=False,
                explain=False,
                **kwargs
            )
            results[model] = result
        except Exception as e:
            print(f"Error training {model}: {str(e)}")
            results[model] = {'error': str(e)}
    
    # Rank based on metric
    return _rank_models(results, task or "classification")

def compare_models(data: Union[str, pd.DataFrame], target: str, 
                  task: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Compare several model families automatically.
    """
    if task == "classification":
        default_models = [
            'logistic_regression', 'decision_tree', 'random_forest', 
            'xgboost', 'lightgbm', 'catboost'
        ]
    else:
        default_models = [
            'linear_regression', 'decision_tree', 'random_forest', 
            'xgboost', 'lightgbm', 'catboost'
        ]
    
    return select_models(default_models, data, target, task, **kwargs)

def explain_model(model_result: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """
    Generate an explanation for a trained model.
    """
    if 'model' not in model_result:
        raise ValueError("Invalid model result")
    
    model = model_result['model']
    
    # If the model is an MLJAR AutoML object
    if hasattr(model, 'report'):
        report_path = output_path or "model_report.html"
        model.report().save(report_path)
        return report_path
    else:
        print("Model explanation is not available for manual models")
        return ""

def _convert_metric_to_mljar(metric: Optional[str], task_type: str) -> str:
    """
    Convert metric to MLJAR format.
    """
    if metric is None:
        return "logloss" if task_type == "classification" else "rmse"
    
    metric_mapping = {
        "classification": {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "f1_macro": "f1",
            "f1_micro": "f1"
        },
        "regression": {
            "rmse": "rmse",
            "mse": "mse",
            "mae": "mae",
            "r2": "r2"
        }
    }
    
    return metric_mapping.get(task_type, {}).get(metric.lower(), 
                                                "logloss" if task_type == "classification" else "rmse")

def _get_feature_importance(automl_model) -> Optional[pd.DataFrame]:
    """
    Extract feature importance from the MLJAR model.
    """
    try:
        if hasattr(automl_model, 'get_leaderboard'):
            # Get best model
            leaderboard = automl_model.get_leaderboard()
            if len(leaderboard) > 0:
                best_model_name = leaderboard.iloc[0]['name']
                return automl_model.get_feature_importance()
    except:
        pass
    return None

def _get_leaderboard(automl_model) -> Optional[pd.DataFrame]:
    """
    Get the leaderboard from the MLJAR model.
    """
    try:
        if hasattr(automl_model, 'get_leaderboard'):
            return automl_model.get_leaderboard()
    except:
        pass
    return None

def _save_model(model, output_path: str):
    """
    Save the model to a file.
    """
    try:
        if hasattr(model, 'save'):
            # MLJAR model
            model.save(output_path.replace('.pkl', ''))
        else:
            # Manual model
            with open(output_path, 'wb') as f:
                pickle.dump(model, f)
    except Exception as e:
        print(f"Warning: Failed to save model - {str(e)}")

def _generate_report(automl_model, output_path: Optional[str] = None) -> Optional[str]:
    """
    Generate HTML report from MLJAR.
    """
    try:
        if hasattr(automl_model, 'report'):
            report_path = output_path.replace('.pkl', '_report.html') if output_path else 'automl_report.html'
            automl_model.report().save(report_path)
            return report_path
    except Exception as e:
        print(f"Warning: Failed to generate report - {str(e)}")
    return None

def _rank_models(results: Dict[str, Any], task_type: str) -> Dict[str, Any]:

    rankings = []
    
    for model_name, result in results.items():
        if 'error' in result:
            continue
            
        metrics = result.get('metrics', {})
        
        if task_type == "classification":
            score = metrics.get('f1_score', metrics.get('accuracy', 0))
        else:
            score = metrics.get('r2_score', 0)
        
        rankings.append({
            'model': model_name,
            'score': score,
            'metrics': metrics
        })
    
    rankings.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        'rankings': rankings,
        'best_model': rankings[0]['model'] if rankings else None,
        'all_results': results
    }

AutoML = automl