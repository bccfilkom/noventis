# NoventisAutoML Documentation

**A comprehensive automated machine learning library for classification and regression tasks**

---

## Overview

NoventisAutoML is a Python library that provides automated machine learning capabilities with extensive visualization, model comparison, and reporting features. It combines FLAML's AutoML engine with manual model training options to find the best performing model for your dataset.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [API Reference](#api-reference)
4. [Examples](#examples)
5. [Visualization Guide](#visualization-guide)
6. [Best Practices](#best-practices)

---

## Installation

### Prerequisites
```bash
pip install pandas numpy scikit-learn flaml matplotlib seaborn scipy
```

### Optional Dependencies
```python
from IPython.display import HTML  # For Jupyter notebook reports
```

---

## Quick Start

### Basic Usage
```python
from noventis_automl import NoventisAutoML

# Initialize with data
automl = NoventisAutoML(
    data='your_data.csv',
    target='target_column',
    task='classification'
)

# Train models
results = automl.fit(time_budget=120)

# Display interactive report (in Jupyter)
automl.generate_html_report()
```

---

## API Reference

### Class: NoventisAutoML
```python
NoventisAutoML(
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
    random_state: int = 42
)
```

An automated machine learning class that handles data preprocessing, model training, evaluation, and visualization.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | str or DataFrame | Required | Path to CSV file or pandas DataFrame |
| `target` | str | Required | Name of the target column |
| `task` | str or None | None | Task type: 'classification' or 'regression'. Auto-detected if None |
| `models` | List[str] or None | None | Specific models to train. If None, uses AutoML |
| `explain` | bool | True | Generate visualizations and explanations |
| `compare` | bool | True | Compare multiple models |
| `metrics` | str or None | None | Primary metric for evaluation |
| `time_budget` | int | 60 | Time budget in seconds for AutoML |
| `output_dir` | str | 'Noventis_Results' | Directory for saving outputs |
| `test_size` | float | 0.2 | Proportion of dataset for testing (0.0-1.0) |
| `random_state` | int | 42 | Random seed for reproducibility |

#### Available Models

**Classification:**
- `'logistic_regression'`
- `'random_forest'`
- `'xgboost'`
- `'decision_tree'`
- `'lightgbm'`
- `'catboost'`
- `'gradient_boosting'`

**Regression:**
- `'linear_regression'`
- `'random_forest'`
- `'xgboost'`
- `'gradient_boosting'`
- `'lightgbm'`
- `'catboost'`

#### Available Metrics

**Classification:**
- `'accuracy'` - Overall accuracy
- `'precision'` - Precision score (weighted)
- `'recall'` - Recall score (weighted)
- `'f1_score'` - F1 score (macro)

**Regression:**
- `'mae'` - Mean Absolute Error
- `'mse'` - Mean Squared Error
- `'rmse'` - Root Mean Squared Error
- `'r2_score'` - R² Score

---

### Methods

#### fit()
```python
fit(time_budget: int = 60, metric: Optional[str] = None) -> Dict
```

Train the model(s) and generate results.

**Parameters:**
- `time_budget` (int): Maximum time in seconds for training
- `metric` (str, optional): Evaluation metric to optimize

**Returns:**
- Dictionary containing results with keys:
  - `'AutoML'` or model names: Model results
  - `'model_comparison'`: Comparison rankings
  - `'visualization_paths'`: List of generated plots

**Example:**
```python
results = automl.fit(time_budget=180, metric='f1_score')
print(f"Best model: {results['model_comparison']['best_model']}")
```

---

#### compare_models()
```python
compare_models(
    models_to_compare: Optional[List[str]] = None,
    output_dir: str = "Noventis_results"
) -> Dict
```

Compare multiple models and rank them by performance.

**Parameters:**
- `models_to_compare` (list, optional): List of model names to compare
- `output_dir` (str): Directory to save comparison results

**Returns:**
- Dictionary with keys:
  - `'rankings'`: List of models sorted by performance
  - `'best_model'`: Name of the best performing model
  - `'primary_metric'`: Metric used for ranking

**Example:**
```python
comparison = automl.compare_models(
    models_to_compare=['random_forest', 'xgboost', 'lightgbm']
)
```

---

#### predict()
```python
predict(
    X_new: Union[pd.DataFrame, np.ndarray],
    model_path: Optional[str] = None
) -> Dict
```

Make predictions on new data.

**Parameters:**
- `X_new` (DataFrame or array): Features for prediction
- `model_path` (str, optional): Path to saved model file

**Returns:**
- Dictionary with keys:
  - `'predictions'`: Array of predictions
  - `'probabilities'`: Class probabilities (classification only)

**Example:**
```python
new_data = pd.read_csv('new_data.csv')
predictions = automl.predict(new_data)
print(predictions['predictions'])
```

---

#### generate_html_report()
```python
generate_html_report(report_height: int = 800) -> HTML
```

Generate an interactive HTML report for Jupyter notebooks.

**Parameters:**
- `report_height` (int): Height of the report frame in pixels

**Returns:**
- IPython.display.HTML object

**Example:**
```python
# In Jupyter notebook
automl.generate_html_report(report_height=1000)
```

---

#### load_model()
```python
load_model(model_path: str)
```

Load a previously saved model.

**Parameters:**
- `model_path` (str): Path to the pickled model file

**Returns:**
- Loaded model object or None if error occurs

**Example:**
```python
model = automl.load_model('Noventis_Results/best_model.pkl')
```

---

#### export_results_to_csv()
```python
export_results_to_csv(output_dir: str = "noventis_output")
```

Export predictions, metrics, and feature importance to CSV files.

**Parameters:**
- `output_dir` (str): Directory to save CSV files

**Generates:**
- `predictions.csv`: Actual vs predicted values
- `metrics.csv`: Performance metrics
- `feature_importance.csv`: Feature importance scores (if available)

**Example:**
```python
automl.export_results_to_csv('my_results')
```

---

#### get_model_info()
```python
get_model_info() -> Dict
```

Get detailed information about the trained model.

**Returns:**
- Dictionary containing:
  - `'best_estimator'`: Name of best model
  - `'best_config'`: Hyperparameters
  - `'task_type'`: Classification or regression
  - `'training_duration'`: Training time
  - `'classes_'`: Class labels (classification)
  - `'feature_names'`: Feature names

**Example:**
```python
info = automl.get_model_info()
print(f"Model: {info['best_estimator']}")
print(f"Config: {info['best_config']}")
```

---

## Examples

### Example 1: Classification with AutoML
```python
from noventis_automl import NoventisAutoML

# Load data and initialize
automl = NoventisAutoML(
    data='customer_churn.csv',
    target='Churn',
    task='classification',
    time_budget=180,
    metrics='f1_score'
)

# Train and evaluate
results = automl.fit()

# Access best model metrics
best_metrics = results['AutoML']['metrics']
print(f"F1 Score: {best_metrics['f1_score']:.4f}")
print(f"Accuracy: {best_metrics['accuracy']:.4f}")

# Make predictions
new_customers = pd.read_csv('new_customers.csv')
predictions = automl.predict(new_customers)
```

---

### Example 2: Regression with Model Comparison
```python
automl = NoventisAutoML(
    data='house_prices.csv',
    target='Price',
    task='regression',
    compare=True,
    models=['random_forest', 'xgboost', 'lightgbm'],
    metrics='r2_score'
)

results = automl.fit(time_budget=240)

# View rankings
rankings = results['model_comparison']['rankings']
for i, rank in enumerate(rankings, 1):
    print(f"{i}. {rank['model']}: R² = {rank['score']:.4f}")
```

---

### Example 3: Manual Model Selection
```python
# Train only specific models without AutoML
automl = NoventisAutoML(
    data='sales_data.csv',
    target='Revenue',
    task='regression',
    models=['linear_regression', 'random_forest'],
    compare=False,
    explain=True
)

results = automl.fit()

# Export results
automl.export_results_to_csv('sales_predictions')
```

---

### Example 4: Custom Train/Test Split
```python
automl = NoventisAutoML(
    data=my_dataframe,
    target='outcome',
    test_size=0.3,  # 70% train, 30% test
    random_state=123
)

results = automl.fit()
```

---

## Visualization Guide

NoventisAutoML automatically generates comprehensive visualizations when `explain=True`.

### Classification Visualizations

1. **Confusion Matrix** (`confusion_matrix.png`)
   - Normalized heatmap showing prediction accuracy per class

2. **Classification Metrics** (`classification_metrics.png`)
   - Bar chart of accuracy, precision, recall, and F1 score

3. **ROC & PR Curves** (`roc_pr_curves.png`)
   - ROC curve with AUC score
   - Precision-Recall curve (binary classification only)

4. **Class Distribution** (`class_distribution.png`)
   - Pie charts comparing actual vs predicted distributions

### Regression Visualizations

1. **Predictions vs Actual** (`predictions_vs_actual.png`)
   - Scatter plot with perfect prediction line and R² score

2. **Residuals Plot** (`residuals_plot.png`)
   - Distribution of prediction errors

3. **Regression Analysis** (`regression_analysis.png`)
   - Residuals histogram
   - Actual vs predicted distributions
   - Q-Q plot for normality
   - Metrics bar chart

4. **Error Distribution** (`error_distribution.png`)
   - Absolute percentage error per sample

### AutoML Specific

1. **Feature Importance** (`feature_importance.png`)
   - Top 20 most important features

2. **Training History** (`training_history.png`)
   - Validation loss over training time

3. **Model Comparison** (`model_comparison.png`)
   - Performance comparison across all models

---

## Best Practices

### 1. Data Preparation
```python
# Clean data before passing to NoventisAutoML
df = pd.read_csv('data.csv')
df = df.dropna()  # Handle missing values
df = df.drop_duplicates()  # Remove duplicates

automl = NoventisAutoML(data=df, target='target')
```

### 2. Task Type Selection

Let the library auto-detect task type when uncertain:
```python
# Auto-detection
automl = NoventisAutoML(data='data.csv', target='target')

# Explicit specification (recommended for clarity)
automl = NoventisAutoML(data='data.csv', target='target', task='classification')
```

### 3. Time Budget Allocation

- **Quick exploration:** 60-120 seconds
- **Production models:** 300-600 seconds
- **Best performance:** 900+ seconds

### 4. Model Selection Strategy
```python
# Start with AutoML for baseline
automl = NoventisAutoML(data=df, target='y', compare=True)

# Then compare specific models
comparison = automl.compare_models(
    models_to_compare=['xgboost', 'lightgbm', 'catboost']
)
```

### 5. Metric Selection

Choose metrics based on your problem:
```python
# Imbalanced classification
automl = NoventisAutoML(data=df, target='y', metrics='f1_score')

# Regression with outliers
automl = NoventisAutoML(data=df, target='y', metrics='mae')
```

### 6. Memory Management

For large datasets:
```python
# Disable visualizations to save memory
automl = NoventisAutoML(
    data='large_data.csv',
    target='target',
    explain=False,
    compare=False
)
```

---

## Output Files

NoventisAutoML generates the following files in `output_dir`:

### Models
- `best_automl_model.pkl` - Trained AutoML model
- `best_model.pkl` - Best performing model overall

### Logs
- `flaml.log` - AutoML training log
- `model_summary.txt` - Text summary of results
- `model_comparison_report.txt` - Detailed comparison report

### Visualizations
- Various PNG files (see Visualization Guide)

---

## Attributes

After training, access these attributes:
```python
automl.df                    # Original DataFrame
automl.X_train, automl.X_test  # Train/test features
automl.y_train, automl.y_test  # Train/test targets
automl.flaml_model           # Trained FLAML model
automl.manual_model          # Trained manual model
automl.results               # Complete results dictionary
automl.task_type             # Detected or specified task
```

---

## Error Handling

NoventisAutoML includes built-in error handling:
```python
try:
    automl = NoventisAutoML(data='data.csv', target='missing_column')
except ValueError as e:
    print(f"Error: {e}")  # "Target column 'missing_column' not found"
```

---

## Performance Tips

1. **Use appropriate time budget** - More time generally yields better models
2. **Select relevant features** - Drop irrelevant columns before training
3. **Balance dataset** - For classification, consider class imbalance
4. **Normalize features** - FLAML handles this automatically
5. **Use GPU** - Install GPU versions of XGBoost, LightGBM for faster training

---

## License and Credits

This library uses:
- FLAML for AutoML capabilities
- Scikit-learn for machine learning algorithms
- Matplotlib and Seaborn for visualizations

---

**Version:** 1.0  
**Last Updated:** 2025