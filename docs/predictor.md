# Predictor - Automated Machine Learning

## Overview

**Predictor** is a Noventis module that provides automated and manual machine learning with two main approaches: **NoventisAutoML** (powered by FLAML) for quick model selection, and **NoventisManualPredictor** (powered by Optuna) for fine-grained control over specific algorithms.

---

## Key Features

### AutoML Mode
- FLAML-powered automatic model selection
- Time-based training budget
- Automatic hyperparameter optimization
- Built-in model comparison
- Interactive HTML reports

### Manual Mode
- Choose specific algorithms
- Optuna-based hyperparameter tuning
- Cross-validation strategies (standard/repeated)
- SHAP explanations
- Detailed performance analysis

---

## Installation

```bash
pip install noventis
```

---

## Quick Start

### AutoML Mode

```python
from noventis.predictor import NoventisAutoML
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Initialize AutoML
automl = NoventisAutoML(
    data=df,
    target='target_column',
    time_budget=60,
    explain=True
)

# Train
results = automl.fit()

# Generate report
report = automl.generate_html_report()
```

### Manual Mode

```python
from noventis.predictor import NoventisManualPredictor

# Initialize with specific models
predictor = NoventisManualPredictor(
    model_name=['random_forest', 'xgboost'],
    task='classification',
    tune_hyperparameters=True,
    n_trials=50
)

# Train
results = predictor.fit(
    df=df,
    target_column='target',
    test_size=0.2
)

# Display report
predictor.display_report()
```

---

## Available Models

### Classification
- `logistic_regression`
- `decision_tree`
- `random_forest`
- `gradient_boosting`
- `xgboost`
- `lightgbm`
- `catboost`

### Regression
- `linear_regression`
- `decision_tree`
- `random_forest`
- `gradient_boosting`
- `xgboost`
- `lightgbm`
- `catboost`

---

## NoventisAutoML

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `str` or `DataFrame` | Required | Data source |
| `target` | `str` | Required | Target column name |
| `task` | `str` | `None` | `'classification'`, `'regression'`, or `'auto'` |
| `models` | `list` | `None` | Specific models to train alongside AutoML |
| `explain` | `bool` | `True` | Generate visualizations |
| `compare` | `bool` | `True` | Compare multiple models |
| `time_budget` | `int` | `60` | Training time in seconds |
| `output_dir` | `str` | `'Noventis_Results'` | Output directory |
| `test_size` | `float` | `0.2` | Test set proportion |
| `random_state` | `int` | `42` | Random seed |

### Methods

#### `fit(time_budget, metric)`
Train the model(s).

```python
results = automl.fit(time_budget=120)
```

Returns dictionary with model results, metrics, and predictions.

---

#### `predict(X_new, model_path)`
Make predictions on new data.

```python
predictions = automl.predict(X_new)
```

---

#### `compare_models(models_to_compare, output_dir)`
Compare multiple models.

```python
comparison = automl.compare_models(
    models_to_compare=['random_forest', 'xgboost']
)
```

---

#### `generate_html_report(report_height)`
Generate interactive HTML report.

```python
report = automl.generate_html_report(report_height=800)
```

---

#### `export_results_to_csv(output_dir)`
Export predictions and metrics to CSV.

```python
automl.export_results_to_csv('results_csv')
```

---

## NoventisManualPredictor

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` or `list` | Required | Model(s) to train |
| `task` | `str` | Required | `'classification'` or `'regression'` |
| `random_state` | `int` | `42` | Random seed |
| `tune_hyperparameters` | `bool` | `False` | Enable Optuna tuning |
| `n_trials` | `int` | `50` | Optuna trials |
| `cv_folds` | `int` | `3` | Cross-validation folds |
| `cv_strategy` | `str` | `'repeated'` | `'repeated'` or `'standard'` |
| `output_dir` | `str` | `None` | Output directory |

### Methods

#### `fit(df, target_column, test_size, compare, explain)`
Main training pipeline.

```python
results = predictor.fit(
    df=df,
    target_column='target',
    test_size=0.2,
    compare=True,
    explain=True
)
```

Returns dictionary with best model and all results.

---

#### `predict(X_new, model_path)`
Make predictions.

```python
predictions = predictor.predict(X_new)
```

---

#### `save_model(filepath)`
Save trained model.

```python
predictor.save_model('model.pkl')
```

---

#### `load_model(filepath)`
Load saved model (static method).

```python
model = NoventisManualPredictor.load_model('model.pkl')
```

---

#### `explain_model(plot_type, feature)`
Generate SHAP explanations.

```python
# Summary plot
predictor.explain_model(plot_type='summary')

# Beeswarm plot
predictor.explain_model(plot_type='beeswarm')

# Dependence plot
predictor.explain_model(
    plot_type='dependence',
    feature='feature_name'
)
```

---

#### `display_report()`
Display HTML report in notebook.

```python
predictor.display_report()
```

---

#### `get_results_dataframe()`
Get model comparison as DataFrame.

```python
df_results = predictor.get_results_dataframe()
```

---

## Hyperparameter Tuning

### AutoML (FLAML)
FLAML automatically optimizes hyperparameters within the time budget.

```python
automl = NoventisAutoML(
    data=df,
    target='target',
    time_budget=300  # FLAML tunes for 5 minutes
)
```

---

### Manual (Optuna)
Optuna uses Bayesian optimization with cross-validation.

```python
predictor = NoventisManualPredictor(
    model_name='xgboost',
    task='classification',
    tune_hyperparameters=True,
    n_trials=100,  # More trials = better optimization
    cv_folds=5
)
```

**Search Spaces:**

**Random Forest:**
- n_estimators: 100-1000
- max_depth: 5-50
- min_samples_split: 2-20
- min_samples_leaf: 1-10

**XGBoost:**
- n_estimators: 100-2000
- learning_rate: 0.01-0.3
- max_depth: 3-12
- subsample: 0.6-1.0
- colsample_bytree: 0.6-1.0
- gamma: 0-5

**LightGBM:**
- n_estimators: 100-2000
- learning_rate: 0.01-0.3
- num_leaves: 20-150
- max_depth: 3-15
- reg_alpha: 0.0-1.0
- reg_lambda: 0.0-1.0

---

## Evaluation Metrics

### Classification
- **accuracy**: Overall correctness
- **precision**: True positives / (True positives + False positives)
- **recall**: True positives / (True positives + False negatives)
- **f1_score**: Harmonic mean of precision and recall

### Regression
- **mae**: Mean Absolute Error
- **mse**: Mean Squared Error
- **rmse**: Root Mean Squared Error
- **r2_score**: R-squared (coefficient of determination)

---

## Use Cases

### Use Case 1: Quick Model Selection

```python
# Use AutoML for fast prototyping
automl = NoventisAutoML(
    data='data.csv',
    target='price',
    task='regression',
    time_budget=120
)

results = automl.fit()
print(f"Best model: {results['AutoML']['best_estimator']}")
```

---

### Use Case 2: Production Model

```python
# Fine-tune specific models
predictor = NoventisManualPredictor(
    model_name=['xgboost', 'lightgbm'],
    task='classification',
    tune_hyperparameters=True,
    n_trials=100,
    output_dir='production'
)

results = predictor.fit(df, target_column='churn')
predictor.save_model('production/churn_model.pkl')
```

---

### Use Case 3: Model Comparison

```python
# Compare AutoML with specific models
automl = NoventisAutoML(
    data=df,
    target='target',
    models=['random_forest', 'xgboost'],  # Train these too
    compare=True,
    time_budget=180
)

results = automl.fit()

# Check rankings
for rank in results['model_comparison']['rankings']:
    print(f"{rank['model']}: {rank['score']:.4f}")
```

---

### Use Case 4: Model Explanation

```python
# Train and explain
predictor = NoventisManualPredictor(
    model_name='random_forest',
    task='classification'
)

predictor.fit(df, target_column='outcome')

# SHAP analysis
predictor.explain_model(plot_type='summary')
```

---

## Visualizations

### AutoML Visualizations

**Feature Importance:**
- Top 20 features
- Bar chart sorted by importance

**Training History:**
- Optimization progress over time
- Best validation loss

**Classification:**
- Confusion matrix
- ROC curve (binary)
- Precision-Recall curve (binary)
- Class distribution

**Regression:**
- Predictions vs Actual
- Residuals plot
- Residuals distribution
- Error distribution

---

### Manual Predictor Visualizations

**Classification (4-panel):**
1. Confusion Matrix
2. ROC Curve
3. Precision-Recall Curve
4. Model Comparison

**Regression (4-panel):**
1. Predicted vs Actual
2. Residuals Plot
3. Residuals Distribution
4. Model Comparison

**SHAP:**
- Summary plot
- Beeswarm plot
- Dependence plots

---

## Best Practices

### 1. Start with AutoML

```python
# Quick baseline
automl = NoventisAutoML(
    data=df,
    target='target',
    time_budget=120
)
baseline = automl.fit()
```

---

### 2. Use Manual for Production

```python
# Fine-tune for deployment
predictor = NoventisManualPredictor(
    model_name=['xgboost', 'lightgbm'],
    task='classification',
    tune_hyperparameters=True,
    n_trials=100
)
```

---

### 3. Set Appropriate Time Budgets

```python
# Development: 60-120 seconds
# Production: 300-600 seconds
```

---

### 4. Always Use random_state

```python
# For reproducibility
automl = NoventisAutoML(
    data=df,
    target='target',
    random_state=42
)
```

---

### 5. Save Important Models

```python
predictor.fit(df, target_column='target')
predictor.save_model('important_model.pkl')
```

---

## Data Cleaning Integration

```python
from noventis.data_cleaner import NoventisDataCleaner

# Clean data first
cleaner = NoventisDataCleaner()
df_clean = cleaner.fit_transform(df.drop('target', axis=1))
df_clean['target'] = df['target']

# Train on clean data
automl = NoventisAutoML(data=df_clean, target='target')
```

---

## Troubleshooting

### Issue 1: Time Budget Too Short

```python
# Increase time budget
automl = NoventisAutoML(
    data=df,
    target='target',
    time_budget=300  # Increase from 60
)
```

---

### Issue 2: Tuning Takes Too Long

```python
# Reduce trials and folds
predictor = NoventisManualPredictor(
    model_name='xgboost',
    task='classification',
    tune_hyperparameters=True,
    n_trials=20,  # Reduce from 50
    cv_folds=3    # Reduce from 5
)
```

---

### Issue 3: Out of Memory

```python
# Use lighter models
predictor = NoventisManualPredictor(
    model_name=['logistic_regression', 'decision_tree'],
    task='classification'
)

# Or sample data
df_sample = df.sample(frac=0.5, random_state=42)
```

---

### Issue 4: Wrong Task Detection

```python
# Specify explicitly
automl = NoventisAutoML(
    data=df,
    target='age',
    task='classification'  # Force classification
)
```

---

## Examples

### Example 1: Titanic

```python
import pandas as pd
from noventis.predictor import NoventisAutoML

df = pd.read_csv('titanic.csv')

automl = NoventisAutoML(
    data=df,
    target='Survived',
    task='classification',
    time_budget=120
)

results = automl.fit()
print(f"Accuracy: {results['AutoML']['metrics']['accuracy']:.4f}")
```

---

### Example 2: House Prices

```python
from noventis.predictor import NoventisManualPredictor

df = pd.read_csv('house_prices.csv')

predictor = NoventisManualPredictor(
    model_name=['random_forest', 'xgboost', 'lightgbm'],
    task='regression',
    tune_hyperparameters=True,
    n_trials=50
)

results = predictor.fit(df, target_column='SalePrice')
predictor.display_report()
```

---

### Example 3: Customer Churn

```python
from noventis.predictor import NoventisAutoML

df = pd.read_csv('churn.csv')

automl = NoventisAutoML(
    data=df,
    target='Churn',
    time_budget=180,
    explain=True
)

results = automl.fit()

# Predict new customers
new_data = pd.read_csv('new_customers.csv')
predictions = automl.predict(new_data)
```

---


**Made with ❤️ by the Noventis Team**