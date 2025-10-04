# Data Cleaner - Intelligent Data Preprocessing Pipeline

## Overview

**Data Cleaner** is a Noventis module that provides an intelligent, automated data preprocessing pipeline. It orchestrates multiple cleaning steps including missing value imputation, outlier handling, categorical encoding, and feature scaling—all with a single, unified interface.

---

## Key Features

### 1. **Complete Pipeline Orchestration**
- Sequential execution of cleaning steps
- Automatic dependency handling
- Flexible step configuration
- Progress tracking and reporting

### 2. **Four Core Components**
- **Imputer**: Intelligent missing value handling
- **Outlier Handler**: Smart outlier detection and treatment
- **Encoder**: Advanced categorical encoding
- **Scaler**: Optimal feature scaling

### 3. **Quality Scoring System**
- Data Quality ROI calculation
- Weighted scoring across dimensions
- Detailed quality metrics
- Before/after comparison

### 4. **Interactive HTML Reports**
- Modern, responsive UI
- Visual comparisons
- Comprehensive statistics
- Easy-to-read summaries

---

## Installation

```bash
pip install noventis
```

---

## Quick Start

### Basic Usage

```python
from noventis.data_cleaner import data_cleaner
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# One-line data cleaning
cleaned_df = data_cleaner(
    data=df,
    target_column='target',
    verbose=True
)
```

### Advanced Usage with Custom Parameters

```python
from noventis.data_cleaner import NoventisDataCleaner

# Initialize with custom parameters
cleaner = NoventisDataCleaner(
    pipeline_steps=['impute', 'outlier', 'encode', 'scale'],
    imputer_params={'method': 'knn', 'n_neighbors': 5},
    outlier_params={'default_method': 'iqr_trim'},
    encoder_params={'method': 'auto'},
    scaler_params={'method': 'auto'},
    verbose=True
)

# Fit and transform
cleaned_df = cleaner.fit_transform(df)

# Generate HTML report
report = cleaner.generate_html_report()
```

---

## Parameters

### `data_cleaner()` Function

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `str` or `pd.DataFrame` | Required | Path to CSV file or DataFrame |
| `target_column` | `str` | `None` | Target variable name for supervised learning |
| `null_handling` | `str` | `'auto'` | Method for handling missing values |
| `outlier_handling` | `str` | `'auto'` | Method for handling outliers |
| `encoding` | `str` | `'auto'` | Method for encoding categorical variables |
| `scaling` | `str` | `'auto'` | Method for feature scaling |
| `verbose` | `bool` | `True` | Print detailed progress information |
| `return_instance` | `bool` | `False` | Return (DataFrame, cleaner_instance) tuple |

### `NoventisDataCleaner()` Class

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline_steps` | `list` | `['impute', 'outlier', 'encode', 'scale']` | Steps to execute in order |
| `imputer_params` | `dict` | `{}` | Parameters for imputer |
| `outlier_params` | `dict` | `{}` | Parameters for outlier handler |
| `encoder_params` | `dict` | `{}` | Parameters for encoder |
| `scaler_params` | `dict` | `{}` | Parameters for scaler |
| `verbose` | `bool` | `False` | Print detailed information |

---

## Pipeline Components

### 1. Imputer (Missing Value Handling)

**Purpose**: Fill or remove missing values intelligently.

**Available Methods**:
- `'auto'`: Mean for numeric, mode for categorical
- `'mean'`: Mean imputation (numeric only)
- `'median'`: Median imputation (numeric only)
- `'mode'`: Most frequent value
- `'knn'`: K-Nearest Neighbors imputation
- `'constant'`: Fill with specific value
- `'ffill'`: Forward fill
- `'bfill'`: Backward fill
- `'drop'`: Drop rows with missing values

**Example**:
```python
imputer_params = {
    'method': 'knn',
    'n_neighbors': 5,
    'columns': ['Age', 'Income']  # Specific columns
}

cleaner = NoventisDataCleaner(imputer_params=imputer_params)
```

**Features**:
- Automatic type detection
- Integer column preservation (automatic rounding)
- Per-column method specification
- Quality metrics tracking

---

### 2. Outlier Handler

**Purpose**: Detect and handle statistical outliers.

**Available Methods**:
- `'auto'`: Intelligent method selection based on data characteristics
- `'iqr_trim'`: Remove outliers using IQR method
- `'quantile_trim'`: Remove outliers using quantile boundaries
- `'winsorize'`: Cap outliers at boundaries
- `'none'`: No outlier handling

**Example**:
```python
outlier_params = {
    'default_method': 'iqr_trim',
    'iqr_multiplier': 1.5,
    'quantile_range': (0.05, 0.95),
    'feature_method_map': {
        'Price': 'winsorize',  # Specific method per column
        'Age': 'iqr_trim'
    }
}

cleaner = NoventisDataCleaner(outlier_params=outlier_params)
```

**Auto Method Selection**:
```python
# Auto chooses based on:
# - Data size < 100 → iqr_trim
# - High skewness (>0.5) → winsorize
# - Normal distribution → quantile_trim
```

**Features**:
- IQR-based detection
- Quantile-based detection
- Winsorization (capping)
- Per-column method specification
- Skewness-aware selection

---

### 3. Encoder (Categorical Encoding)

**Purpose**: Transform categorical variables into numeric representations.

**Available Methods**:
- `'auto'`: Intelligent encoding selection per column
- `'label'`: Label encoding (ordinal)
- `'ohe'`: One-Hot Encoding
- `'target'`: Target encoding (supervised)
- `'ordinal'`: Ordinal encoding with custom mapping
- `'binary'`: Binary encoding
- `'hashing'`: Feature hashing

**Example**:
```python
encoder_params = {
    'method': 'auto',
    'target_column': 'Survived',
    'cv': 5,  # Cross-validation folds for target encoding
    'smooth': 'auto',  # Smoothing parameter
    'columns_to_encode': ['Gender', 'Embarked']  # Specific columns
}

cleaner = NoventisDataCleaner(encoder_params=encoder_params)
```

**Auto Selection Logic**:
```python
# Auto chooses based on:
# - Cardinality == 2 → label
# - Cardinality > 50 → target or hashing
# - Cardinality 15-50 → binary or target
# - Cardinality 3-15 → ohe or target
# - High correlation with target → target encoding
```

**Advanced Features**:
- Cramér's V correlation calculation
- Memory impact analysis
- Cross-validated target encoding
- Automatic dimensionality management
- Handling of unseen categories

**Ordinal Encoding Example**:
```python
encoder_params = {
    'method': 'ordinal',
    'category_mapping': {
        'Education': {
            'High School': 1,
            'Bachelor': 2,
            'Master': 3,
            'PhD': 4
        }
    }
}
```

---

### 4. Scaler (Feature Scaling)

**Purpose**: Standardize numeric feature ranges.

**Available Methods**:
- `'auto'`: Intelligent scaler selection per column
- `'standard'`: StandardScaler (z-score normalization)
- `'minmax'`: MinMaxScaler (0-1 normalization)
- `'robust'`: RobustScaler (median and IQR based)
- `'power'`: PowerTransformer (Box-Cox/Yeo-Johnson)

**Example**:
```python
scaler_params = {
    'method': 'auto',
    'optimize': True,
    'skew_threshold': 2.0,
    'outlier_threshold': 0.01,
    'custom_params': {
        'with_mean': True,
        'with_std': True
    }
}

cleaner = NoventisDataCleaner(scaler_params=scaler_params)
```

**Auto Selection Logic**:
```python
# Auto chooses based on:
# - High skewness (>2.0) → power
# - Has outliers (>1% ratio) → robust
# - Normal distribution → standard
# - Default → standard
```

**Features**:
- Distribution analysis (skewness, normality)
- Outlier detection for scaler selection
- Per-column parameter optimization
- Shapiro-Wilk and Anderson-Darling tests
- Inverse transform support

---

## Quality Scoring System

### Overall Score Calculation

The final quality score is a weighted combination of four dimensions:

```python
Final Score = (Completeness × 40%) + 
              (Consistency × 30%) + 
              (Distribution × 20%) + 
              (Feature Engineering × 10%)
```

### Score Components

#### 1. Completeness Score (40%)
- Measures missing data handling effectiveness
- Based on final missing value percentage
- Higher is better

#### 2. Consistency Score (30%)
- Measures outlier removal effectiveness
- Based on data stability metrics
- Higher is better

#### 3. Distribution Quality (20%)
- Measures skewness improvement
- Rewards distribution normalization
- Based on before/after comparison

#### 4. Feature Engineering Score (10%)
- Evaluates encoding efficiency
- Optimal: 2-5 features per encoded column
- Penalizes excessive feature explosion

### Score Interpretation

| Score Range | Rating | Description |
|-------------|--------|-------------|
| 90-100 | Excellent | Data is ML-ready with minimal issues |
| 80-89 | Very Good | Data is well-prepared with minor issues |
| 70-79 | Good | Data is acceptable with some concerns |
| <70 | Needs Improvement | Significant issues remain |

---

## Use Cases

### Use Case 1: Quick Data Cleaning

```python
# Simple one-liner for basic cleaning
cleaned_df = data_cleaner(
    data='dataset.csv',
    verbose=True
)
```

### Use Case 2: ML Pipeline Preparation

```python
# Prepare data for machine learning
cleaned_df, cleaner = data_cleaner(
    data=df,
    target_column='price',
    null_handling='knn',
    outlier_handling='iqr_trim',
    encoding='auto',
    scaling='standard',
    return_instance=True
)

# Use cleaned data for ML
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    cleaned_df.drop('price', axis=1),
    cleaned_df['price'],
    test_size=0.2
)
```

### Use Case 3: Custom Pipeline with Specific Methods

```python
cleaner = NoventisDataCleaner(
    pipeline_steps=['impute', 'encode', 'scale'],  # Skip outlier handling
    imputer_params={
        'method': {
            'Age': 'median',
            'Salary': 'mean',
            'Category': 'mode'
        }
    },
    encoder_params={
        'method': 'ohe',
        'columns_to_encode': ['Gender', 'City']
    },
    scaler_params={
        'method': 'minmax'
    },
    verbose=True
)

cleaned_df = cleaner.fit_transform(df)
```

### Use Case 4: Iterative Cleaning with Reports

```python
# Step 1: Initial cleaning
cleaner = NoventisDataCleaner(verbose=True)
cleaned_df = cleaner.fit_transform(df)

# Step 2: Review quality
cleaner.display_summary_report()

# Step 3: Generate detailed report
report = cleaner.generate_html_report()

# Step 4: Adjust parameters if needed
if cleaner.quality_score_['final_score_numeric'] < 80:
    # Re-run with different parameters
    cleaner = NoventisDataCleaner(
        imputer_params={'method': 'knn'},
        outlier_params={'default_method': 'winsorize'}
    )
    cleaned_df = cleaner.fit_transform(df)
```

---

## Advanced Features

### Custom Pipeline Steps

```python
# Execute only specific steps
cleaner = NoventisDataCleaner(
    pipeline_steps=['impute', 'scale'],  # Skip outlier and encoding
    verbose=True
)
```

### Per-Component Access

```python
# Access individual components after fitting
cleaner.fit_transform(df)

# Access fitted imputer
imputer = cleaner.imputer_
print(imputer.quality_report_)

# Access fitted encoder
encoder = cleaner.encoder_
print(encoder.get_encoding_info())

# Access fitted scaler
scaler = cleaner.scaler_
print(scaler.analysis_)
```

### Quality Reports

```python
# Get detailed reports per component
impute_report = cleaner.reports_['impute']
outlier_report = cleaner.reports_['outlier']
encode_report = cleaner.reports_['encode']
scale_report = cleaner.reports_['scale']

# Overall quality score
print(cleaner.quality_score_)
```

---

## HTML Report Features

The interactive HTML report includes:

### Overview Tab
- Final quality score with visual gauge
- Score breakdown by dimension
- Initial data profile
- Processing summary
- Data preview table

### Component Tabs (Imputer, Outlier, Encoder, Scaler)
Each tab shows:
- Summary statistics
- Visual comparisons (before/after)
- Method descriptions
- Quality metrics

### Visual Comparisons
- Distribution plots
- Box plots
- Heatmaps (encoding)
- Q-Q plots (scaling)

---

## Best Practices

### 1. Data Exploration First

```python
# Always explore data before cleaning
print(df.info())
print(df.describe())
print(df.isnull().sum())
```

### 2. Start with Auto Mode

```python
# Let the system choose optimal methods
cleaned_df = data_cleaner(data=df, verbose=True)

# Review the report
# Then adjust if needed
```

### 3. Handle Target Separately

```python
# Separate target before cleaning
X = df.drop('target', axis=1)
y = df['target']

# Clean features
cleaner = NoventisDataCleaner()
X_cleaned = cleaner.fit_transform(X, y)

# Recombine
cleaned_df = pd.concat([X_cleaned, y], axis=1)
```

### 4. Check Quality Score

```python
cleaner.fit_transform(df)

score = cleaner.quality_score_['final_score_numeric']
if score < 70:
    print("Warning: Quality score is low. Review the report.")
    cleaner.display_summary_report()
```

### 5. Save Fitted Pipeline

```python
import pickle

# Save fitted cleaner for future use
with open('cleaner_pipeline.pkl', 'wb') as f:
    pickle.dump(cleaner, f)

# Load and use later
with open('cleaner_pipeline.pkl', 'rb') as f:
    loaded_cleaner = pickle.load(f)

new_data_cleaned = loaded_cleaner.transform(new_data)
```

---

## Technical Details


### Memory Considerations

```python
# For large datasets, consider:
# 1. Process in chunks
# 2. Use memory-efficient encodings (hashing, binary)
# 3. Limit KNN neighbors
# 4. Use sampling for outlier detection

encoder_params = {
    'method': 'hashing',  # Memory efficient
}

imputer_params = {
    'method': 'median',  # Faster than KNN
}
```

### Performance Tips

```python
# 1. Disable verbose for faster execution
cleaner = NoventisDataCleaner(verbose=False)

# 2. Skip unnecessary steps
cleaner = NoventisDataCleaner(
    pipeline_steps=['impute', 'scale']
)

# 3. Use simpler methods for large data
outlier_params = {'default_method': 'iqr_trim'}
encoder_params = {'method': 'label'}
```

---

## Troubleshooting

### Issue 1: Memory Error with Encoding

```python
# Problem: Too many unique categories causing OHE explosion

# Solution: Use alternative encoding
encoder_params = {
    'method': 'hashing',  # or 'binary'
}
```

### Issue 2: Target Column Not Found

```python
# Ensure target column name is correct (case-sensitive)
print(df.columns.tolist())

cleaner = NoventisDataCleaner(
    encoder_params={'target_column': 'correct_name'}
)
```

### Issue 3: Negative Quality Score

```python
# This can happen with extreme outlier removal
# Solution: Use winsorize instead of trim

outlier_params = {'default_method': 'winsorize'}
```

### Issue 4: Scaler Issues with Constant Columns

```python
# Remove constant columns before scaling
constant_cols = df.columns[df.nunique() <= 1]
df = df.drop(constant_cols, axis=1)
```

---

## API Reference

### Main Functions

#### `data_cleaner()`
High-level wrapper function for quick cleaning.

#### `NoventisDataCleaner`
Main orchestrator class.

**Methods**:
- `fit(X, y)`: Learn cleaning parameters
- `transform(X)`: Apply cleaning transformations
- `fit_transform(X, y)`: Fit and transform in one step
- `display_summary_report()`: Print console summary
- `generate_html_report()`: Generate interactive HTML report

**Attributes**:
- `imputer_`: Fitted imputer instance
- `outlier_handler_`: Fitted outlier handler instance
- `encoder_`: Fitted encoder instance
- `scaler_`: Fitted scaler instance
- `quality_score_`: Quality score dictionary
- `reports_`: Per-component quality reports

---

## Examples

### Example 1: Titanic Dataset

```python
import pandas as pd
from noventis.data_cleaner import data_cleaner

# Load Titanic data
df = pd.read_csv('titanic.csv')

# Clean data
cleaned_df = data_cleaner(
    data=df,
    target_column='Survived',
    null_handling='auto',
    outlier_handling='iqr_trim',
    encoding='auto',
    scaling='standard',
    verbose=True
)
```

### Example 2: House Prices

```python
# Advanced custom pipeline
cleaner = NoventisDataCleaner(
    imputer_params={'method': 'knn', 'n_neighbors': 10},
    outlier_params={
        'default_method': 'auto',
        'feature_method_map': {
            'LotArea': 'winsorize',
            'SalePrice': 'iqr_trim'
        }
    },
    encoder_params={
        'method': 'auto',
        'target_column': 'SalePrice'
    },
    scaler_params={'method': 'robust'},
    verbose=True
)

cleaned_df = cleaner.fit_transform(df)
report = cleaner.generate_html_report()
```

### Example 3: Time Series Data

```python
# Use forward fill for time series
cleaner = NoventisDataCleaner(
    pipeline_steps=['impute'],  # Only imputation
    imputer_params={'method': 'ffill'},
    verbose=True
)

cleaned_df = cleaner.fit_transform(df)
```

---


**Made with ❤️ by the Noventis Team**