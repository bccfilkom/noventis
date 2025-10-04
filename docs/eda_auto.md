# EDA Auto - Automated Exploratory Data Analysis

## Overview

**EDA Auto** is a Noventis module that provides automated exploratory data analysis (EDA) with comprehensive visualizations and deep insights. This module is designed to save time during the initial data analysis phase by generating interactive reports that can be customized based on your needs.

---

## Key Features

### 1. **Multi-Personality Reports**
- **Default**: Standard EDA report with complete visualizations
- **Business**: Business dashboard with ROI, Customer Intelligence, and Priority Matrix
- **Academic**: Statistical dashboard with Distribution Test, Correlation Validation, and Model Diagnostics
- **All**: Combination of Business + Academic + Default visualizations

### 2. **Comprehensive Analysis Components**
- Dataset Overview & Statistics
- Target Variable Analysis
- Missing Values Analysis
- Outlier Detection & Visualization
- Numerical Distribution Analysis
- Correlation Analysis with VIF (Variance Inflation Factor)
- Interactive HTML Report

### 3. **Advanced Statistical Tests**
- Shapiro-Wilk Normality Test
- Q-Q Plot for normality assessment
- Multicollinearity Detection (VIF)
- Cross-Validation Model Diagnostics
- Residual Analysis

---

## Installation

```bash
pip install noventis
```

---

## Quick Start

### Basic Usage

```python
from noventis.eda_auto import NoventisAutoEDA
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize EDA Auto
eda = NoventisAutoEDA(
    data=df,
    target='target_column',  # Optional
    personality='default'
)

# Generate report
report = eda.run(show_base_viz=True)
```

---

## Parameters

### `NoventisAutoEDA()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` or `str` | Required | DataFrame or path to CSV file |
| `target` | `str` | `None` | Target column name for supervised learning analysis |
| `personality` | `str` | `'default'` | Report type: `'default'`, `'business'`, `'academic'`, `'all'` |

### `run()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `show_base_viz` | `bool` | `True` | Display base visualizations (Overview, Stats, Missing Values, etc.) |

---

## Personality Modes

### 1. Default Personality

Standard EDA report with complete visualizations:

```python
eda = NoventisAutoEDA(data=df, personality='default')
report = eda.run()
```

**Sections included:**
- Overview
- Target Analysis (if target specified)
- Descriptive Statistics
- Missing Values
- Outlier Distribution
- Numerical Distribution
- Correlation Analysis

---

### 2. Business Personality

Business dashboard focused on actionable insights:

```python
eda = NoventisAutoEDA(data=df, target='revenue', personality='business')
report = eda.run(show_base_viz=False)  # Only business dashboard
```

**Business Dashboard Components:**

#### a. Data Quality ROI
- **Quality Score Gauge**: Data quality score based on missing values, outliers, and duplicates
- **KPI Cards**: Missing cells, outliers detected, duplicate rows
- **Top Missing Values Chart**: Top 5 columns with highest missing data
- **Top Outliers Chart**: Top 5 columns with most outliers

#### b. Customer Intelligence
- **Impact Analysis**: Segmentation based on most influential category on target
- **Revenue Impact Table**: Contribution breakdown per segment
- **Pie Chart Visualization**: Impact proportion per segment

#### c. Priority Matrix
- **Feature Impact vs Quality**: Matrix for prioritizing improvements
- **Quadrant Classification**:
  - **Focus Here**: High impact + High quality
  - **Strategic Fix**: High impact + Low quality (high priority!)
  - **Easy Win**: Low impact + High quality
  - **Low Priority**: Low impact + Low quality

---

### 3. Academic Personality

Statistical dashboard for academic validation:

```python
eda = NoventisAutoEDA(data=df, target='score', personality='academic')
report = eda.run(show_base_viz=False)
```

**Academic Dashboard Components:**

#### a. Distribution Test Panel
- **Shapiro-Wilk Test**: Normality test for top 4 important variables
- **Visual Histogram**: Mini histogram for each variable
- **Interpretation Badges**:
  - Normal (p > 0.05)
  - Non-Normal (p ≤ 0.05)
  - Test Failed
- **Smart Variable Selection**: Based on target correlation, variance, missing rates, and business keywords

#### b. Correlation Validation Panel
- **Correlation Heatmap**: Correlation matrix for ≤8 features
- **High Correlations List**: Pairs with |r| > 0.5
- **VIF Analysis**: Multicollinearity detection
  - OK: VIF < 5
  - MEDIUM: VIF 5-10
  - HIGH: VIF > 10 (action needed!)

#### c. Model Diagnostics Panel
- **Residual Plot**: Residuals vs Fitted values
- **Pattern Detection**: Random vs Pattern detected
- **CV Score Gauge**: Cross-validation accuracy/R² score
  - EXCELLENT: ≥80%
  - GOOD: 60-80%
  - POOR: <60%
- **Top 3 Feature Importance**: Most influential variables

---

### 4. All Personality

Complete combination of all dashboards:

```python
eda = NoventisAutoEDA(data=df, target='price', personality='all')
report = eda.run()  # Business + Academic + Base viz
```

---

## Detailed Features

### Target Variable Analysis

Automatic analysis based on target type:

**Classification (Binary/Multiclass):**
- Class distribution table & chart
- Imbalance detection
- Count plot visualization

**Regression:**
- Descriptive statistics (mean, std, quartiles)
- Distribution plot (histogram + KDE)
- Box plot for outlier detection

```python
# Example with classification target
eda = NoventisAutoEDA(data=df, target='churn')
report = eda.run()

# Example with regression target
eda = NoventisAutoEDA(data=df, target='price')
report = eda.run()
```

---

### Missing Values Analysis

**Features:**
- Summary table with count & percentage
- Heatmap of missing value patterns
- Visual pattern detection

**Interpretation:**
- MCAR (Missing Completely At Random)
- MAR (Missing At Random)
- MNAR (Missing Not At Random)

---

### Outlier Detection

**Method:** IQR (Interquartile Range)

```
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

**Output:**
- Box plots per numeric variable
- Outlier count & percentage
- Bounds information

---

### Numerical Distribution

**Analysis per variable:**
- Histogram + KDE plot
- Skewness calculation
- Distribution label (Normal/Skewed)

**Academic mode additional:**
- Q-Q Plot for visual normality check
- Shapiro-Wilk Test (for n < 5000)
- Interpretation guidance

---

### Correlation Analysis

**Features:**
- Correlation matrix heatmap (for ≤30 features)
- Interactive table (for >30 features)
- Filter options: Show All, >0.5, >0.7
- Top positive/negative correlations

**Academic mode additional:**
- VIF (Variance Inflation Factor) calculation
- Multicollinearity alerts
- Interpretation recommendations

---

## Use Cases

### Use Case 1: Quick Data Profiling

```python
# For new datasets, get quick overview
eda = NoventisAutoEDA(data='new_dataset.csv')
report = eda.run()
```

### Use Case 2: Business Presentation

```python
# For stakeholder meetings, focus on business impact
eda = NoventisAutoEDA(
    data=df,
    target='revenue',
    personality='business'
)
report = eda.run(show_base_viz=False)
```

### Use Case 3: Academic Research

```python
# For papers/thesis, statistical validation
eda = NoventisAutoEDA(
    data=df,
    target='dependent_var',
    personality='academic'
)
report = eda.run(show_base_viz=False)
```

### Use Case 4: Comprehensive Analysis

```python
# For deep dive analysis, use all features
eda = NoventisAutoEDA(
    data=df,
    target='target',
    personality='all'
)
report = eda.run()
```

---

## Customization

### Report Height

```python
# Adjust report height (default: 800px)
report_html = eda.generate_html_report(
    report_height=1000,
    show_base_viz=True
)
```

### Correlation Threshold

```python
# Built-in threshold for columns (default: 30)
# For >30 columns, will use interactive table
# For ≤30 columns, will use heatmap
```

---

## Advanced Features

### Smart Variable Selection

EDA Auto automatically selects top 4 variables for **Distribution Test** based on:

1. **Target Correlation**: 2 variables with highest correlation to target
2. **High Variance**: 3 variables with high variability
3. **Data Quality**: 2 variables with high missing rates (5-80%)
4. **Business Keywords**: Variables with names like age, income, price, cost, revenue, score, rating, amount

### Automatic Problem Type Detection

```python
# Binary Classification: detected if n_unique == 2
# Multiclass: detected if 2 < n_unique ≤ 25
# Regression: detected if n_unique > 25
```

### VIF Calculation

```python
# VIF > 10: High multicollinearity (action needed)
# VIF 5-10: Medium multicollinearity
# VIF < 5: OK
```

---



## Troubleshooting

### Issue 1: "Target column not found"

```python
# Ensure target column exists in DataFrame
print(df.columns.tolist())

# Case sensitive!
eda = NoventisAutoEDA(data=df, target='Revenue')  # Wrong
eda = NoventisAutoEDA(data=df, target='revenue')  # Correct
```

### Issue 2: Empty report sections

```python
# If no numeric columns exist
# Ensure data types are correct
df['numeric_col'] = pd.to_numeric(df['numeric_col'], errors='coerce')
```

### Issue 3: Shapiro-Wilk test failed

```python
# For n > 5000, test automatically uses sample
# For n < 3, test cannot be performed
# Solution: visual assessment with Q-Q plot
```

---

## Best Practices

### 1. Data Preparation

```python
# Clean data before EDA
df = df.drop_duplicates()  # Remove duplicates
df = df.dropna(how='all')  # Remove empty rows
```

### 2. Target Selection

```python
# For supervised learning, ALWAYS specify target
eda = NoventisAutoEDA(data=df, target='target_var')
```

### 3. Personality Choice

```python
# Choose personality based on audience:
# - Stakeholders/Business → 'business'
# - Researchers/Students → 'academic'
# - Data Scientists → 'all' or 'default'
```

### 4. Report Export

```python
# Save report to HTML file
from IPython.display import HTML

report = eda.run()

# In Jupyter/Colab, report will display automatically
# To save:
with open('eda_report.html', 'w', encoding='utf-8') as f:
    f.write(report.data)
```

---

## Roadmap

Future enhancements planned:

- Time series analysis support
- Categorical feature analysis enhancement
- Automated feature engineering suggestions
- Export to PDF/PowerPoint
- Custom theme support
- Multilingual reports (ID, EN, etc.)

---

## Examples

### Example 1: Titanic Dataset

```python
import pandas as pd
from noventis.eda_auto import NoventisAutoEDA

# Load Titanic dataset
df = pd.read_csv('titanic.csv')

# Business analysis
eda = NoventisAutoEDA(
    data=df,
    target='Survived',
    personality='business'
)
report = eda.run()
```

### Example 2: House Prices

```python
# Load house prices data
df = pd.read_csv('house_prices.csv')

# Academic statistical validation
eda = NoventisAutoEDA(
    data=df,
    target='SalePrice',
    personality='academic'
)
report = eda.run(show_base_viz=False)
```

### Example 3: Customer Churn

```python
# Load customer data
df = pd.read_csv('customer_churn.csv')

# Comprehensive analysis
eda = NoventisAutoEDA(
    data=df,
    target='Churn',
    personality='all'
)
report = eda.run()
```

---



**Made with ❤️ by the Noventis Team**