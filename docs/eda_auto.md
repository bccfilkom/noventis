# 🔍 EDA Auto - Automated Exploratory Data Analysis

## Overview

**EDA Auto** adalah modul Noventis yang menyediakan analisis eksploratori data (EDA) otomatis dengan visualisasi komprehensif dan insights mendalam. Modul ini dirancang untuk menghemat waktu dalam fase awal analisis data dengan menghasilkan laporan interaktif yang dapat disesuaikan berdasarkan kebutuhan.

---

## 🎯 Key Features

### 1. **Multi-Personality Reports**
- **Default**: Laporan EDA standar dengan visualisasi lengkap
- **Business**: Dashboard bisnis dengan ROI, Customer Intelligence, dan Priority Matrix
- **Academic**: Dashboard statistik dengan Distribution Test, Correlation Validation, dan Model Diagnostics
- **All**: Kombinasi Business + Academic + Default visualizations

### 2. **Comprehensive Analysis Components**
- Dataset Overview & Statistics
- Target Variable Analysis
- Missing Values Analysis
- Outlier Detection & Visualization
- Numerical Distribution Analysis
- Correlation Analysis dengan VIF (Variance Inflation Factor)
- Interactive HTML Report

### 3. **Advanced Statistical Tests**
- Shapiro-Wilk Normality Test
- Q-Q Plot untuk normalitas
- Multicollinearity Detection (VIF)
- Cross-Validation Model Diagnostics
- Residual Analysis

---

## 📦 Installation

```bash
pip install noventis
```

---

## 🚀 Quick Start

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

## 📋 Parameters

### `NoventisAutoEDA()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` or `str` | Required | DataFrame atau path ke file CSV |
| `target` | `str` | `None` | Nama kolom target untuk supervised learning analysis |
| `personality` | `str` | `'default'` | Jenis laporan: `'default'`, `'business'`, `'academic'`, `'all'` |

### `run()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `show_base_viz` | `bool` | `True` | Tampilkan visualisasi dasar (Overview, Stats, Missing Values, dll.) |

---

## 🎨 Personality Modes

### 1. Default Personality

Laporan EDA standar dengan visualisasi lengkap:

```python
eda = NoventisAutoEDA(data=df, personality='default')
report = eda.run()
```

**Sections included:**
- 📊 Overview
- 🎯 Target Analysis (if target specified)
- 📈 Descriptive Statistics
- ❓ Missing Values
- 📉 Outlier Distribution
- 📊 Numerical Distribution
- 🔗 Correlation Analysis

---

### 2. Business Personality

Dashboard bisnis dengan fokus pada actionable insights:

```python
eda = NoventisAutoEDA(data=df, target='revenue', personality='business')
report = eda.run(show_base_viz=False)  # Hanya business dashboard
```

**Business Dashboard Components:**

#### a. Data Quality ROI
- **Quality Score Gauge**: Skor kualitas data berdasarkan missing values, outliers, dan duplikat
- **KPI Cards**: Missing cells, outliers detected, duplicate rows
- **Top Missing Values Chart**: 5 kolom dengan missing data tertinggi
- **Top Outliers Chart**: 5 kolom dengan outliers terbanyak

#### b. Customer Intelligence
- **Impact Analysis**: Segmentasi berbasis kategori paling berpengaruh terhadap target
- **Revenue Impact Table**: Breakdown kontribusi per segment
- **Pie Chart Visualization**: Proporsi impact tiap segment

#### c. Priority Matrix
- **Feature Impact vs Quality**: Matrix untuk prioritas perbaikan
- **Quadrant Classification**:
  - 🟢 **Focus Here**: High impact + High quality
  - 🟠 **Strategic Fix**: High impact + Low quality (prioritas tinggi!)
  - 🔵 **Easy Win**: Low impact + High quality
  - ⚫ **Low Priority**: Low impact + Low quality

---

### 3. Academic Personality

Dashboard statistik untuk validasi akademis:

```python
eda = NoventisAutoEDA(data=df, target='score', personality='academic')
report = eda.run(show_base_viz=False)
```

**Academic Dashboard Components:**

#### a. Distribution Test Panel
- **Shapiro-Wilk Test**: Uji normalitas untuk top 4 variabel penting
- **Visual Histogram**: Mini histogram untuk setiap variabel
- **Interpretation Badges**:
  - ✓ Normal (p > 0.05)
  - ✗ Non-Normal (p ≤ 0.05)
  - ? Test Failed
- **Smart Variable Selection**: Berdasarkan target correlation, variance, missing rates, dan business keywords

#### b. Correlation Validation Panel
- **Correlation Heatmap**: Matriks korelasi untuk ≤8 features
- **High Correlations List**: Pasangan dengan |r| > 0.5
- **VIF Analysis**: Deteksi multicollinearity
  - ✓ OK: VIF < 5
  - ⚠ MEDIUM: VIF 5-10
  - ⚠ HIGH: VIF > 10 (perlu action!)

#### c. Model Diagnostics Panel
- **Residual Plot**: Residuals vs Fitted values
- **Pattern Detection**: Random vs Pattern detected
- **CV Score Gauge**: Cross-validation accuracy/R² score
  - 🟢 EXCELLENT: ≥80%
  - 🟡 GOOD: 60-80%
  - 🔴 POOR: <60%
- **Top 3 Feature Importance**: Variabel paling berpengaruh

---

### 4. All Personality

Kombinasi lengkap semua dashboard:

```python
eda = NoventisAutoEDA(data=df, target='price', personality='all')
report = eda.run()  # Business + Academic + Base viz
```

---

## 📊 Detailed Features

### Target Variable Analysis

Analisis otomatis berdasarkan tipe target:

**Classification (Binary/Multiclass):**
- Class distribution table & chart
- Imbalance detection
- Count plot visualization

**Regression:**
- Descriptive statistics (mean, std, quartiles)
- Distribution plot (histogram + KDE)
- Box plot untuk outlier detection

```python
# Contoh dengan target classification
eda = NoventisAutoEDA(data=df, target='churn')
report = eda.run()

# Contoh dengan target regression
eda = NoventisAutoEDA(data=df, target='price')
report = eda.run()
```

---

### Missing Values Analysis

**Features:**
- Summary table dengan count & percentage
- Heatmap pola missing values
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
- Q-Q Plot untuk visual normality check
- Shapiro-Wilk Test (untuk n < 5000)
- Interpretation guidance

---

### Correlation Analysis

**Features:**
- Correlation matrix heatmap (untuk ≤30 features)
- Interactive table (untuk >30 features)
- Filter options: Show All, >0.5, >0.7
- Top positive/negative correlations

**Academic mode additional:**
- VIF (Variance Inflation Factor) calculation
- Multicollinearity alerts
- Interpretation recommendations

---

## 💡 Use Cases

### Use Case 1: Quick Data Profiling

```python
# Untuk dataset baru, lihat overview cepat
eda = NoventisAutoEDA(data='new_dataset.csv')
report = eda.run()
```

### Use Case 2: Business Presentation

```python
# Untuk stakeholder meeting, fokus pada business impact
eda = NoventisAutoEDA(
    data=df,
    target='revenue',
    personality='business'
)
report = eda.run(show_base_viz=False)
```

### Use Case 3: Academic Research

```python
# Untuk paper/thesis, validasi statistik
eda = NoventisAutoEDA(
    data=df,
    target='dependent_var',
    personality='academic'
)
report = eda.run(show_base_viz=False)
```

### Use Case 4: Comprehensive Analysis

```python
# Untuk deep dive analysis, gunakan semua fitur
eda = NoventisAutoEDA(
    data=df,
    target='target',
    personality='all'
)
report = eda.run()
```

---

## 🎨 Customization

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
# Built-in threshold untuk kolom (default: 30)
# Untuk >30 kolom, akan menggunakan tabel interaktif
# Untuk ≤30 kolom, akan menggunakan heatmap
```

---

## 📈 Advanced Features

### Smart Variable Selection

EDA Auto secara otomatis memilih top 4 variabel untuk **Distribution Test** berdasarkan:

1. **Target Correlation**: 2 variabel dengan korelasi tertinggi terhadap target
2. **High Variance**: 3 variabel dengan variabilitas tinggi
3. **Data Quality**: 2 variabel dengan missing rates tinggi (5-80%)
4. **Business Keywords**: Variabel dengan nama seperti age, income, price, cost, revenue, score, rating, amount

### Automatic Problem Type Detection

```python
# Binary Classification: detected jika n_unique == 2
# Multiclass: detected jika 2 < n_unique ≤ 25
# Regression: detected jika n_unique > 25
```

### VIF Calculation

```python
# VIF > 10: High multicollinearity (⚠ action needed)
# VIF 5-10: Medium multicollinearity
# VIF < 5: OK
```

---

## ⚙️ Technical Details

### Dependencies

```python
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scipy >= 1.7.0
scikit-learn >= 0.24.0
statsmodels >= 0.12.0
```

### Performance

- **Memory efficient**: Chunked processing untuk large datasets
- **Fast rendering**: Optimized matplotlib plots
- **Responsive**: Interactive filters dengan JavaScript

### Browser Compatibility

- Chrome/Edge: ✓ Full support
- Firefox: ✓ Full support
- Safari: ✓ Full support
- IE11: ✗ Not supported

---

## 🔧 Troubleshooting

### Issue 1: "Target column not found"

```python
# Pastikan target column ada di DataFrame
print(df.columns.tolist())

# Case sensitive!
eda = NoventisAutoEDA(data=df, target='Revenue')  # ✗
eda = NoventisAutoEDA(data=df, target='revenue')  # ✓
```

### Issue 2: Empty report sections

```python
# Jika tidak ada numeric columns
# Pastikan data types sudah benar
df['numeric_col'] = pd.to_numeric(df['numeric_col'], errors='coerce')
```

### Issue 3: Shapiro-Wilk test failed

```python
# Untuk n > 5000, test otomatis menggunakan sample
# Untuk n < 3, test tidak dapat dilakukan
# Solusi: visual assessment dengan Q-Q plot
```

---

## 📝 Best Practices

### 1. Data Preparation

```python
# Clean data sebelum EDA
df = df.drop_duplicates()  # Remove duplicates
df = df.dropna(how='all')  # Remove empty rows
```

### 2. Target Selection

```python
# Untuk supervised learning, ALWAYS specify target
eda = NoventisAutoEDA(data=df, target='target_var')
```

### 3. Personality Choice

```python
# Pilih personality berdasarkan audience:
# - Stakeholders/Business → 'business'
# - Researchers/Students → 'academic'
# - Data Scientists → 'all' atau 'default'
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


## 📚 Examples

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

## 🤝 Contributing

Contributions are welcome! Please check our [Contributing Guidelines](../CONTRIBUTING.md).

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

## 💬 Support

- 📧 Email: support@noventis.dev
- 💬 Discord: [Join our community](https://discord.gg/noventis)
- 📖 Documentation: [docs.noventis.dev](https://docs.noventis.dev)
- 🐛 Issues: [GitHub Issues](https://github.com/noventis/noventis/issues)

---

**Made with ❤️ by the Noventis Team**