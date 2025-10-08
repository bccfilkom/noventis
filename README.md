<div align="center">
  
<h1 align="center">
  <img src="https://github.com/user-attachments/assets/8d64296a-55f2-4eb4-bc55-275f5d75ef75" alt="Noventis Logo" width="40" height="40" style="vertical-align: middle;"/>
  Noventis
</h1>


### Intelligent Automation for Your Data Analysis

[![PyPI version](https://badge.fury.io/py/noventis.svg)](https://badge.fury.io/py/noventis)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Website](https://noventis.dev) • [Documentation](https://docs.noventis.dev) 


<img width="1247" height="637" alt="Screenshot From 2025-10-02 09-44-31" src="https://github.com/user-attachments/assets/264f13ce-4f5a-477a-a89d-73f0c9a585bd" />



</div>

---

## 🚀 Overview

**Noventis** is a powerful Python library designed to revolutionize your data analysis workflow through intelligent automation. Built with modern data scientists and analysts in mind, Noventis provides cutting-edge tools for automated exploratory data analysis, predictive modeling, and data cleaning—all with minimal code.

### ✨ Key Features

- **🔍 EDA Auto** - Automated exploratory data analysis with comprehensive visualizations and statistical insights
- **🎯 Predictor** - Intelligent ML model selection and training with automated hyperparameter tuning
- **🧹 Data Cleaner** - Smart data preprocessing and cleaning with advanced imputation strategies
- **⚡ Fast & Efficient** - Optimized for performance with large datasets
- **📊 Rich Visualizations** - Beautiful, publication-ready charts and reports
- **🔧 Highly Customizable** - Fine-tune every aspect to match your needs

---

## 📦 Installation

### Quick Installation

```bash
pip install noventis
```

### Install with Optional Dependencies

```bash
# For handling imbalanced datasets
pip install noventis[imbalanced]

# For development
pip install noventis[dev]

# Install all optional dependencies
pip install noventis[all]
```

### Install from Source

```bash
git clone https://github.com/yourusername/noventis.git
cd noventis
pip install -e .
```

### Verify Installation

```python
import noventis
print(noventis.__version__)
noventis.print_info()  # Show detailed installation info
```

---

## 🎯 Quick Start

### 1️⃣ Data Cleaner
Get started with intelligent data preprocessing and cleaning.

```python
import pandas as pd
from noventis.data_cleaner import AutoCleaner

# Load your data
df = pd.read_csv('your_data.csv')

# Automatic data cleaning
cleaner = AutoCleaner()
df_clean = cleaner.fit_transform(df)

# The cleaned data is ready for analysis!
print(df_clean.info())
```

👉 [Read the Data Cleaner Guide](docs/data_cleaner.md)

### 2️⃣ EDA Auto
Automatically generate comprehensive exploratory data analysis reports.

```python
from noventis.eda_auto import EDAuto

# Create EDA report
eda = EDAuto(df_clean)

# Generate comprehensive analysis
eda.generate_report()

# Show specific analyses
eda.show_distributions()
eda.show_correlations()
eda.show_missing_patterns()
```

👉 [Read the EDA Auto Guide](docs/eda_auto.md)

### 3️⃣ Predictor
Build and train machine learning models with automated optimization.

```python
from noventis.predictor import PredictorAuto

# Prepare data
X = df_clean.drop('target', axis=1)
y = df_clean['target']

# Automatic model training
predictor = PredictorAuto()
predictor.fit(X, y, task='classification')

# Make predictions
predictions = predictor.predict(X_test)

# Get model performance
print(predictor.get_metrics())
```

👉 [Read the Predictor Guide](docs/predictor.md)

### 4️⃣ Complete Pipeline Example

```python
import pandas as pd
from noventis.data_cleaner import AutoCleaner
from noventis.eda_auto import EDAuto
from noventis.predictor import PredictorAuto

# 1. Load data
df = pd.read_csv('your_data.csv')

# 2. Clean data
cleaner = AutoCleaner()
df_clean = cleaner.fit_transform(df)

# 3. Explore data
eda = EDAuto(df_clean)
eda.generate_report()

# 4. Train model
X = df_clean.drop('target', axis=1)
y = df_clean['target']

predictor = PredictorAuto()
predictor.fit(X, y, task='classification')

# 5. Evaluate
print(f"Model Accuracy: {predictor.score(X_test, y_test):.2%}")
```

---

## 📚 Core Modules

### 🧹 Data Cleaner

Intelligent data preprocessing and cleaning with advanced strategies:

- **Missing Data Handling** - Multiple imputation strategies (mean, median, KNN, iterative)
- **Outlier Treatment** - Statistical and ML-based detection (IQR, Z-score, Isolation Forest)
- **Feature Scaling** - Normalization and standardization techniques
- **Encoding** - Automatic categorical variable encoding (One-Hot, Label, Target)
- **Data Type Detection** - Intelligent type inference and conversion
- **Duplicate Removal** - Smart duplicate detection and handling

[Learn more →](docs/data_cleaner.md)

### 🔍 EDA Auto

Comprehensive exploratory data analysis automation:

- **Statistical Summary** - Descriptive statistics for all features
- **Distribution Analysis** - Histograms, KDE plots, and normality tests
- **Correlation Analysis** - Heatmaps and correlation matrices
- **Missing Data Analysis** - Visualization and patterns of missing values
- **Outlier Detection** - Automatic identification of anomalies
- **Feature Relationships** - Scatter plots and pairwise analysis

[Learn more →](docs/eda_auto.md)

### 🎯 Predictor

Automated machine learning with intelligent model selection:

- **Auto Model Selection** - Automatically selects the best algorithm for your data
- **Hyperparameter Tuning** - Optimizes model parameters using advanced search algorithms
- **Feature Engineering** - Creates and selects relevant features automatically
- **Cross-Validation** - Robust model evaluation with k-fold validation
- **Model Explainability** - SHAP values and feature importance analysis
- **Ensemble Methods** - Combines multiple models for better performance

**Supported Algorithms:**
- Scikit-learn: Random Forest, Gradient Boosting, Logistic Regression, SVM
- XGBoost: Extreme Gradient Boosting
- LightGBM: Light Gradient Boosting Machine
- CatBoost: Categorical Boosting
- And many more...

[Learn more →](docs/auto.md)

---

## 🛠️ Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB+ recommended for large datasets)
- Windows, macOS, or Linux

### Core Dependencies
Noventis automatically installs these dependencies:

- **Data Processing**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, xgboost, lightgbm, catboost
- **AutoML**: optuna, flaml, shap
- **Feature Engineering**: category_encoders, statsmodels

See [requirements.txt](requirements.txt) for complete list.

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute

1. **🐛 Report Bugs** - Found a bug? [Open an issue](https://github.com/yourusername/noventis/issues)
2. **💡 Suggest Features** - Have ideas? We'd love to hear them!
3. **📖 Improve Documentation** - Help us make the docs better
4. **🔧 Submit Pull Requests** - Fix bugs or add features

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/noventis.git
cd noventis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/

# Run linting
flake8 noventis/
black noventis/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 👥 Contributors

This project exists thanks to all the people who contribute:

| Contributor            | Role              |
| ---------------------- | ----------------- |
| **Richard**            | Product Manager   |
| **Fatoni Murfids**     | Product Manager   |
| **Orie Abyan Maulana** | Data Analyst      |
| **Grace Wahyuni**      | Data Analyst      |
| **Ahmad Nafi M.**      | Data Scientist    |
| **Alexander Angelo**   | Data Scientist    |
| **Rimba Nevada**       | Data Scientist    |
| **Jason Surya Winata** | Frontend Engineer |
| **Daffa**              | Product Designer  |


### Special Thanks

A huge thank you to the maintainers of our dependencies:
- pandas, numpy, scikit-learn, and the entire Python scientific computing community
- XGBoost, LightGBM, and CatBoost teams for excellent gradient boosting libraries
- Optuna and FLAML teams for amazing AutoML frameworks

---

## 📂 Project Structure

The folder structure of **Noventis** project:

```bash
.
├── 📁 dataset_for_examples/     # Sample datasets for testing
├── 📁 docs/                     # Documentation files
├── 📁 examples/                 # Example notebooks and scripts
├── 📁 noventis/                 # Main library code
│   ├── 📁 __pycache__/
│   ├── 📁 asset/               # Asset files (if any)
│   ├── 📁 core/                # Core functionality
│   ├── 📁 data_cleaner/        # Data cleaning module
│   │   ├── 📄 __init__.py
│   │   ├── 📄 auto.py
│   │   ├── 📄 data_quality.py
│   │   ├── 📄 encoding.py
│   │   ├── 📄 imputing.py
│   │   ├── 📄 orchestrator.py
│   │   ├── 📄 outlier_handling.py
│   │   └── 📄 scaling.py
│   ├── 📁 eda_auto/            # EDA automation module
│   │   ├── 📄 __init__.py
│   │   └── 📄 eda_auto.py
│   ├── 📁 predictor/           # Prediction module
│   │   ├── 📄 __init__.py
│   │   ├── 📄 auto.py
│   │   └── 📄 manual.py
│   └── 📄 __init__.py          # Main package init
├── 📁 noventis.egg-info/       # Package metadata
│   ├── 📄 dependency_links.txt
│   ├── 📄 PKG-INFO
│   ├── 📄 SOURCES.txt
│   └── 📄 top_level.txt
├── 📁 tests/                   # Unit tests
├── 📄 .gitignore               # Git ignore rules
├── 📄 LICENSE                  # MIT License
├── 📄 MANIFEST.in              # Package manifest
├── 📄 pyproject.toml           # Modern Python packaging config
├── 📄 README.md                # This file
├── 📄 requirements.txt         # Production dependencies
├── 📄 requirements-dev.txt     # Development dependencies
└── 📄 setup.py                 # Package setup script
```

### 📌 Notes
- The `noventis/` folder contains the **main library code**
- The `tests/` folder is dedicated to **unit testing and integration testing**
- `setup.py` and `pyproject.toml` are used for **packaging and distribution**
- `requirements.txt` lists the **external dependencies** needed for the project

🚀 With this structure, the project is ready for development, testing, and publishing on **PyPI or GitHub**.

---

## 🔧 Troubleshooting

### Common Issues

**Problem**: `ModuleNotFoundError: No module named 'noventis'`
```bash
# Solution: Reinstall the package
pip uninstall noventis
pip install noventis
```

**Problem**: Dependencies conflict
```bash
# Solution: Create a fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install noventis
```

**Problem**: Import errors after installation
```python
# Solution: Verify installation
import noventis
print(noventis.__version__)
noventis.print_info()  # Check all dependencies
```

### Getting Help

- 📖 [Documentation](https://docs.noventis.dev)
- 🐛 [GitHub Issues](https://github.com/bcc/noventis/issues)

---

## 📊 Performance Benchmarks

Noventis is optimized for speed and efficiency:

| Dataset Size | Data Cleaning | EDA Generation | Model Training |
|-------------|---------------|----------------|----------------|
| 10K rows    | < 1 second    | 2-3 seconds   | 5-10 seconds   |
| 100K rows   | 2-3 seconds   | 10-15 seconds | 30-60 seconds  |
| 1M rows     | 10-15 seconds | 1-2 minutes   | 5-10 minutes   |

*Benchmarks run on: Intel i7-10700K, 32GB RAM, Python 3.10*

---

## 🗺️ Roadmap

### Current Version (v0.1.0)
- ✅ Core data cleaning functionality
- ✅ Automated EDA
- ✅ Basic predictor with AutoML

### Upcoming Features (v0.2.0)
- 🚧 Deep learning integration
- 🚧 Time series forecasting
- 🚧 Advanced feature engineering
- 🚧 Model deployment tools
- 🚧 Web-based dashboard

### Future Plans (v1.0.0)
- 📅 GPU acceleration
- 📅 Distributed computing support
- 📅 Real-time data processing
- 📅 Cloud integration (AWS, GCP, Azure)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

Noventis uses several open-source libraries. We are grateful to their maintainers:

- **Data Processing**: pandas (BSD), numpy (BSD), scipy (BSD)
- **Visualization**: matplotlib (PSF), seaborn (BSD)
- **Machine Learning**: scikit-learn (BSD), xgboost (Apache 2.0), lightgbm (MIT), catboost (Apache 2.0)
- **AutoML**: optuna (MIT), flaml (MIT), shap (MIT)
- **Feature Engineering**: category_encoders (BSD), statsmodels (BSD)

All dependencies are licensed under permissive open-source licenses (BSD, MIT, Apache 2.0).

---

## 📚 Citation

If you use Noventis in your research, please cite:

```bibtex
@software{noventis2025,
  author = {Noventis Team},
  title = {Noventis: Intelligent Automation for Data Analysis},
  year = {2025},
  url = {https://github.com/bccfilkom/noventis}
}
```

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=bccfilkom/noventis&type=Date)](https://star-history.com/#yourusername/noventis&Date)

---


<div align="center">

Made with ❤️ by [Noventis Team](https://noventis.dev)

If you find Noventis useful, please consider giving it a ⭐ on [GitHub](https://github.com/yourusername/noventis)!
