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
```

---

## 🎯 Quick Start

### Data Cleaner
Get started with intelligent data preprocessing and cleaning.

👉 [Read the Data Cleaner Guide](docs/data_cleaner.md)

### EDA Auto
Automatically generate comprehensive exploratory data analysis reports.

👉 [Read the EDA Auto Guide](docs/eda_auto.md)

### Predictor
Build and train machine learning models with automated optimization.

👉 [Read the Predictor Guide](docs/predictor.md)

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

[Learn more →](docs/auto.md)

---


## 👥 Contributors

This project exists thanks to all the people who contribute:

| Contributor | Role |
|------------|------|
| **Orie Abyan Maulana** | Data Analyst |
| **Grace Wahyuni** | Data Analyst |
| **Ahmad Nafi M.** | Data Scientist |
| **Alexander Angelo** | Data Scientist |
| **Rimba Nevada** | Data Scientist |
| **Jason Surya Winata** | Frontend Engineer |
| **Daffa** | Product Designer |

We welcome contributions from the community! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

# 📂 Project Structure
The folder structure of **Noventis** project:

```bash
.
├── 📁 dataset_for_examples/
├── 📁 docs/
├── 📁 examples/
├── 📁 noventis/
│   ├── 📁 __pycache__/
│   ├── 📁 asset/
│   ├── 📁 core/
│   ├── 📁 data_cleaner/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 auto.py
│   │   ├── 📄 data_quality.py
│   │   ├── 📄 encoding.py
│   │   ├── 📄 imputing.py
│   │   ├── 📄 outlier_handling.py
│   │   └── 📄 scaling.py
│   ├── 📁 eda_auto/
│   │   ├── 📄 __init__.py
│   │   └── 📄 eda_auto.py
│   ├── 📁 predictor/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 auto.py
│   │   └── 📄 manual.py
│   └── 📄 __init__.py
├── 📁 noventis.egg-info/
│   ├── 📄 dependency_links.txt
│   ├── 📄 PKG-INFO
│   ├── 📄 SOURCES.txt
│   └── 📄 top_level.txt
├── 📁 tests/
├── 📄 .gitignore
├── 📄 LICENSE
├── 📄 project_folder.txt
├── 📄 pyproject.toml
├── 📄 README.md
├── 📄 requirement.txt
└── 📄 setup.py

```

## 📌 Notes
- The `noventis/` folder contains the **main library code**.  
- The `tests/` folder is dedicated to **unit testing and integration testing**.  
- `setup.py` and `pyproject.toml` are used for **packaging and distribution**.  
- `requirement.txt` lists the **external dependencies** needed for the project.  

🚀 With this structure, the project is ready for development, testing, and publishing on **PyPI or GitHub**.  

---


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by [Noventis Team](https://noventis.dev)

If you find Noventis useful, please consider giving it a ⭐ on [GitHub](https://github.com/yourusername/noventis)!

</div>
