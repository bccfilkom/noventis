import pandas as pd
import numpy as np
from scipy.stats import skew

def assess_data_quality(df: pd.DataFrame) -> dict:
    """
    Analyzes a DataFrame and provides comprehensive quality scores across various aspects.

    Args:
        df (pd.DataFrame): The DataFrame to be analyzed.

    Returns:
        dict: A dictionary containing detailed quality scores.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    results = {}
    total_cells = df.size
    total_rows = len(df)
    total_cols = len(df.columns)
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # 1. Imputation Quality (Completeness)
    missing_cells = df.isnull().sum().sum()
    completion_score = (1 - (missing_cells / total_cells)) * 100 if total_cells > 0 else 100
    results['completeness'] = {
        "score": f"{completion_score:.2f}%",
        "description": "Percentage of data cells that are filled (not missing)."
    }

    # 2. Outlier Quality
    if not numeric_cols.empty:
        outlier_rows = set()
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df.index[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_rows.update(outliers)
        
        inlier_score = (1 - (len(outlier_rows) / total_rows)) * 100 if total_rows > 0 else 100
        results['outlier_quality'] = {
            "score": f"{inlier_score:.2f}%",
            "description": f"Percentage of rows without outliers (based on IQR method across {len(numeric_cols)} numeric columns)."
        }
    else:
        results['outlier_quality'] = {"score": "N/A", "description": "No numeric columns to analyze."}

    # 3. Scaling Quality (Skewness)
    if not numeric_cols.empty:
        avg_abs_skew = df[numeric_cols].apply(lambda x: abs(skew(x.dropna()))).mean()
        skew_score = max(0, 100 - (avg_abs_skew * 20))
        results['distribution_quality'] = {
            "score": f"{skew_score:.2f}/100",
            "description": f"Data symmetry score (avg. absolute skewness: {avg_abs_skew:.2f}). Higher is better."
        }
    else:
        results['distribution_quality'] = {"score": "N/A", "description": "No numeric columns to analyze."}

    # 4. Encoding Analysis (Need & Cardinality)
    if total_cols > 0:
        encoding_needed_pct = len(categorical_cols) / total_cols * 100
        results['encoding_need'] = {
            "score": f"{encoding_needed_pct:.2f}%",
            "description": f"Percentage of columns ({len(categorical_cols)} of {total_cols}) that are categorical."
        }
        if not categorical_cols.empty:
            avg_cardinality = df[categorical_cols].nunique().mean()
            results['cardinality_complexity'] = {
                "score": f"{avg_cardinality:.2f}",
                "description": "Average number of unique values in categorical columns. Lower is simpler."
            }
        else:
            results['cardinality_complexity'] = {"score": "N/A", "description": "No categorical columns."}
    else:
        results['encoding_need'] = {"score": "0.00%", "description": "No columns found."}

    # 5. Data Type Purity
    suspicious_cols = 0
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        if pd.to_numeric(df[col], errors='coerce').notna().sum() / df[col].notna().sum() > 0.8:
            suspicious_cols += 1 # If >80% of non-null values can be numeric, it's suspicious
            
    purity_score = (1 - (suspicious_cols / total_cols)) * 100 if total_cols > 0 else 100
    results['datatype_purity'] = {
        "score": f"{purity_score:.2f}%",
        "description": f"Percentage of columns with appropriate data types ({suspicious_cols} suspicious columns found)."
    }

    return results
