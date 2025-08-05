import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer

def handle_nulls(df, method=None, n_neighbors=5):
    """
    Handle missing values in a DataFrame (numerical & categorical).
    - Auto-detects column types.
    - KNN applied once for all numeric columns (efficient).
    - Detects simple ordinal columns (few unique categories).
    - Integer columns: mean is rounded *before* imputation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to clean.
    method : str, dict, or None
        - str: same method for all columns ('mean', 'median', 'mode', 'knn', 'drop').
        - dict: per-column method, e.g. {'Age': 'median', 'Department': 'mode'}.
        - None: auto (mean for numeric, mode for categorical/ordinal).
    n_neighbors : int
        Neighbors for KNN (numeric only).

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with NaNs handled.
    """
    df = df.copy()

    # Identify categorical and numeric columns
    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(exclude='object').columns

    # Detect ordinal (categorical with few unique values)
    ordinal_cols = [col for col in cat_cols if df[col].nunique() <= 10]

    # --- If KNN: apply once for all numeric columns ---
    if method == 'knn':
        if len(num_cols) > 0:
            knn_imputer = KNNImputer(n_neighbors=n_neighbors)
            df[num_cols] = knn_imputer.fit_transform(df[num_cols])
            
        # For categorical → mode
        for col in cat_cols:
            imp = SimpleImputer(strategy='most_frequent')
            df[[col]] = imp.fit_transform(df[[col]])
        return df

    # --- Handle column by column for other methods ---
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue

        # -- if column is skipped (method[col] = None)
        if isinstance(method, dict) and col in method and method[col] is None:
            continue  # not processed
        
        # Determine method (per-column dict or global)
        if isinstance(method, dict):
            chosen_method = method.get(col, None)
        else:
            chosen_method = method

        # Auto fallback
        if chosen_method is None:
            if col in num_cols:
                chosen_method = 'mean'
            elif col in ordinal_cols:
                chosen_method = 'mode'
            else:
                chosen_method = 'mode'

        # Apply imputations
        if chosen_method == 'mean':
            if col in num_cols:
                # Deteksi apakah semua nilai valid integer (meskipun dtype float karena NaN)
                is_effectively_integer = np.all(df[col].dropna() % 1 == 0)
                
                if is_effectively_integer:
                    mean_val = df[col].mean(skipna=True)
                    mean_val = round(mean_val)   
                    imp = SimpleImputer(strategy='constant', fill_value=mean_val)
                    df[[col]] = imp.fit_transform(df[[col]])
                else:
                    imp = SimpleImputer(strategy='mean')
                    df[[col]] = imp.fit_transform(df[[col]])
            else:
                imp = SimpleImputer(strategy='most_frequent')
                df[[col]] = imp.fit_transform(df[[col]])

        elif chosen_method == 'median':
            if col in num_cols:
                imp = SimpleImputer(strategy='median')
                df[[col]] = imp.fit_transform(df[[col]])
            else:
                imp = SimpleImputer(strategy='most_frequent')
                df[[col]] = imp.fit_transform(df[[col]])

        elif chosen_method == 'mode':
            imp = SimpleImputer(strategy='most_frequent')
            df[[col]] = imp.fit_transform(df[[col]])

        elif chosen_method == 'drop':
            df.dropna(subset=[col], inplace=True)

        else:
            raise ValueError(f"Unknown method: {chosen_method}")

    return df