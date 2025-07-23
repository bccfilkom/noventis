import pandas as pd
from sklearn.impute import KNNImputer

def null_handling(df, columns=None, method=None, n_neighbors=3, rmvdup=True):
    """
    Handle missing values in specified columns of a DataFrame using various strategies.

    Parameters:
    - df: pd.DataFrame
    - columns: list of str, columns to process (default: all with nulls)
    - method: str, method to handle nulls ('mean', 'median', 'mode', 'drop', 'knn')
    - n_neighbors: int, only used when method is 'knn'
    - rmvdup: bool, remove duplicates before processing

    Returns:
    - pd.DataFrame with nulls handled
    """
    df = df.copy()
    
    if rmvdup:
        df = df.drop_duplicates()

    # Automatically detect columns with null if not specified
    if columns is None:
        columns = df.columns[df.isnull().any()].tolist()

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' doesn't exist in the DataFrame.")
        
        # Decide method to use
        col_method = (method or "").lower()
        if not col_method:
            col_method = "mean" if pd.api.types.is_numeric_dtype(df[col]) else "mode"

        if col_method == "drop":
            df = df.dropna(subset=[col])
        elif col_method == "knn":
            if df[col].isnull().all():
                raise ValueError(f"KNN can't be applied on column '{col}' with all values missing.")
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df[[col]] = imputer.fit_transform(df[[col]])
        elif col_method == "mean":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                raise TypeError(f"Cannot use 'mean' on non-numeric column '{col}'")
        elif col_method == "median":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                raise TypeError(f"Cannot use 'median' on non-numeric column '{col}'")
        elif col_method == "mode":
            if df[col].mode().empty:
                raise ValueError(f"No mode found for column '{col}'")
            df[col] = df[col].fillna(df[col].mode().iloc[0])
        else:
            raise ValueError(
                f"Unsupported method '{col_method}'. Choose from: 'drop', 'knn', 'mean', 'median', 'mode'."
            )

    return df
