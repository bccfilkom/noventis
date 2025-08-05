import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder, OrdinalEncoder
from scipy.stats import chi2_contingency

class NoventisEncoder:
    """
    Class for encoding categorical columns using automatic or manual methods.
    """

    def __init__(self, method: str = 'auto', target_column: str = None, columns_to_encode: list = None, category_mapping: dict = None):
        """
        Initializes the NoventisEncoder.

        Args:
            method (str, optional): The encoding method. Defaults to 'auto'.
                'auto': Intelligently selects encoding based on cardinality and correlation with the target.
                'label', 'ohe', 'target', 'ordinal': For manual control.
            target_column (str, optional): The name of the target variable column. Required for 'auto' and 'target' modes.
            columns_to_encode (list, optional): A list of columns to encode. Required for manual modes.
            category_mapping (dict, optional): A mapping dictionary for ordinal encoding.
        """
        self.method = method
        self.target_column = target_column
        self.columns_to_encode = columns_to_encode
        self.category_mapping = category_mapping
        self.encoders = {}
        self.learned_cols = {}

    def _cramers_v(self, x, y):
        """
        Calculates Cramér's V statistic for categorical-categorical association.
        """
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        if n == 0:
            return 0
        r, k = confusion_matrix.shape
        phi2 = chi2 / n
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        if min((kcorr-1), (rcorr-1)) == 0:
            return 0
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fit the encoder to the data. Learns the encoding rules.

        Args:
            X (pd.DataFrame): The input DataFrame.
            y (pd.Series, optional): The target variable. Required for 'auto' and 'target' methods.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input 'X' must be a pandas DataFrame.")

        if self.method == 'auto':
            if y is None:
                raise ValueError("Parameter 'y' is required for 'auto' mode.")
            
            y_binned = pd.qcut(y, q=4, duplicates='drop', labels=False) if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10 else y

            cols_to_process = X.select_dtypes(include=['object', 'category']).columns.tolist()

            for col in cols_to_process:
                unique_count = X[col].nunique()
                if unique_count == 2:
                    self.learned_cols[col] = 'label'
                elif unique_count > 15:
                    self.learned_cols[col] = 'target'
                elif 3 <= unique_count <= 15:
                    correlation = self._cramers_v(X[col], y_binned)
                    if correlation < 0.25:
                        self.learned_cols[col] = 'ohe'
                    else:
                        print(f"RECOMMENDATION for '{col}': Manually apply 'ordinal' or 'target' encoding.")
                        self.learned_cols[col] = 'skip'
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the learned encoding rules.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with encoded columns.
        """
        df = X.copy()
        
        columns_to_process = self.columns_to_encode if self.columns_to_encode else df.columns

        if self.method == 'auto':
             for col, method in self.learned_cols.items():
                if method == 'label':
                    le = LabelEncoder()
                    df[f'{col}_label'] = le.fit_transform(df[col])
                    df = df.drop(col, axis=1)
                elif method == 'ohe':
                    df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=False)
                elif method == 'target':
                    te = TargetEncoder()
                    df[f'{col}_target'] = te.fit_transform(df[col], df[self.target_column])
                    df = df.drop(col, axis=1)
        else: # Manual methods
            for col in columns_to_process:
                if col not in df.columns or not pd.api.types.is_categorical_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
                    continue

                if self.method == 'label':
                    if df[col].nunique() > 2:
                        print(f"Warning: Column '{col}' has more than 2 unique values. Label encoding may not be ideal.")
                    le = LabelEncoder()
                    df[f'{col}_label_encoded'] = le.fit_transform(df[col])
                
                elif self.method == 'ohe':
                    if df[col].nunique() > 15:
                         print(f"Warning: Column '{col}' has {df[col].nunique()} unique values, OHE will create many new features.")
                    df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=False)

                elif self.method == 'target':
                     if self.target_column is None:
                         raise ValueError("'target_column' must be specified for target encoding.")
                     te = TargetEncoder()
                     df[f'{col}_target_encoded'] = te.fit_transform(df[col], df[self.target_column])

                elif self.method == 'ordinal':
                    if self.category_mapping is None or col not in self.category_mapping:
                        raise ValueError(f"Mapping for column '{col}' not found in 'category_mapping'.")
                    
                    encoder_mapping = [{'col': col, 'mapping': self.category_mapping[col]}]
                    oe = OrdinalEncoder(mapping=encoder_mapping)
                    df[col] = oe.fit_transform(df)[col]
                    df.rename(columns={col: f'{col}_ordinal_encoded'}, inplace=True)

        return df

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Args:
            X (pd.DataFrame): The input DataFrame.
            y (pd.Series, optional): The target variable.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        self.fit(X, y)
        return self.transform(X)