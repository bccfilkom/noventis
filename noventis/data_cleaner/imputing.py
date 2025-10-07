import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns

class NoventisImputer:
    """
    Intelligently handles missing values (null/NaN) using a class-based approach.
    
    Features:
    - Compatible with Scikit-learn API (fit, transform).
    - Automatic detection of column types (numeric & categorical).
    - Supports methods: 'mean', 'median', 'mode', 'knn', 'constant', 'ffill', 'bfill', 'drop'.
    - Flexible: accepts global method (string) or per-column method (dictionary).
    - Special handling for integer columns (automatic rounding).
    """

    def __init__(self, 
                method: Optional[str] = None, 
                columns: Optional[List[str]] = None,
                fill_value: Any = None,
                n_neighbors: int = 5, 
                verbose: bool = False):
        """
        Initialize NoventisImputer.

        Args:
            method (str, dict, or None): 
                - str: 'mean', 'median', 'mode', 'knn', 'constant', 'ffill', 'bfill', 'drop' for all columns.
                - dict: Specific method per column, e.g.: {'Age': 'median'}.
                - None: Auto mode ('mean' for numeric, 'mode' for categorical).
            columns (list, optional): List of specific columns to process. 
                                    If None, process all relevant columns.
            fill_value (any, optional): Value to use when method='constant'.
            n_neighbors (int): Number of neighbors for 'knn' method.
        """
        self.method = method
        self.columns = columns
        self.fill_value = fill_value
        self.n_neighbors = n_neighbors
        self._column_methods: Dict[str, str] = {}
        self.verbose = verbose
        
        self.is_fitted_ = False
        self.imputers_: Dict[str, Any] = {}
        self.columns_to_process_: List[str] = []
        self.quality_report_: Dict[str, Any] = {}
        self._original_df_snapshot: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y=None) -> 'NoventisImputer':
        """
        Learn imputation strategy from training data X.
        """
        df = X.copy()
        self._original_df_snapshot = df
        self.imputers_ = {}
        
        # Determine columns to process
        cols_with_null = [col for col in df.columns if df[col].isnull().sum() > 0]
        if self.columns:
            self.columns_to_process_ = [col for col in self.columns if col in cols_with_null]
        else:
            self.columns_to_process_ = cols_with_null

        # Special and efficient handling for KNN
        if self.method == 'knn':
            num_cols = [c for c in df.select_dtypes(include=np.number).columns if c in self.columns_to_process_]
            cat_cols = [c for c in df.select_dtypes(include='object').columns if c in self.columns_to_process_]
            
            if num_cols:
                knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
                knn_imputer.fit(df[num_cols])
                self.imputers_['knn'] = {'imputer': knn_imputer, 'cols': num_cols}

            # For categorical columns in KNN mode, use mode as fallback
            for col in cat_cols:
                imp = SimpleImputer(strategy='most_frequent')
                self.imputers_[col] = imp.fit(df[[col]])
        else:
            # Handling for other methods (per column)
            for col in self.columns_to_process_:
                chosen_method = self.method
                if isinstance(self.method, dict):
                    chosen_method = self.method.get(col, None)
                
                # Automatic Fallback
                if chosen_method is None:
                    chosen_method = 'mean' if pd.api.types.is_numeric_dtype(df[col]) else 'mode'
                
                # Methods that don't need fitting
                if chosen_method in ['drop', 'ffill', 'bfill']:
                    self.imputers_[col] = chosen_method
                    continue

                strategy = ''
                fill_val = None
                is_numeric = pd.api.types.is_numeric_dtype(df[col])

                if chosen_method == 'mean' and is_numeric:
                    # Check if column contains integers for rounding
                    if np.all(df[col].dropna() % 1 == 0):
                        strategy = 'constant'
                        fill_val = round(df[col].mean())
                    else:
                        strategy = 'mean'
                elif chosen_method == 'median' and is_numeric:
                    strategy = 'median'
                elif chosen_method == 'mode':
                    strategy = 'most_frequent'
                elif chosen_method == 'constant':
                    strategy = 'constant'
                    fill_val = self.fill_value # Use fill_value from user
                else: # Fallback for categorical
                    strategy = 'most_frequent'
                
                imp = SimpleImputer(strategy=strategy, fill_value=fill_val)
                self.imputers_[col] = imp.fit(df[[col]])
        
        self.is_fitted_ = True

        cols_with_null = [col for col in df.columns if df[col].isnull().sum() > 0]
        if self.columns:
            cols_to_process = [col for col in self.columns if col in cols_with_null]
        else:
            cols_to_process = cols_with_null
            
        # Store columns to be processed for later use
        self.columns_to_process_ = cols_to_process

        if not self.columns_to_process_:
            missing_before = 0
        else:
            missing_before = df[self.columns_to_process_].isnull().sum().sum()

        self.quality_report_ = {
            'missing_values_before': int(missing_before)
        }

        if self.verbose:
            self._print_summary()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply missing value imputation to DataFrame.
        """
        if not self.is_fitted_:
            raise RuntimeError("Imputer must be fitted before transform.")
            
        df_out = X.copy()
        
        # Handle KNN
        if 'knn' in self.imputers_:
            knn_info = self.imputers_['knn']
            # Ensure columns exist in transform dataframe
            cols_to_transform = [c for c in knn_info['cols'] if c in df_out.columns]
            if cols_to_transform:
                df_out[cols_to_transform] = knn_info['imputer'].transform(df_out[cols_to_transform])
        
        # Handle other methods
        for col, imputer in self.imputers_.items():
            if col == 'knn' or col not in df_out.columns:
                continue
            
            if imputer == 'drop':
                df_out.dropna(subset=[col], inplace=True)
            elif imputer in ['ffill', 'bfill']:
                df_out[col].fillna(method=imputer, inplace=True)
            else: # SimpleImputer
                df_out[[col]] = imputer.transform(df_out[[col]])
        
        if not self.columns_to_process_:
            missing_after = 0
        else:
            # Ensure we only count in columns that exist in df_out
            processed_cols_in_df = [col for col in self.columns_to_process_ if col in df_out.columns]
            missing_after = df_out[processed_cols_in_df].isnull().sum().sum()

        missing_before = self.quality_report_.get('missing_values_before', 0)
        processed_cols_in_df = [col for col in self.columns_to_process_ if col in df_out.columns]
        missing_after = df_out[processed_cols_in_df].isnull().sum().sum() if processed_cols_in_df else 0
        values_imputed = missing_before - missing_after
        completion_score = (1 - (missing_after / self._original_df_snapshot.size)) * 100 if self._original_df_snapshot.size > 0 else 100

        self.quality_report_.update({
            'missing_values_after': int(missing_after),
            'values_imputed': int(values_imputed),
            'completion_score': f"{completion_score:.2f}%"
        })

        return df_out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Perform fit and transform in one step.
        """
        return self.fit(X, y).transform(X)
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Calculate and return detailed quality report from imputation process."""
        if not self.is_fitted_ or self._original_df_snapshot is None:
            print("Imputer has not been fitted.")
            return {}

        df_before = self._original_df_snapshot
        df_after = self.transform(df_before.copy())

        missing_before = df_before[self.columns_to_process_].isnull().sum().sum() if self.columns_to_process_ else 0
        processed_cols = [c for c in self.columns_to_process_ if c in df_after.columns]
        missing_after = df_after[processed_cols].isnull().sum().sum() if processed_cols else 0
        
        values_imputed = missing_before - missing_after
        score = (values_imputed / missing_before * 100) if missing_before > 0 else 100.0

        report = {'overall_summary': {}, 'column_details': {}}
        report['overall_summary'] = {
            'total_missing_before': int(missing_before),
            'total_missing_after': int(missing_after),
            'total_values_imputed': int(values_imputed),
            'completion_score': f"{score:.2f}%"
        }
        for col in self.columns_to_process_:
            report['column_details'][col] = {
                'missing_before': int(df_before[col].isnull().sum()),
                'method': self._column_methods.get(col, 'N/A')
            }
        
        self.quality_report_ = report
        return self.quality_report_

    def _print_summary(self):
        """Print an easy-to-read summary to console."""
        report = self.get_quality_report()
        summary = report.get('overall_summary', {})
        
        print("\n📋" + "="*23 + " IMPUTATION SUMMARY " + "="*23 + "📋")
        method_str = self.method if isinstance(self.method, str) else 'CUSTOM MAP'
        print(f"{'Method':<25} | {method_str if self.method is not None else 'AUTO'}")
        print(f"{'Total Values Imputed':<25} | {summary.get('total_values_imputed', 'N/A')}")
        print(f"{'Completion Score':<25} | {summary.get('completion_score', 'N/A')}")
        print("="*68)

    def get_summary_text(self) -> str:
        """Generates a formatted string summary for the HTML report."""
        if not self.is_fitted_: return "<p>Imputer has not been fitted.</p>"

        report = self.quality_report_.get('overall_summary', {})
        
        strategy_text = 'Auto'
        if isinstance(self.method, str):
            strategy_text = self.method.upper()
        elif isinstance(self.method, dict):
            strategy_text = "Custom per Column"

        summary_html = f"""
            <h4>Imputation Summary</h4>
            <p><b>Total Values Imputed:</b> {report.get('total_values_imputed', 0)}</p>
            <p><b>Completion Score:</b> {report.get('completion_score', 'N/A')}</p>
            <p><b>Columns Processed:</b> {len(self.columns_to_process_)}</p>
            <h4>Methodology</h4>
            <p><b>Strategy:</b> {strategy_text}</p>
            <p>In 'auto' mode, 'mean' is used for numeric columns and 'mode' for categorical columns.</p>
        """
        return summary_html
    
    def plot_comparison(self, max_cols: int = 1):
        """Plot before/after comparison of imputation results."""
        if not self.is_fitted_ or self._original_df_snapshot is None: return None

        original_data = self._original_df_snapshot
        transformed_data = self.transform(original_data.copy())

        # Find the first numeric column that was actually processed
        numeric_cols_processed = [
            col for col in self.columns_to_process_
            if pd.api.types.is_numeric_dtype(original_data[col])
        ]
        if not numeric_cols_processed: return None
        col_to_plot = numeric_cols_processed[0]

        # Define colors from the new palette
        color_before = '#58A6FF' # Primary Blue
        color_after = '#F78166' # Primary Orange
        bg_color = '#0D1117' # BG Dark 1
        text_color = '#C9D1D9' # Text Light

        fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=bg_color,
                                 gridspec_kw={'height_ratios': [1, 2]})
        fig.suptitle(f"Imputation Comparison for '{col_to_plot}'", fontsize=20, color=text_color, weight='bold')

        sns.heatmap(original_data.isnull(), cbar=False, ax=axes[0, 0], cmap='viridis')
        axes[0, 0].set_title("Before: Location of Missing Data (Yellow)", color=text_color, fontsize=14)
        sns.histplot(original_data[col_to_plot].dropna(), kde=True, ax=axes[1, 0], color=color_before)
        axes[1, 0].set_title(f"Before: Distribution of '{col_to_plot}'", color=text_color, fontsize=14)

        sns.heatmap(transformed_data.isnull(), cbar=False, ax=axes[0, 1], cmap='viridis')
        axes[0, 1].set_title("After: Missing Data Handled", color=text_color, fontsize=14)
        sns.histplot(transformed_data[col_to_plot], kde=True, ax=axes[1, 1], color=color_after)
        axes[1, 1].set_title(f"After: Distribution of '{col_to_plot}'", color=text_color, fontsize=14)

        # Apply styling to all axes
        for ax_row in axes:
            for ax in ax_row:
                ax.set_facecolor(bg_color)
                ax.tick_params(colors=text_color, which='both')
                for spine in ax.spines.values():
                    spine.set_edgecolor(text_color)
                ax.xaxis.label.set_color(text_color)
                ax.yaxis.label.set_color(text_color)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig