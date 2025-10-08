import pandas as pd
import numpy as np
from scipy.stats import skew
from typing import Dict, Tuple, Optional, Set, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

class NoventisOutlierHandler:
    """
    Intelligently handles outliers in DataFrame numeric columns using
    a class-based method consistent with Scikit-Learn.
    """

    def __init__(self,
                feature_method_map: Optional[Dict[str, str]] = None,
                default_method: str = 'auto',
                iqr_multiplier: float = 1.5,
                quantile_range: Tuple[float, float] = (0.05, 0.95),
                min_data_threshold: int = 100,
                skew_threshold: float = 0.5,
                verbose: bool = False):
        self.feature_method_map = feature_method_map or {}
        self.default_method = default_method or 'auto'
        self.iqr_multiplier = iqr_multiplier
        self.quantile_range = quantile_range
        self.min_data_threshold = min_data_threshold
        self.skew_threshold = skew_threshold
        self.verbose = verbose
        
        self.is_fitted_ = False
        self.boundaries_: Dict[str, Tuple[float, float]] = {}
        self.methods_: Dict[str, str] = {}
        self.train_indices_to_drop_: Set[int] = set() 
        
        self.quality_report_: Dict[str, Any] = {}
        self._plot_data_snapshot: Dict[str, pd.Series] = {}  

    def _choose_auto_method(self, col_data: pd.Series) -> str:
        """Helper function to choose automatic method."""
        clean_data = col_data.dropna()
        if len(clean_data) < self.min_data_threshold:
            return 'iqr_trim' 
        elif abs(skew(clean_data)) > self.skew_threshold:
            return 'winsorize'
        else:
            return 'quantile_trim' 

    def fit(self, X: pd.DataFrame, y=None) -> 'NoventisOutlierHandler':
        """Learn outlier boundaries from training data X."""
        df = X.copy()
        self.boundaries_ = {}
        self.methods_ = {}
        self.train_indices_to_drop_ = set()
        self._plot_data_snapshot = {}
        
        column_reports = {}

        for col in df.select_dtypes(include=np.number).columns:
            if df[col].nunique() <= 1: 
                continue               

            method = self.feature_method_map.get(col, self.default_method)
            if method == 'auto':
                method = self._choose_auto_method(df[col])

            self.methods_[col] = method

            self._plot_data_snapshot[col] = df[col].copy()

            if method == 'none':
                continue

            lower_bound, upper_bound = None, None
            outlier_count = 0
            
            if method in ['quantile_trim', 'winsorize']: 
                q_low, q_high = df[col].quantile(self.quantile_range)
                lower_bound, upper_bound = q_low, q_high
            elif method == 'iqr_trim': 
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.iqr_multiplier * IQR
                upper_bound = Q3 + self.iqr_multiplier * IQR

            self.boundaries_[col] = (lower_bound, upper_bound)

            if method in ['quantile_trim', 'iqr_trim']:
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_indices = df.index[outlier_mask]
                outlier_count = len(outlier_indices)
                self.train_indices_to_drop_.update(outlier_indices)
            elif method == 'winsorize':
                outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            # FIXED: Store per-column stats
            column_reports[col] = {
                'method': method,
                'outlier_count': outlier_count,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        self.is_fitted_ = True

        self.quality_report_ = {
            'rows_total': len(df),
            'rows_to_remove': len(self.train_indices_to_drop_),
            'removal_percentage': f"{(len(self.train_indices_to_drop_) / len(df) * 100):.2f}%" if len(df) > 0 else "0.00%",
            'column_reports': column_reports
        }

        if self.verbose:
            self._print_summary()
            
        return self

    def transform(self, X: pd.DataFrame, apply_train_removals: bool = False) -> pd.DataFrame:
        """
        Apply outlier handling to DataFrame.
        
        Args:
            X: DataFrame to transform
            apply_train_removals: If True, apply training set removal indices (for fit_transform only)
        """
        if not self.is_fitted_:
            raise RuntimeError("Handler must be fitted before transform.")
        
        df_out = X.copy()
        winsorized_count = 0
        removed_count = 0
        
        for col, method in self.methods_.items():
            if col not in df_out.columns: 
                continue                  
            
            if method == 'winsorize':
                lower_bound, upper_bound = self.boundaries_[col]
                winsorized_count += ((df_out[col] < lower_bound) | (df_out[col] > upper_bound)).sum()
                df_out[col] = np.clip(df_out[col], lower_bound, upper_bound)

        if apply_train_removals and self.train_indices_to_drop_:
            indices_in_df = self.train_indices_to_drop_.intersection(df_out.index) 
            df_out.drop(index=list(indices_in_df), inplace=True)
            removed_count = len(indices_in_df)
        else:
            # For new data, identify outliers based on learned boundaries
            indices_to_remove = set()
            for col, method in self.methods_.items():
                if col not in df_out.columns or method not in ['quantile_trim', 'iqr_trim']:
                    continue
                lower_bound, upper_bound = self.boundaries_[col]
                outlier_mask = (df_out[col] < lower_bound) | (df_out[col] > upper_bound)
                indices_to_remove.update(df_out.index[outlier_mask])
            
            if indices_to_remove:
                df_out.drop(index=list(indices_to_remove), inplace=True)
                removed_count = len(indices_to_remove)

        rows_before = len(X)
        rows_after = len(df_out)
        
        self.quality_report_.update({
            'rows_before': rows_before,
            'rows_after': rows_after,
            'outliers_removed': removed_count,
            'removal_percentage': f"{(removed_count / rows_before * 100):.2f}%" if rows_before > 0 else "0.00%",
            'outliers_winsorized': winsorized_count
        })
        
        return df_out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Perform fit and transform in one step."""
        # FIXED: Explicitly apply train removals for fit_transform
        return self.fit(X).transform(X, apply_train_removals=True)
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Return detailed quality report from outlier handling process."""
        if not self.is_fitted_:
            print("Handler has not been fitted.")
            return {}
        return self.quality_report_

    def _print_summary(self):
        """Print an easy-to-read summary to console."""
        report = self.quality_report_
        
        print("\n📋" + "="*23 + " OUTLIER HANDLING SUMMARY " + "="*23 + "📋")
        print(f"{'Method':<25} | {self.default_method.upper()}")
        print(f"{'Total Rows':<25} | {report.get('rows_total', 'N/A')}")
        print(f"{'Rows to Remove':<25} | {report.get('rows_to_remove', 'N/A')}")
        print(f"{'Removal Percentage':<25} | {report.get('removal_percentage', 'N/A')}")
        print("="*72)
        
        if 'column_reports' in report:
            print("\n📊 Per-Column Details:")
            for col, col_report in report['column_reports'].items():
                print(f"  • {col}: {col_report['method'].upper()} ({col_report['outlier_count']} outliers)")
    
    def get_summary_text(self) -> str:
        """Generates a formatted string summary for the HTML report."""
        if not self.is_fitted_: 
            return "<p>Outlier Handler has not been fitted.</p>"

        report = self.quality_report_
        methods_html = "".join([f"<li><b>{col}:</b> '{method.upper()}'</li>" 
                               for col, method in self.methods_.items()])

        summary_html = f"""
            <div class="grid-item">
                <h4>Outlier Summary</h4>
                <p><b>Rows Before:</b> {report.get('rows_before', report.get('rows_total', 0))}</p>
                <p><b>Rows After:</b> {report.get('rows_after', 'N/A')}</p>
                <p><b>Outlier Rows Removed:</b> {report.get('outliers_removed', report.get('rows_to_remove', 0))}</p>
                <p><b>Outlier Rows Winsorized:</b> {report.get('outliers_winsorized', 0)}</p>
            </div>
            <div class="grid-item">
                <h4>Methodology per Column</h4>
                <ul>{methods_html if methods_html else "<li>No columns handled.</li>"}</ul>
            </div>
        """
        return summary_html

    def plot_comparison(self, max_cols: int = 1):
        """Plot before/after comparison of outlier handling results."""
        if not self.is_fitted_ or not self._plot_data_snapshot: 
            return None
            
        cols_to_plot = [col for col, method in self.methods_.items() if method != 'none']
        if not cols_to_plot: 
            return None
            
        col_to_plot = cols_to_plot[0]

        original_series = self._plot_data_snapshot[col_to_plot]
        
        temp_df = pd.DataFrame({col_to_plot: original_series})
        for col in self._plot_data_snapshot:
            if col != col_to_plot:
                temp_df[col] = self._plot_data_snapshot[col]
        
        transformed_df = self.transform(temp_df.copy(), apply_train_removals=True)

        color_before, color_after = '#58A6FF', '#F78166'
        bg_color, text_color = '#0D1117', '#C9D1D9'

        fig = plt.figure(figsize=(16, 8), facecolor=bg_color)
        gs = fig.add_gridspec(2, 2, height_ratios=(3, 1), hspace=0.05)
        fig.suptitle(f"Outlier Handling Comparison for '{col_to_plot}' (Method: {self.methods_[col_to_plot].upper()})",
                    fontsize=20, color=text_color, weight='bold')

        ax_hist_before = fig.add_subplot(gs[0, 0])
        ax_box_before = fig.add_subplot(gs[1, 0], sharex=ax_hist_before)
        sns.histplot(data=original_series, kde=True, ax=ax_hist_before, color=color_before)
        sns.boxplot(data=original_series, ax=ax_box_before, color=color_before)
        ax_hist_before.set_title("Before", color=text_color, fontsize=14)
        plt.setp(ax_hist_before.get_xticklabels(), visible=False)

        ax_hist_after = fig.add_subplot(gs[0, 1])
        ax_box_after = fig.add_subplot(gs[1, 1], sharex=ax_hist_after)
        sns.histplot(data=transformed_df[col_to_plot], kde=True, ax=ax_hist_after, color=color_after)
        sns.boxplot(data=transformed_df[col_to_plot], ax=ax_box_after, color=color_after)
        ax_hist_after.set_title("After", color=text_color, fontsize=14)
        plt.setp(ax_hist_after.get_xticklabels(), visible=False)

        for ax in [ax_hist_before, ax_box_before, ax_hist_after, ax_box_after]:
            ax.set_facecolor(bg_color)
            ax.tick_params(colors=text_color, which='both')
            for spine in ax.spines.values(): 
                spine.set_edgecolor(text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.set_xlabel('')
            ax.set_ylabel('')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig