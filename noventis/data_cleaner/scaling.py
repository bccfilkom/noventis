from typing import Any, Dict, Literal, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
)

import matplotlib.pyplot as plt
import seaborn as sns

# Default parameters for each scaling method
LIST_SCALER = {
    "standard": {"with_mean": True, "with_std": True},
    "minmax": {"feature_range": (0, 1)},
    "robust": {"with_centering": True, "with_scaling": True, "quantile_range": (25.0, 75.0)},
    "power": {"standardize": True, "method": 'yeo-johnson'},
}

class NoventisScaler:
    """
    An advanced scaler that analyzes each numerical column, selects the optimal scaling method,
    and applies the scaling per column without overwriting other columns' scalers.

    Supports both automatic selection (`method='auto'`) and manual forcing of a scaling method.
    Stores analysis results, scaling method choices, and reasons for later inspection.
    """

    def __init__(self, 
                 method: Optional[Literal['auto', 'standard', 'minmax', 'robust', 'power']] = 'auto',
                 optimize: bool = True,
                 custom_params: Optional[Dict] = None,
                 skew_threshold: float = 2.0,
                 outlier_threshold: float = 0.01,
                 normality_alpha: float = 0.05,
                 verbose: bool = False):
        """
        Initializes the NoventisScaler.

        Args:
            method (str, optional): Scaling method. Can be:
                - 'auto' (default): automatically choose per column based on data characteristics.
                - 'standard', 'minmax', 'robust', 'power': force all columns to use the same method.
            optimize (bool, optional): If True, optimizes parameters for the selected scaler. Defaults to True.
            custom_params (dict, optional): Custom parameters to override the optimized/default parameters.
            skew_threshold (float, optional): Threshold of absolute skewness to consider a column as highly skewed. Defaults to 2.0.
            outlier_threshold (float, optional): Proportion threshold to consider a column as having outliers. Defaults to 0.01.
            normality_alpha (float, optional): Alpha value for normality testing. Defaults to 0.05.
            verbose (bool, optional): If True, prints summary after fitting. Defaults to True.
        """
        allowed_methods = ['auto', 'standard', 'minmax', 'robust', 'power']
        if method not in allowed_methods:
            raise ValueError(f"Invalid method. Allowed methods are: {allowed_methods}")

        self.method = method
        self.optimize = optimize
        self.custom_params = custom_params or {}
        self.skew_threshold = skew_threshold
        self.outlier_threshold = outlier_threshold
        self.normality_alpha = normality_alpha
        self.verbose = verbose

        self.scalers_ = {}
        self.analysis_ = {}
        self.fitted_methods_ = {}
        self.reasons_ = {}
        self.is_fitted_ = False

        self.quality_report_ = {}
        self._original_df_snapshot = None

    def _analyze_column(self, data: pd.Series) -> Dict[str, Any]:
        """Analyze a single column's distribution, skewness, outliers, and normality."""
        clean_data = data.dropna()
        if len(clean_data) < 5:
            return {'error': 'Not enough data to analyze.'}

        analysis = {
            'n_samples': len(clean_data),
            'mean': np.mean(clean_data),
            'median': np.median(clean_data),
            'std': np.std(clean_data),
            'min': np.min(clean_data),
            'max': np.max(clean_data),
            'range': np.max(clean_data) - np.min(clean_data),
            'skewness': abs(stats.skew(clean_data)),
            'is_bounded_01': np.all((clean_data >= 0) & (clean_data <= 1)),
            'is_positive_only': np.all(clean_data > 0)
        }

        # Outlier detection (IQR method)
        q1, q3 = np.percentile(clean_data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((clean_data < lower_bound) | (clean_data > upper_bound)).sum()
        analysis['outlier_ratio'] = outliers / len(clean_data)
        analysis['has_outliers'] = analysis['outlier_ratio'] > self.outlier_threshold

        # Normality test (Shapiro for <=5000, Anderson otherwise)
        try:
            if len(clean_data) <= 5000:
                _, p_value = stats.shapiro(clean_data)
                analysis['is_normal'] = p_value > self.normality_alpha
                analysis['normality_p_value'] = p_value
                if analysis['is_normal']:
                    analysis['normality_reason'] = f"p-value {p_value:.3f} > {self.normality_alpha}"
            else:
                statistic, critical_values, _ = stats.anderson(clean_data, dist='norm')
                analysis['is_normal'] = statistic < critical_values[2]
                analysis['normality_statistic'] = statistic
                if analysis['is_normal']:
                    analysis['normality_reason'] = f"stat {statistic:.3f} < crit {critical_values[2]:.3f}"
        except Exception:
            analysis['is_normal'] = False

        return analysis

    def _select_optimal_method(self, analysis: Dict[str, Any], is_for_knn: bool) -> Tuple[str, str]:
        """Select the most suitable scaling method based on column analysis."""
        if is_for_knn:
            return 'minmax', "Forced by user for KNN"
        if analysis.get('error'):
            return 'standard', analysis['error']
        if analysis.get('skewness', 0) > self.skew_threshold:
            return 'power', f"High skewness ({analysis['skewness']:.2f})"
        if analysis.get('has_outliers', False):
            return 'robust', f"Outliers (ratio: {analysis['outlier_ratio']:.1%})"
        if analysis.get('is_normal', False):
            return 'standard', analysis.get('normality_reason', "Data appears normal")
        return 'standard', "Default fallback"

    def _optimize_parameters(self, method: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize parameters for the chosen scaling method."""
        params = LIST_SCALER[method].copy()
        if method == 'power':
            params['method'] = 'box-cox' if analysis.get('is_positive_only', False) else 'yeo-johnson'
        elif method == 'robust':
            outlier_ratio = analysis.get('outlier_ratio', 0)
            if outlier_ratio > 0.05:
                lower_q = max(10.0, (outlier_ratio / 2) * 100)
                upper_q = min(90.0, 100 - lower_q)
                params['quantile_range'] = (lower_q, upper_q)
        elif method == 'minmax':
            if analysis.get('is_bounded_01', False):
                params['feature_range'] = (-1, 1)
        return params

    def fit(self, X: pd.DataFrame, is_for_knn: bool = False) -> 'NoventisScaler':
        """
        Analyze each numeric column, choose scaling method, fit scaler for each column.
        
        Args:
            X (pd.DataFrame): Input dataframe.
            is_for_knn (bool, optional): If True, force 'minmax' scaling for all columns. Defaults to False.
        """

        self._original_df_snapshot = X.copy() 
        numeric_cols = X.select_dtypes(include=np.number).columns
        if numeric_cols.empty:
            raise ValueError("No numerical features to scale.")

        for col in numeric_cols:
            analysis = self._analyze_column(X[col])
            self.analysis_[col] = analysis

            if self.method == 'auto':
                selected_method, reason = self._select_optimal_method(analysis, is_for_knn)
            else:
                selected_method, reason = self.method, "Forced by user"

            self.fitted_methods_[col] = selected_method
            self.reasons_[col] = reason

            params = self._optimize_parameters(selected_method, analysis) if self.optimize else LIST_SCALER[selected_method].copy()
            params.update(self.custom_params)

            scaler_map = {
                'standard': StandardScaler,
                'minmax': MinMaxScaler,
                'robust': RobustScaler,
                'power': PowerTransformer
            }
            self.scalers_[col] = scaler_map[selected_method](**params).fit(X[[col]])

        self.is_fitted_ = True
        if self.verbose:
            self._print_summary()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scalers to transform dataframe."""
        if not self.is_fitted_:
            raise RuntimeError("Fit the scaler before transform.")
        df = X.copy()
        for col, scaler in self.scalers_.items():
            if col in df.columns:
                df[[col]] = scaler.transform(df[[col]])
        return df

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse transformation to original scale."""
        if not self.is_fitted_:
            raise RuntimeError("Fit the scaler before inverse_transform.")
        df = X.copy()
        for col, scaler in self.scalers_.items():
            if col in df.columns:
                df[[col]] = scaler.inverse_transform(df[[col]])
        return df

    def fit_transform(self, X: pd.DataFrame, is_for_knn: bool = False) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, is_for_knn=is_for_knn).transform(X)

    def _print_summary(self):
        """Print summary of scaling methods used and reasons."""
    
        method_counts = pd.Series(self.fitted_methods_).value_counts()
        for method, count in method_counts.items():
            print(f"   - {method.upper()}: {count} columns")

        for col in self.fitted_methods_:
            print(f"  Column: {col}")
            print(f"     - Method: {self.fitted_methods_[col].upper()}")
            print(f"     - Reason: {self.reasons_[col]}")
            print(f"     - Skewness: {self.analysis_[col].get('skewness', 0):.2f} | Outlier Ratio: {self.analysis_[col].get('outlier_ratio', 0):.2%}")
        print("=" * 60)

    def get_quality_report(self) -> Dict[str, Any]:
        """
        Generate quality report about statistical changes after scaling.
        """
        if not self.is_fitted_ or self._original_df_snapshot is None:
            print("Scaler has not been fitted. Run .fit() or .fit_transform() first.")
            return {}
        
        transformed_df = self.transform(self._original_df_snapshot.copy())
        
        report = {'column_details': {}}
        for col, analysis_before in self.analysis_.items():
            if col in transformed_df.columns:
                analysis_after = self._analyze_column(transformed_df[col])
                report['column_details'][col] = {
                    'method': self.fitted_methods_.get(col, 'N/A').upper(),
                    'skewness_before': f"{analysis_before.get('skewness', 0):.3f}",
                    'skewness_after': f"{analysis_after.get('skewness', 0):.3f}",
                    'mean_before': f"{analysis_before.get('mean', 0):.3f}",
                    'mean_after': f"{analysis_after.get('mean', 0):.3f}",
                    'std_dev_before': f"{analysis_before.get('std', 0):.3f}",
                    'std_dev_after': f"{analysis_after.get('std', 0):.3f}",
                }
        self.quality_report_ = report
        return self.quality_report_


    def plot_comparison(self, max_cols: int = None):
        """Plot before/after comparison of scaling results."""
        if max_cols is None:
            max_cols = len(self.scalers_.keys)
        if not self.is_fitted_ or self._original_df_snapshot is None: return None
        cols_to_plot = list(self.scalers_.keys())[:max_cols]
        if not cols_to_plot: return None

        original_data = self._original_df_snapshot
        transformed_data = self.transform(original_data.copy())
        col = [cols_to_plot[i] for i in range(len(cols_to_plot))]

        color_before, color_after = '#58A6FF', '#F78166'
        bg_color, text_color = '#0D1117', '#C9D1D9'

        for i in range(len(col)):
            fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=bg_color)
            fig.suptitle(f"Scaling Comparison for '{col[i]}' (Method: {self.fitted_methods_[col[i]].upper()})",
                        fontsize=20, color=text_color, weight='bold')

            # --- BEFORE ---
            sns.histplot(original_data[col[i]], kde=True, ax=axes[0, 0], color=color_before)
            axes[0, 0].set_title("Before: Distribution", color=text_color, fontsize=14)
            stats.probplot(original_data[col[i]].dropna(), dist="norm", plot=axes[0, 1])
            axes[0, 1].get_lines()[0].set(color=color_before, markerfacecolor=color_before)
            axes[0, 1].get_lines()[1].set(color='white', linestyle='--')
            axes[0, 1].set_title("Before: Q-Q Plot vs Normal", color=text_color, fontsize=14)

            # --- AFTER ---
            sns.histplot(transformed_data[col[i]], kde=True, ax=axes[1, 0], color=color_after)
            axes[1, 0].set_title("After: Distribution", color=text_color, fontsize=14)
            stats.probplot(transformed_data[col[i]].dropna(), dist="norm", plot=axes[1, 1])
            axes[1, 1].get_lines()[0].set(color=color_after, markerfacecolor=color_after)
            axes[1, 1].get_lines()[1].set(color='white', linestyle='--')
            axes[1, 1].set_title("After: Q-Q Plot vs Normal", color=text_color, fontsize=14)

            for ax_row in axes:
                for ax in ax_row:
                    ax.set_facecolor(bg_color)
                    ax.tick_params(colors=text_color, which='both')
                    for spine in ax.spines.values(): spine.set_edgecolor(text_color)
                    ax.title.set_color(text_color)
                    ax.xaxis.label.set_color(text_color)
                    ax.yaxis.label.set_color(text_color)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
        # return fig

    def get_summary_text(self) -> str:
        """Generates a formatted string summary for the HTML report."""
        if not self.is_fitted_: return "<p>Scaler has not been fitted.</p>"

        reasons_html = "".join([f"<li><b>{col}:</b> Used '{self.fitted_methods_.get(col, 'N/A').upper()}' because: {reason}</li>" for col, reason in self.reasons_.items()])

        summary_html = f"""
            <div class="grid-item">
                <h4>Scaling Summary</h4>
                <p><b>Strategy:</b> {self.method.upper()}</p>
                <p><b>Numeric Columns Scaled:</b> {len(self.scalers_)}</p>
            </div>
            <div class="grid-item">
                <h4>Automatic Decision Reasons</h4>
                <ul>{reasons_html if reasons_html else "<li>No automatic decisions made or reported.</li>"}</ul>
            </div>
        """
        return summary_html