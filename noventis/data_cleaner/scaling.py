from typing import Any, Dict, Literal, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
)

# Default parameter setiap metode scaling
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
                 verbose: bool = True):
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
        print("\n📋 SCALING SUMMARY\n" + "-" * 40)
        method_counts = pd.Series(self.fitted_methods_).value_counts()
        for method, count in method_counts.items():
            print(f"   - {method.upper()}: {count} columns")

        print("\n📊 DETAILED REPORT\n" + "-" * 40)
        for col in self.fitted_methods_:
            print(f"  Column: {col}")
            print(f"     - Method: {self.fitted_methods_[col].upper()}")
            print(f"     - Reason: {self.reasons_[col]}")
            print(f"     - Skewness: {self.analysis_[col].get('skewness', 0):.2f} | Outlier Ratio: {self.analysis_[col].get('outlier_ratio', 0):.2%}")
        print("=" * 60)

    def get_quality_report(self) -> Dict[str, Any]:
        """
        Menghasilkan laporan kualitas tentang perubahan statistik setelah scaling.
        """
        if not self.is_fitted_ or self._original_df_snapshot is None:
            print("Scaler belum di-fit. Jalankan .fit() atau .fit_transform() terlebih dahulu.")
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
        
    def plot_comparison(self, max_cols: int = 5):
        """
        Membuat visualisasi perbandingan distribusi data sebelum dan sesudah scaling.
        """
        if not self.is_fitted_ or self._original_df_snapshot is None:
            print("Scaler belum di-fit. Jalankan .fit() atau .fit_transform() terlebih dahulu.")
            return

        print("Membuat visualisasi perbandingan untuk scaling...")
        
        original_data = self._original_df_snapshot
        transformed_data = self.transform(original_data.copy())
        
        cols_to_plot = list(self.scalers_.keys())[:max_cols]

        for col in cols_to_plot:
            if col not in transformed_data.columns:
                continue
                
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Perbandingan Distribusi untuk Kolom '{col}' (Metode: {self.fitted_methods_[col].upper()})", fontsize=16)

            # Plot Sebelum
            sns.histplot(original_data[col], kde=True, ax=axes[0], color='skyblue')
            axes[0].set_title(f"Sebelum Scaling")
            axes[0].axvline(original_data[col].mean(), color='r', linestyle='--', label=f"Mean: {original_data[col].mean():.2f}")
            axes[0].legend()

            # Plot Sesudah
            sns.histplot(transformed_data[col], kde=True, ax=axes[1], color='lightgreen')
            axes[1].set_title(f"Sesudah Scaling")
            axes[1].axvline(transformed_data[col].mean(), color='r', linestyle='--', label=f"Mean: {transformed_data[col].mean():.2f}")
            axes[1].legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    def print_quality_report(self):
        """
        Analyzes data before and after scaling, then prints
        a side-by-side data quality comparison report.
        """
        if not self.is_fitted_:
            print("⚠️ Scaler must be fitted first. Run .fit() or .fit_transform().")
            return
        
        if self._original_df_snapshot is None:
            print("⚠️ Original data snapshot not found. Run .fit() on your data.")
            return

        print("📊" + "="*23 + " SCALING QUALITY REPORT " + "="*23 + "📊")
        
        # Get the before and after dataframes
        df_before = self._original_df_snapshot
        df_after = self.transform(df_before.copy())
        
        # Analyze both versions
        report_before = assess_data_quality(df_before)
        report_after = assess_data_quality(df_after)
        
        # Print the comparison
        order = [
            'completeness', 'datatype_purity', 'outlier_quality', 
            'distribution_quality'
        ]

        print(f"{'METRIC':<25} | {'BEFORE':<12} | {'AFTER':<12}")
        print("-" * 55)

        for key in order:
            if key in report_before and key in report_after:
                title = key.replace('_', ' ').title()
                score_before = report_before[key]['score']
                score_after = report_after[key]['score']
                
                # Add an indicator for improvement
                indicator = "✅" if score_after > score_before else "  "
                
                print(f"{title:<25} | {score_before:<12} | {score_after:<12} {indicator}")
        
        print("="*55)
