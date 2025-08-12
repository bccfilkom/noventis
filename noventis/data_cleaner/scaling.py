from typing import Any, Dict, Literal, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
)

class NoventisScaler:
    """
    An advanced scaler that automatically selects and optimizes the best scaling method 
    for each numerical column in a DataFrame.
    """
    def __init__(self, 
                 method: Literal['auto', 'standard', 'minmax', 'robust', 'power'] = 'auto', 
                 optimize: bool = True, 
                 verbose: bool = True,
                 skew_threshold: float = 2.0): # DIKEMBALIKAN: Threshold sebagai parameter
        """
        Initializes the NoventisScaler.

        Args:
            method (str): Scaling method. Can be 'auto' or a specific method. Defaults to 'auto'.
            optimize (bool): If True, optimizes scaler parameters. Defaults to True.
            verbose (bool): If True, prints a summary after fitting. Defaults to True.
            skew_threshold (float): Threshold to consider data as skewed. Defaults to 2.0.
        """
        allowed_methods = ['auto', 'standard', 'minmax', 'robust', 'power']
        if method not in allowed_methods:
            raise ValueError(f"Invalid method. Allowed methods are: {allowed_methods}")

        self.method = method
        self.optimize = optimize
        self.verbose = verbose
        self.skew_threshold = skew_threshold # Menyimpan threshold
        
        self.scalers_ = {}
        self.fitted_methods_ = {}
        self.analysis_ = {}
        self.reasons_ = {}
        self.is_fitted_ = False

    def _analyze_column(self, data: pd.Series) -> Dict[str, Any]:
        """Analyzes a single numerical column and returns its statistical properties."""
        clean_data = data.dropna()
        if len(clean_data) < 5:
            return {'error': 'Not enough data to analyze.'}

        analysis = {'skewness': abs(stats.skew(clean_data))}
        
        q1, q3 = np.percentile(clean_data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((clean_data < lower_bound) | (clean_data > upper_bound)).sum()
        analysis['outlier_ratio'] = outliers / len(clean_data) if len(clean_data) > 0 else 0
        analysis['has_outliers'] = analysis['outlier_ratio'] > 0.01

        analysis['is_bounded_01'] = np.all((clean_data >= 0) & (clean_data <= 1))
        analysis['is_positive_only'] = np.all(clean_data > 0)
        
        normality_alpha = 0.05
        try:
            if len(clean_data) <= 5000:
                _, p_value = stats.shapiro(clean_data)
                analysis['is_normal'] = p_value > normality_alpha
                analysis['normality_p_value'] = p_value
                if analysis['is_normal']:
                    analysis['normality_reason'] = f"p-value {p_value:.3f} > {normality_alpha}"
            else:
                statistic, critical_values, _ = stats.anderson(clean_data, dist='norm')
                analysis['is_normal'] = statistic < critical_values[2]  # significance level 5%
                analysis['normality_statistic'] = statistic
                if analysis['is_normal']:
                    analysis['normality_reason'] = f"statistic {statistic:.3f} < critical value {critical_values[2]:.3f}"
        except Exception:
            analysis['is_normal'] = False
            analysis['normality_p_value'] = 0
            analysis['normality_statistic'] = float('inf')

        return analysis

    def _select_optimal_method(self, analysis: Dict[str, Any], is_for_knn: bool) -> Tuple[str, str]:
        """Selects the best scaling method based on data analysis."""
        # DIKEMBALIKAN: Logika prioritas untuk KNN
        if is_for_knn:
            return 'minmax', "Forced by user for KNN compatibility"
        if analysis.get('error'):
            return 'standard', analysis['error']
        # Menggunakan threshold dari parameter __init__
        if analysis.get('skewness', 0) > self.skew_threshold:
            return 'power', f"High skewness ({analysis['skewness']:.2f}) detected"
        if analysis.get('has_outliers', False):
            return 'robust', f"Outliers detected (ratio: {analysis['outlier_ratio']:.1%})"
        if analysis.get('is_normal', False):
            return 'standard', analysis.get('normality_reason', "Data appears normal")
        
        return 'standard', "Default fallback"

    def _optimize_parameters(self, method: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimizes parameters for the selected scaling method."""
        params = {}
        if method == 'power':
            params['method'] = 'box-cox' if analysis.get('is_positive_only', False) else 'yeo-johnson'
            params['standardize'] = True
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

    def fit(self, X: pd.DataFrame, is_for_knn: bool = False) -> 'NoventisScaler': # DIKEMBALIKAN: parameter is_for_knn
        """
        Analyzes each numerical column, selects the optimal scaling method,
        and fits the scaler.
        """
        numeric_cols = X.select_dtypes(include=np.number).columns
        if numeric_cols.empty:
            raise ValueError("DataFrame has no numerical features to scale.")

        for col in numeric_cols:
            analysis = self._analyze_column(X[col])
            self.analysis_[col] = analysis
            
            if self.method == 'auto':
                # Melewatkan parameter is_for_knn ke fungsi pemilihan
                selected_method, reason = self._select_optimal_method(analysis, is_for_knn=is_for_knn)
            else:
                selected_method, reason = self.method, "Method forced by user"
            
            self.fitted_methods_[col] = selected_method
            self.reasons_[col] = reason
            
            params = self._optimize_parameters(selected_method, analysis) if self.optimize else {}
            
            scaler_map = {
                'standard': StandardScaler,
                'minmax': MinMaxScaler,
                'robust': RobustScaler,
                'power': PowerTransformer
            }
            scaler = scaler_map[selected_method](**params)
            self.scalers_[col] = scaler.fit(X[[col]])
            
        self.is_fitted_ = True
        if self.verbose:
            self._print_summary()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("NoventisScaler must be fitted before using .transform()")
        
        df = X.copy()
        for col, scaler in self.scalers_.items():
            if col in df.columns:
                df[[col]] = scaler.transform(df[[col]])
        return df
            
    def fit_transform(self, X: pd.DataFrame, is_for_knn: bool = False) -> pd.DataFrame:
        return self.fit(X, is_for_knn=is_for_knn).transform(X)
        
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("NoventisScaler must be fitted before using .inverse_transform()")
        
        df = X.copy()
        for col, scaler in self.scalers_.items():
            if col in df.columns:
                df[[col]] = scaler.inverse_transform(df[[col]])
        return df

    def _print_summary(self):
        print("\n" + "📋 SCALING SUMMARY" + "\n" + "-" * 40)
        method_counts = pd.Series(self.fitted_methods_).value_counts()
        for method, count in method_counts.items():
            print(f"   - {method.upper()}: {count} columns")
        
        print("\n" + "📊 DETAILED COLUMN REPORT" + "\n" + "-" * 40)
        for col, method in self.fitted_methods_.items():
            reason = self.reasons_.get(col, "N/A")
            skewness = self.analysis_.get(col, {}).get('skewness', 0)
            outliers = self.analysis_.get(col, {}).get('outlier_ratio', 0)
            
            print(f"  Column: {col}\n"
                  f"     - Method: {method.upper()}\n"
                  f"     - Reason: {reason}\n"
                  f"     - Skewness: {skewness:.2f} | Outlier Ratio: {outliers:.2%}\n")
        print("=" * 60)

    def get_scaling_info(self) -> Dict:
        return {
            'scalers': self.scalers_,
            'fitted_methods': self.fitted_methods_,
            'analysis_results': self.analysis_,
            'selection_reasons': self.reasons_,
            'is_fitted': self.is_fitted_
        }