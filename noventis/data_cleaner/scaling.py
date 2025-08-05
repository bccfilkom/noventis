from typing import Any, Dict, Literal, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
)

list_scaler = {
    "standard" : {"with_mean" : True, "with_std": True},
    "minmax" : {"feature_range" : (0,1)},
    "robust": {"with_centering": True, "with_scaling": True, "quantile_range": (25.0, 75.0)},
    "power": {"standardize": True, "method": 'yeo-johnson'},
}


SKEWNESS_THRESHOLD = 2.0     
NORMALITY_ALPHA = 0.05
OUTLIER_THRESHOLD = 2.0

OUTLIER_RESISTANT = ['robust', 'power']
BOUNDED_OUTPUT = ['minmax']
DISTRIBUTION_PRESERVING = ['standard', 'robust']
def analyze_data(data):

    clean_data = data[~np.isnan(data)]

    if len(clean_data) == 0:
        return {'error':''}

    analysis = {
        'n_samples': len(clean_data),
        'mean': np.mean(clean_data),
        'median': np.median(clean_data),
        'std': np.std(clean_data),
        'min': np.min(clean_data),
        'max': np.max(clean_data),
        'range': np.max(clean_data) - np.min(clean_data)
    }

    # calculate skewness
    try:
        analysis['skewness'] = abs(stats.skew(clean_data))
    except Exception:
        analysis['skewness'] = 0

    # calculate outlier with IQR
    q1, q3 = np.percentile(clean_data, [25, 75])
    iqr = q3 - q1
    batas_bawah = q1 - 1.5 * iqr
    batas_atas = q3 + 1.5 * iqr
    outliers = ((clean_data < batas_bawah) | (clean_data > batas_atas)).sum()

    analysis['outlier_ratio'] = outliers / len(clean_data)
    analysis['has_outliers'] = analysis['outlier_ratio'] > OUTLIER_THRESHOLD

    analysis['has_outliers'] = outliers > 0

    # check if all the data is strictly positive (x > 0)
    analysis['is_positive_only'] = np.all(clean_data > 0)

    # check if all of the data is in range 0-1
    analysis['is_bounded_01'] = np.all((clean_data >= 0) & (clean_data <= 1))

    # Normality Test: Use Shapiro-Wilk when data <= 5000, else use Anderson-Darling
    if len(clean_data) <= 5000:
        try:
            _, p_value = stats.shapiro(clean_data)
            analysis['is_normal'] = p_value > NORMALITY_ALPHA
            analysis['normality_p_value'] = p_value
            if analysis['is_normal']:
                analysis['normality_reason'] = f"p-value {p_value:.3f} > {NORMALITY_ALPHA}"
        except Exception:
            analysis['is_normal'] = False
            analysis['normality_p_value'] = 0
    else:
        try:
            statistic, critical_values, _ = stats.anderson(clean_data, dist='norm')
            analysis['is_normal'] = statistic < critical_values[2]  # significance level 5%
            analysis['normality_statistic'] = statistic
            if analysis['is_normal']:
                analysis['normality_reason'] = f"statistic {statistic:.3f} < critical value {critical_values[2]:.3f} (at significance level 5%)"
        except Exception:
            analysis['is_normal'] = False
            analysis['normality_statistic'] = float('inf')

    return analysis

def select_optimal_method(analysis: Dict[str, Any], prefer_robust: bool = False, is_knn: bool=False) -> Tuple[str, str]:
    if analysis.get('skewness', 0) > SKEWNESS_THRESHOLD:
        return 'power', f"high skewness ({analysis['skewness']:.2f}) requires normalization"
    
    if analysis.get('has_outliers', False):
        return 'robust', f"outlier detected (ratio: {analysis['outlier_ratio']:.1%})"

    if is_knn:
        return 'minmax', "KNN algorithms requires minmax scaling"

    if analysis.get('is_normal', False):
        return 'standard', f"normal distribution detected (p-value: {analysis.get('normality_p_value', 'n/a')})"

    return ('robust' if prefer_robust else 'standard',
            "default fallback for unclear data characteristics")

def optimize_parameters(method: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    base_params = list_scaler[method].copy()
    if method == 'robust':
        outlier_ratio = analysis.get('outlier_ratio', 0)
        if outlier_ratio > 0.01:
            lower_q = (outlier_ratio / 2) * 100
            upper_q = 100 - lower_q
            base_params['quantile_range'] = (max(10.0, lower_q), min(90.0, upper_q))

    elif method == 'minmax':
        if analysis.get('is_bounded_01', False):
            base_params['feature_range'] = (-1, 1)

    elif method == 'power':
        if analysis.get('is_positive_only', False):
            base_params['method'] = 'box-cox'
        else:
            base_params['method'] = 'yeo-johnson'

    return base_params


class NoventisScaler:
    def __init__(self, method: Literal['standard', 'minmax', 'robust', 'power'] = None, optimize: bool = True, custom_params: Dict = None):
        """
        NoventisScaler Constructor
        
        Args:
            method (str, optional): forced user to choose 1 of the following methods: 
                                    ['standard', 'minmax', 'robust', 'power'].
                                    Defaults to None
            optimize (bool, optional): options to optimize parameters. 
                                       Defaults to True.
            custom_params (Dict, optional): custom parameters to override the optimized parameters.
                                            Defaults to None.
        """
        # check valid methods
        allowed_methods = ['standard', 'minmax', 'robust', 'power']
        if method not in allowed_methods:
            raise ValueError(f"Invalid method. Allowed methods are: {allowed_methods}")

        self.method = method
        self.auto_select = method is None
        self.optimize = optimize
        self.custom_params = custom_params or {}
        
        self.scaler_ = None
        self.is_fitted_ = False
        self.selection_reason_ = "Not fitted yet."
        self.data_analysis_ = {}
        self.fitted_method_ = None

        self.standard_cols = []
        self.minmax_cols = []
        self.power_cols = []
        self.robust_cols = []

        self.standard_scaler = []
        self.minmax_scaler = []
        self.power_transformer = []
        self.robust_scaler = []
        
    def fit(self, X: pd.DataFrame) -> 'NoventisScaler':
        """
        consists of 3 steps:
        1. analyze the data -> analyze_data()
        2. select the most optimal scaler methods -> select_optimal_method()
        3. fit the scaler 
        
        Args:
            X (pd.DataFrame): Pandas Dataframe.
            
        Returns:
            NoventisScaler: scaler instances that has been fitted (stored in lists based on the scaler methods)
        """
        fitted_df = X.copy()
        numeric_data = X.select_dtypes(include=np.number)
        if numeric_data.empty:
            raise ValueError("Dataframe has no available numeric features to be scaled.")
        columns = numeric_data.columns

        # Initialize the list
        self.standard_cols = []
        self.minmax_cols = []
        self.power_cols = []
        self.robust_cols = []

        self.standard_scaler = []
        self.minmax_scaler = []
        self.power_transformer = []
        self.robust_scaler = []

        for i in range(len(columns)):
            # 1. analyze data
            self.data_analysis_ = analyze_data(numeric_data[columns[i]])
            
            # 2. selecting optimal method
            if self.auto_select:
                selected_method, reason = select_optimal_method(self.data_analysis_)
                self.fitted_method_ = selected_method
                self.selection_reason_ = reason
            else:
                self.fitted_method_ = self.method
                self.selection_reason_ = "Method was picked manually by user."
                
            # 3. parameter optimization
            if self.optimize:
                params = optimize_parameters(self.fitted_method_, self.data_analysis_)
            else:
                params = list_scaler[self.fitted_method_].copy()
                
            # add/override with custom parameter if exist
            params.update(self.custom_params)

            # 4. Initialize and fit scaler
            scaler_map = {
                'standard': StandardScaler,
                'minmax': MinMaxScaler,
                'robust': RobustScaler,
                'power': PowerTransformer
            }
            self.scaler_ = scaler_map[self.fitted_method_](**params)    

            if self.fitted_method_ == 'standard':
                self.standard_scaler.append(self.scaler_.fit(fitted_df[[columns[i]]]))
            
            elif self.fitted_method_ == 'minmax':
                self.minmax_scaler.append(self.scaler_.fit(fitted_df[[columns[i]]]))

            elif self.fitted_method_ == 'power':
                self.power_transformer.append(self.scaler_.fit(fitted_df[[columns[i]]]))

            elif self.fitted_method_ == 'robust':
                self.robust_scaler.append(self.scaler_.fit(fitted_df[[columns[i]]]))
            
            # 5. append numerical columns into a list based on their scaling method
            scaler_list = getattr(self, f"{self.fitted_method_}_cols")
            scaler_list.append(columns[i])            
            
            print(f"+ Column: '{columns[i]}'. Method: '{self.fitted_method_}'. Reason: {self.selection_reason_}")
        
        self.is_fitted_ = True
        print(f"NoventisScaler: Fitting complete for {len(columns)} numerical features.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        transform data after fitting the scaler
        
        Args:
            X (pd.DataFrame): Pandas Dataframe.
            
        Returns:
            pd.DataFrame: Pandas Dataframe that has been transformed.
        """
        if not self.is_fitted_:
            raise RuntimeError("NoventisScaler has to be fitted! \nCall .fit() before using .transform().")
            
        if isinstance(X, pd.DataFrame):
            transformed_df = X.copy()
            
            # standard scaler
            for i in range(len(self.standard_cols)):
                cols = self.standard_cols[i]
                transformed_df[[cols]] = self.standard_scaler[i].transform(transformed_df[[cols]])

            # minmax scaler
            for i in range(len(self.minmax_cols)):
                cols = self.minmax_cols[i]
                transformed_df[[cols]] = self.minmax_scaler[i].transform(transformed_df[[cols]])

            # power transformer
            for i in range(len(self.power_cols)):
                cols = self.power_cols[i]
                transformed_df[[cols]] = self.power_transformer[i].transform(transformed_df[[cols]])

            # robust scaler
            for i in range(len(self.robust_cols)):
                cols = self.robust_cols[i]
                transformed_df[[cols]] = self.robust_scaler[i].transform(transformed_df[[cols]])


            return transformed_df
        else:
            return self.scaler_.transform(X)
            
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        fit and transform in 1 step
        """
        return self.fit(X).transform(X)
        
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        revert scaled data to its original scale
        """
        df = X.copy()
        if not self.is_fitted_:
            raise RuntimeError("NoventisScaler has to be fitted! \nCall .fit() before using inverse_transform().")
            
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = self.scaler_.inverse_transform(df[numeric_cols])
        
        return df




