from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
)

list_scaler = {
    # with mean itu centering dengan mean apakah rata rata nya mau di centerd di 0 atau engga
    # with std apakah standard deviasi nya madu dijadiin satu atau engga
    "standard" : {"with_mean" : True, "with_std": True},

    # feature reange batas bawah dan atas
    "minmax" : {"feature_range" : (0,1)},
    
    # mengontrol kalo data mau di center menggunakan median, kalo iya maka
    # mengurangi setiap nilai dengan media hasilnya media dari data scaled = 0
    "robust": {"with_centering": True, "with_scaling": True, "quantile_range": (25.0, 75.0)},
    
    #  setelah di power trans mau dibuat standr scaling engga biar mea-0 dan std nya = 1
    "power": {"standardize": True, "method": 'yeo-johnson'},
}


SKEWNESS_THRESHOLD = 2.0     
NORMALITY_ALPHA = 0.05

OUTLIER_RESISTANT = ['robust', 'power']
BOUNDED_OUTPUT = ['minmax']
DISTRIBUTION_PRESERVING = ['standard', 'robust']
def analyze_data(data):

    # Pastikan array-nya bersih dari NaN dan flatten jika 2D
    clean_data = data[~np.isnan(data)].flatten()

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

    #  ngitung apakah skew atau engga
    try:
        analysis['skewness'] = abs(stats.skew(clean_data))
    except Exception:
        analysis['skewness'] = 0

    # gnitung kurtosis buat deteksi banyak gak outlier nya biasanya kalo lebih dari 0 itu banyak
    try:
        analysis['kurtosis'] = stats.kurtosis(clean_data)
    except Exception:
        analysis['kurtosis'] = 0

    # ngitung outlier menggunakan metode IQR
    q1, q3 = np.percentile(clean_data, [25, 75])
    iqr = q3 - q1
    batas_bawah = q1 - 1.5 * iqr
    batas_atas = q3 + 1.5 * iqr
    outliers = ((clean_data < batas_bawah) | (clean_data > batas_atas)).sum()

    # analysis['outlier_ratio'] = outliers / len(clean_data)
    # analysis['has_outliers'] = analysis['outlier_ratio'] > OUTLIER_THRESHOLD

    analysis['has_outliers'] = outliers > 0

    # ngitung apakah semua data bernilai positif (>= 0)
    analysis['is_positive_only'] = np.all(clean_data >= 0)

    # mendeteksi apakah semua data berada dalam rentang 0-1
    analysis['is_bounded_01'] = np.all((clean_data >= 0) & (clean_data <= 1))

    # Uji normalitas: Gunakan Shapiro-Wilk jika data <= 5000, jika lebih gunakan Anderson-Darling
    if len(clean_data) <= 5000:
        try:
            _, p_value = stats.shapiro(clean_data)
            analysis['is_normal'] = p_value > NORMALITY_ALPHA
            analysis['normality_p_value'] = p_value
        except Exception:
            analysis['is_normal'] = False
            analysis['normality_p_value'] = 0
    else:
        try:
            statistic, critical_values, _ = stats.anderson(clean_data, dist='norm')
            analysis['is_normal'] = statistic < critical_values[2]  # Level signifikansi 5%
            analysis['normality_statistic'] = statistic
        except Exception:
            analysis['is_normal'] = False
            analysis['normality_statistic'] = float('inf')

    return analysis

def select_optimal_method(analysis: Dict[str, Any], prefer_robust: bool = False) -> Tuple[str, str]:
    

    if analysis.get('skewness', 0) > SKEWNESS_THRESHOLD:
        return 'power', f"high skewness ({analysis['skewness']:.2f}) requires normalization"
    
    if analysis.get('has_outliers', False):
        return 'robust', f"outlier detected (ratio: {analysis['outlier_ratio']:.1%})"

    if analysis.get('is_bounded_01', False):
        return 'standard', "data is bounded within [0,1], standard scaling preserves distribution"

    if analysis.get('is_positive_only', False) and not prefer_robust:
        return 'minmax', "all values are positive, minmax scaling fits target range [0,1]"

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

    elif method == 'power':
        if analysis.get('is_positive_only', False):
            base_params['method'] = 'box-cox'


    elif method == 'minmax':
        if analysis.get('is_bounded_01', False):
            base_params['feature_range'] = (-1, 1)

    elif method == 'power':
        if analysis.get('is_positive_only', False):
            base_params['method'] = 'box-cox'
        else:
            base_params['method'] = 'yeo-johnson'

    elif method == 'standard':
        if analysis.get('is_bounded_01', False):
            base_params['with_mean'] = False

    return base_params


class NoventisScaler:
    def __init__(self, method: str = None, optimize: bool = True, custom_params: Dict = None):
        """
        Inisialisasi NoventisScaler.
        
        Args:
            method (str, optional): Paksa penggunaan metode tertentu ('standard', 'minmax', 
                                    'robust', 'power'). Jika None, akan dipilih otomatis. 
                                    Defaults to None.
            optimize (bool, optional): Apakah akan melakukan optimisasi parameter. 
                                       Defaults to True.
            custom_params (Dict, optional): Parameter kustom untuk dioverride. 
                                            Defaults to None.
        """
        self.method = method
        self.auto_select = method is None
        self.optimize = optimize
        self.custom_params = custom_params or {}
        
        self.scaler_ = None
        self.is_fitted_ = False
        self.selection_reason_ = "Not fitted yet."
        self.data_analysis_ = {}
        self.fitted_method_ = None
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> 'NoventisScaler':
        """
        Menganalisis data, memilih metode, dan melatih (fit) scaler.
        
        Args:
            X (pd.DataFrame atau np.ndarray): Data pelatihan.
            
        Returns:
            NoventisScaler: Instance scaler yang sudah dilatih.
        """
        if isinstance(X, pd.DataFrame):
            numeric_data = X.select_dtypes(include=np.number)
            if numeric_data.empty:
                raise ValueError("DataFrame tidak mengandung kolom numerik untuk di-scaling.")
            values = numeric_data.values
        else:
            values = X

        # 1. Analisis Data
        self.data_analysis_ = analyze_data(values)
        
        # 2. Pilih Metode
        if self.auto_select:
            selected_method, reason = select_optimal_method(self.data_analysis_)
            self.fitted_method_ = selected_method
            self.selection_reason_ = reason
        else:
            self.fitted_method_ = self.method
            self.selection_reason_ = "Metode ditentukan secara manual oleh pengguna."
            
        # 3. Optimisasi Parameter
        if self.optimize:
            params = optimize_parameters(self.fitted_method_, self.data_analysis_)
        else:
            params = LIST_SCALER[self.fitted_method_].copy()
            
        # Terapkan parameter kustom jika ada
        params.update(self.custom_params)

        # 4. Inisialisasi dan Fit Scaler
        scaler_map = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler,
            'power': PowerTransformer
        }
        self.scaler_ = scaler_map[self.fitted_method_](**params)
        
        fit_data = numeric_data if isinstance(X, pd.DataFrame) else X
        self.scaler_.fit(fit_data)
        self.is_fitted_ = True
        
        print(f"Scaler fitted. Metode terpilih: '{self.fitted_method_}'. Alasan: {self.selection_reason_}")
        
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Mengaplikasikan transformasi scaling ke data.
        
        Args:
            X (pd.DataFrame atau np.ndarray): Data yang akan ditransformasi.
            
        Returns:
            Union[pd.DataFrame, np.ndarray]: Data yang sudah di-scaling.
        """
        if not self.is_fitted_:
            raise RuntimeError("Scaler harus di-fit terlebih dahulu. Panggil .fit() sebelum .transform().")
            
        if isinstance(X, pd.DataFrame):
            numeric_cols = X.select_dtypes(include=np.number).columns
            non_numeric_cols = X.select_dtypes(exclude=np.number).columns
            
            transformed_data = self.scaler_.transform(X[numeric_cols])
            
            df_transformed = pd.DataFrame(transformed_data, index=X.index, columns=numeric_cols)
            
            # Gabungkan kembali dengan kolom non-numerik
            return pd.concat([df_transformed, X[non_numeric_cols]], axis=1)
        else:
            return self.scaler_.transform(X)
            
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Melakukan fit dan transform dalam satu langkah.
        """
        return self.fit(X).transform(X)
        
    def inverse_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Mengembalikan data yang sudah di-scaling ke skala aslinya.
        """
        if not self.is_fitted_:
            raise RuntimeError("Scaler harus di-fit terlebih dahulu sebelum .inverse_transform().")
            
        if isinstance(X, pd.DataFrame):
            numeric_cols = X.select_dtypes(include=np.number).columns
            non_numeric_cols = X.select_dtypes(exclude=np.number).columns
            
            inversed_data = self.scaler_.inverse_transform(X[numeric_cols])
            
            df_inversed = pd.DataFrame(inversed_data, index=X.index, columns=numeric_cols)
            
            return pd.concat([df_inversed, X[non_numeric_cols]], axis=1)
        else:
            return self.scaler_.inverse_transform(X)




