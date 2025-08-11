import pandas as pd
import numpy as np
from scipy.stats import skew
from typing import Dict, Tuple, Optional, Set, List

class NoventisOutlierHandler:
    """
    Menangani outlier pada kolom numerik DataFrame dengan cerdas menggunakan
    metode class-based yang konsisten dengan Scikit-Learn.

    Mendukung metode 'auto', 'quantile_trim', 'iqr_trim', 'winsorize', dan 'none'.
    Metode 'quantile_trim' dan 'iqr_trim' akan mengidentifikasi semua baris 
    outlier dari semua kolom yang relevan terlebih dahulu, lalu menghapusnya
    dalam satu operasi pada tahap transform.
    """

    def __init__(self,
                 feature_method_map: Optional[Dict[str, str]] = None,
                 default_method: str = 'auto',
                 iqr_multiplier: float = 1.5,
                 quantile_range: Tuple[float, float] = (0.05, 0.95),
                 min_data_threshold: int = 100,
                 skew_threshold: float = 1.0):
        """
        Inisialisasi NoventisOutlierHandler.

        Args:
            feature_method_map (dict, optional): Peta untuk metode spesifik per kolom.
            default_method (str): Metode fallback ('auto', 'quantile_trim', 'iqr_trim', 'winsorize', 'none'). 
            iqr_multiplier (float): Pengali untuk metode 'iqr_trim'. 
            quantile_range (tuple): Batas kuantil untuk 'quantile_trim' dan 'winsorize'. 
            min_data_threshold (int): Batas data untuk metode 'auto' memilih 'iqr_trim'. 
            skew_threshold (float): Batas skewness untuk metode 'auto'.
        """
        self.feature_method_map = feature_method_map or {}
        self.default_method = default_method or 'auto'
        self.iqr_multiplier = iqr_multiplier
        self.quantile_range = quantile_range
        self.min_data_threshold = min_data_threshold
        self.skew_threshold = skew_threshold

        self.is_fitted_ = False
        self.boundaries_: Dict[str, Tuple[float, float]] = {}
        self.methods_: Dict[str, str] = {}
        self.indices_to_drop_: Set[int] = set()

    def _choose_auto_method(self, col_data: pd.Series) -> str:
        """Fungsi helper untuk memilih metode otomatis."""
        if len(col_data.dropna()) < self.min_data_threshold:
            return 'iqr_trim' 
        elif abs(skew(col_data.dropna())) > self.skew_threshold:
            return 'winsorize'
        else:
            return 'quantile_trim' 

    def fit(self, X: pd.DataFrame, y=None) -> 'NoventisOutlierHandler':
        """
        Mempelajari batas outlier dari data training X.
        """
        df = X.copy()
        self.boundaries_ = {}
        self.methods_ = {}
        self.indices_to_drop_ = set()

        for col in df.select_dtypes(include=np.number).columns:
            # Penanganan edge case jika kolom tidak bervariasi
            if df[col].nunique() <= 1: 
                continue               

            method = self.feature_method_map.get(col, self.default_method)
            if method == 'auto':
                method = self._choose_auto_method(df[col])

            self.methods_[col] = method

            if method == 'none':
                continue

            lower_bound, upper_bound = None, None
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
                outlier_indices = df.index[(df[col] < lower_bound) | (df[col] > upper_bound)]
                self.indices_to_drop_.update(outlier_indices)
        
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Menerapkan penanganan outlier ke DataFrame.
        """
        if not self.is_fitted_:
            raise RuntimeError("Handler harus di-fit terlebih dahulu sebelum transform.")
        
        df_out = X.copy()

        for col, method in self.methods_.items():
            # Pengecekan robustness jika kolom tidak ada di data transform
            if col not in df_out.columns: 
                continue                  
            
            if method == 'winsorize':
                lower_bound, upper_bound = self.boundaries_[col]
                df_out[col] = np.clip(df_out[col], lower_bound, upper_bound)

        if self.indices_to_drop_:
            # Pastikan hanya drop indeks yang ada di dataframe saat ini
            indices_in_df = self.indices_to_drop_.intersection(df_out.index) 
            df_out.drop(index=list(indices_in_df), inplace=True) 
            
        return df_out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Melakukan fit dan transform dalam satu langkah.
        """
        return self.fit(X).transform(X)