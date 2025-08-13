import pandas as pd
import numpy as np
from scipy.stats import skew
from typing import Dict, Tuple, Optional, Set, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

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

        # Atribut baru untuk laporan
        self.quality_report_: Dict[str, Any] = {}
        self._original_df_snapshot: Optional[pd.DataFrame] = None

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
        self._original_df_snapshot = df # <-- Simpan snapshot data asli
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
            
        rows_before = len(X)
        rows_after = len(df_out)
        outliers_removed = rows_before - rows_after
        removal_percentage = (outliers_removed / rows_before * 100) if rows_before > 0 else 0

        self.quality_report_ = {
            'rows_before': rows_before,
            'rows_after': rows_after,
            'outliers_removed': outliers_removed,
            'removal_percentage': f"{removal_percentage:.2f}%"
        }
        return df_out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Melakukan fit dan transform dalam satu langkah.
        """
        return self.fit(X).transform(X)
    
    # Tambahkan metode baru ini di dalam kelas NoventisOutlierHandler

    def get_quality_report(self) -> Dict[str, Any]:
        """
        Mengembalikan ringkasan statistik dan skor dari proses penanganan outlier.

        Returns:
            dict: Kamus berisi metrik sebelum dan sesudah proses.
        """
        if not self.is_fitted_:
            print("Handler belum di-fit. Jalankan .fit() atau .fit_transform() terlebih dahulu.")
            return {}
        return self.quality_report_

    def plot_comparison(self, max_cols: int = 1):
        if not self.is_fitted_ or self._original_df_snapshot is None: return
        cols_to_plot = list(self.methods_.keys())[:max_cols]
        if not cols_to_plot: return

        original_data = self._original_df_snapshot
        transformed_data = self.transform(original_data.copy())
        col = cols_to_plot[0]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(f"Perbandingan Penanganan Outlier untuk '{col}'", fontsize=16)
        
        # Hapus spasi antar subplot vertikal
        fig.subplots_adjust(hspace=0)

        # Sebelum
        sns.histplot(original_data[col], kde=True, ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title("Sebelum")
        axes[0, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Sembunyikan label x
        sns.boxplot(x=original_data[col], ax=axes[1, 0], color='skyblue')

        # Sesudah
        sns.histplot(transformed_data[col], kde=True, ax=axes[0, 1], color='lightgreen')
        axes[0, 1].set_title("Sesudah")
        axes[0, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Sembunyikan label x
        sns.boxplot(x=transformed_data[col], ax=axes[1, 1], color='lightgreen')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig