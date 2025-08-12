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

    def plot_comparison(self, max_cols: int = 5):
        """
        Membuat visualisasi perbandingan distribusi data sebelum dan sesudah 
        penanganan outlier untuk beberapa kolom numerik.

        Args:
            max_cols (int): Jumlah maksimum kolom yang akan divisualisasikan.
        """
        if not self.is_fitted_ or self._original_df_snapshot is None:
            print("Handler belum di-fit. Jalankan .fit() atau .fit_transform() terlebih dahulu.")
            return

        # Ambil data setelah transform (perlu panggil transform lagi jika belum ada)
        # Cara sederhana: asumsikan pengguna memanggil ini setelah fit_transform
        # Cara lebih robust: perlu menyimpan df_out, tapi bisa memakan memori.
        # Kita mulai dengan cara sederhana.

        print("Membuat visualisasi perbandingan...")
        
        # Ambil hanya kolom yang diproses
        cols_to_plot = list(self.methods_.keys())[:max_cols]
        
        # DataFrame asli dan yang sudah ditransformasi (dengan asumsi baris yang sama)
        original_data = self._original_df_snapshot
        # Untuk perbandingan, kita perlu data transform, tapi karena baris bisa hilang,
        # kita perlu cara untuk membandingkannya. Salah satu caranya adalah plot distribusi.
        
        # Untuk plot ini, kita akan panggil transform lagi secara internal
        # Ini tidak efisien, tapi paling mudah untuk perbandingan visual
        transformed_data = self.transform(original_data.copy())

        for col in cols_to_plot:
            if col not in transformed_data.columns:
                continue
                
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Perbandingan Distribusi untuk Kolom '{col}'", fontsize=16)

            # Plot Sebelum
            sns.histplot(original_data[col], kde=True, ax=axes[0], color='skyblue')
            axes[0].set_title(f"Sebelum ({self.quality_report_['rows_before']} baris)")
            axes[0].axvline(original_data[col].mean(), color='r', linestyle='--', label='Mean')
            axes[0].axvline(original_data[col].median(), color='g', linestyle='-', label='Median')
            axes[0].legend()

            # Plot Sesudah
            sns.histplot(transformed_data[col], kde=True, ax=axes[1], color='lightgreen')
            axes[1].set_title(f"Sesudah ({self.quality_report_['rows_after']} baris)")
            axes[1].axvline(transformed_data[col].mean(), color='r', linestyle='--', label='Mean')
            axes[1].axvline(transformed_data[col].median(), color='g', linestyle='-', label='Median')
            axes[1].legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()