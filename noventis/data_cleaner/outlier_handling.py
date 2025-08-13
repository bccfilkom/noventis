import pandas as pd
import numpy as np
from scipy.stats import skew
from typing import Dict, Tuple, Optional, Set, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

class NoventisOutlierHandler:
    """
    Menangani outlier pada kolom numerik dengan metode cerdas.
    Dilengkapi dengan laporan kualitas, ringkasan, dan visualisasi otomatis.
    """

    def __init__(self,
                 feature_method_map: Optional[Dict[str, str]] = None,
                 default_method: str = 'auto',
                 iqr_multiplier: float = 1.5,
                 quantile_range: Tuple[float, float] = (0.05, 0.95),
                 min_data_threshold: int = 100,
                 skew_threshold: float = 1.0,
                 verbose: bool = True):
        """
        Inisialisasi NoventisOutlierHandler.

        Args:
            feature_method_map (dict, optional): Peta untuk metode spesifik per kolom.
            default_method (str): Metode fallback ('auto', 'quantile_trim', 'iqr_trim', 'winsorize', 'none').
            iqr_multiplier (float): Pengali untuk metode 'iqr_trim'.
            quantile_range (tuple): Batas kuantil untuk 'quantile_trim' dan 'winsorize'.
            verbose (bool, optional): Jika True, cetak ringkasan setelah fit. Defaults to True.
        """
        self.feature_method_map = feature_method_map or {}
        self.default_method = default_method or 'auto'
        self.iqr_multiplier = iqr_multiplier
        self.quantile_range = quantile_range
        self.min_data_threshold = min_data_threshold
        self.skew_threshold = skew_threshold
        self.verbose = verbose

        # --- Atribut Internal ---
        self.is_fitted_ = False
        self.boundaries_: Dict[str, Tuple[float, float]] = {}
        self.methods_: Dict[str, str] = {}
        self.reasons_: Dict[str, str] = {}
        self.indices_to_drop_: Set[int] = set()
        
        # --- Atribut Laporan & Visualisasi ---
        self.quality_report_: Dict[str, Any] = {}
        self._original_df_snapshot: Optional[pd.DataFrame] = None

    # --- Metode Inti ---

    def _choose_auto_method(self, col_data: pd.Series) -> Tuple[str, str]:
        """Fungsi helper untuk memilih metode otomatis dan alasannya."""
        if len(col_data.dropna()) < self.min_data_threshold:
            return 'iqr_trim', f"Data < {self.min_data_threshold} samples"
        elif abs(skew(col_data.dropna())) > self.skew_threshold:
            return 'winsorize', f"High skewness ({abs(skew(col_data.dropna())):.2f})"
        else:
            return 'quantile_trim', "Default fallback"

    def fit(self, X: pd.DataFrame, y=None) -> 'NoventisOutlierHandler':
        self._original_df_snapshot = X.copy()
        self.boundaries_, self.methods_, self.reasons_, self.indices_to_drop_ = {}, {}, {}, set()

        for col in X.select_dtypes(include=np.number).columns:
            if X[col].nunique() <= 1: continue

            method = self.feature_method_map.get(col, self.default_method)
            reason = "Forced by user"
            if method == 'auto':
                method, reason = self._choose_auto_method(X[col])

            self.methods_[col] = method
            self.reasons_[col] = reason
            if method == 'none': continue

            lower_bound, upper_bound = None, None
            if method in ['quantile_trim', 'winsorize']:
                q_low, q_high = X[col].quantile(self.quantile_range)
                lower_bound, upper_bound = q_low, q_high
            elif method == 'iqr_trim':
                Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound, upper_bound = Q1 - self.iqr_multiplier * IQR, Q3 + self.iqr_multiplier * IQR

            self.boundaries_[col] = (lower_bound, upper_bound)
            if method in ['quantile_trim', 'iqr_trim']:
                outlier_indices = X.index[(X[col] < lower_bound) | (X[col] > upper_bound)]
                self.indices_to_drop_.update(outlier_indices)
        
        self.is_fitted_ = True
        
        # Panggil ringkasan jika verbose, tapi laporan lengkap dibuat di transform
        if self.verbose:
            self._print_summary(X)
            
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Handler harus di-fit terlebih dahulu.")
        
        df_out = X.copy()
        for col, method in self.methods_.items():
            if col not in df_out.columns: continue
            if method == 'winsorize':
                lower_bound, upper_bound = self.boundaries_[col]
                df_out[col] = np.clip(df_out[col], lower_bound, upper_bound)

        if self.indices_to_drop_:
            indices_in_df = self.indices_to_drop_.intersection(df_out.index)
            df_out.drop(index=list(indices_in_df), inplace=True)
        
        self._create_quality_report(X, df_out)
        return df_out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # --- Metode Laporan & Visualisasi ---

    def _create_quality_report(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        """Membuat laporan kualitas lengkap setelah transformasi."""
        report = {'overall_summary': {}, 'column_details': {}}
        rows_before = len(df_before)
        rows_after = len(df_after)
        rows_removed = rows_before - rows_after
        retained_score = (rows_after / rows_before * 100) if rows_before > 0 else 100.0

        report['overall_summary'] = {
            'rows_before': rows_before,
            'rows_after': rows_after,
            'outliers_removed (rows)': rows_removed,
            'data_retained_score': f"{retained_score:.2f}%"
        }

        for col, method in self.methods_.items():
            report['column_details'][col] = {
                'method': method.upper(),
                'reason': self.reasons_.get(col, 'N/A')
            }
        self.quality_report_ = report

    def get_quality_report(self) -> Dict[str, Any]:
        """Mengembalikan laporan kualitas detail dari proses penanganan outlier."""
        if not self.is_fitted_:
            print("Handler belum di-fit.")
            return {}
        return self.quality_report_

    def _print_summary(self, X: pd.DataFrame):
        """Mencetak ringkasan yang mudah dibaca ke konsol."""
        # Perlu .transform() dummy untuk mendapatkan hasil akhir
        df_after = self.transform(X.copy())
        report = self.get_quality_report()
        summary = report.get('overall_summary', {})
        
        print("\n📋" + "="*23 + " OUTLIER HANDLING SUMMARY " + "="*23 + "📋")
        print(f"{'Method':<25} | {self.default_method.upper() if self.default_method == 'auto' else 'CUSTOM MAP'}")
        print(f"{'Total Rows Removed':<25} | {summary.get('outliers_removed (rows)', 'N/A')}")
        print(f"{'Data Retained Score':<25} | {summary.get('data_retained_score', 'N/A')}")
        print("="*72)

    def plot_comparison(self, max_cols: int = 3, color_before: str = '#FF6849', color_after: str = '#31B7AE'):
        if not self.is_fitted_ or self._original_df_snapshot is None:
            print("Handler belum di-fit.")
            return

        print("\n📈 Membuat visualisasi perbandingan untuk penanganan outlier...")
        
        original_data = self._original_df_snapshot
        transformed_data = self.transform(original_data.copy())
        cols_to_plot = [col for col in self.methods_ if self.methods_[col] != 'none'][:max_cols]

        if not cols_to_plot:
            print("Tidak ada kolom yang diproses untuk divisualisasikan.")
            return

        for col in cols_to_plot:
            if col not in original_data.columns: continue
                
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Perbandingan Distribusi untuk Kolom '{col}' (Metode: {self.methods_[col].upper()})", fontsize=16)
            
            # Plot Sebelum
            sns.histplot(original_data[col], kde=True, ax=axes[0], color=color_before)
            axes[0].set_title(f"Sebelum ({len(original_data)} baris)")
            axes[0].legend(['Mean', 'Median'])

            # Plot Sesudah
            if col in transformed_data.columns:
                sns.histplot(transformed_data[col], kde=True, ax=axes[1], color=color_after)
                axes[1].set_title(f"Sesudah ({len(transformed_data)} baris)")
                axes[1].legend(['Mean', 'Median'])
            else:
                axes[1].text(0.5, 0.5, 'Kolom dihapus', ha='center', va='center')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()