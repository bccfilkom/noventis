import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from typing import Dict, Any, Optional, List, Union
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class NoventisImputer:
    """
    Menangani nilai yang hilang (null/NaN) secara cerdas dengan metode class-based.
    Dilengkapi dengan laporan kualitas dan ringkasan otomatis.
    """

    def __init__(self, 
                 method: Optional[Union[str, Dict[str, str]]] = None, 
                 columns: Optional[List[str]] = None,
                 fill_value: Any = None,
                 n_neighbors: int = 5,
                 verbose: bool = True):
        """
        Inisialisasi NoventisImputer.

        Args:
            method (str, dict, or None): Metode imputasi. 
                - str: 'mean', 'median', 'mode', 'knn', 'constant', 'ffill', 'bfill', 'drop'.
                - dict: Metode spesifik per kolom, misal: {'Usia': 'median'}.
                - None: Mode otomatis ('median' untuk numerik, 'mode' untuk kategorikal).
            columns (list, optional): Daftar kolom spesifik yang ingin diproses.
            fill_value (any, optional): Nilai yang digunakan jika method='constant'.
            n_neighbors (int): Jumlah tetangga untuk metode 'knn'.
            verbose (bool, optional): Jika True, cetak ringkasan setelah fit. Defaults to True.
        """
        self.method = method
        self.columns = columns
        self.fill_value = fill_value
        self.n_neighbors = n_neighbors
        self.verbose = verbose
        
        # --- Atribut Internal ---
        self.is_fitted_ = False
        self.imputers_: Dict[str, Any] = {}
        self.columns_to_process_: List[str] = []
        self._column_methods: Dict[str, str] = {}
        
        # --- Atribut Laporan & Visualisasi ---
        self.quality_report_: Dict[str, Any] = {}
        self._original_df_snapshot: Optional[pd.DataFrame] = None

    # --- Metode Inti ---

    def fit(self, X: pd.DataFrame, y=None) -> 'NoventisImputer':
        """Mempelajari strategi imputasi dari data X."""
        self._original_df_snapshot = X.copy()
        self.imputers_ = {}
        self._column_methods = {}
        
        cols_with_null = [col for col in X.columns if X[col].isnull().sum() > 0]
        self.columns_to_process_ = [c for c in (self.columns or cols_with_null) if c in cols_with_null]

        # Logika utama untuk fitting
        self._fit_imputers(X)
        
        self.is_fitted_ = True
        
        # Panggil ringkasan jika verbose, laporan lengkap akan dihitung saat dipanggil
        if self.verbose:
            self._print_summary()
            
        return self

    def _fit_imputers(self, X: pd.DataFrame):
        """Logika internal untuk melakukan fitting pada setiap imputer."""
        # Penanganan KNN secara terpisah karena ia menangani banyak kolom sekaligus
        if self.method == 'knn':
            num_cols = [c for c in self.columns_to_process_ if pd.api.types.is_numeric_dtype(X[c])]
            if num_cols:
                knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
                knn_imputer.fit(X[num_cols])
                self.imputers_['knn'] = {'imputer': knn_imputer, 'cols': num_cols}
                for col in num_cols: 
                    self._column_methods[col] = 'knn'
        
        for col in self.columns_to_process_:
            if col in self._column_methods: continue

            chosen_method = self.method
            if isinstance(self.method, dict):
                chosen_method = self.method.get(col, None)
            
            if chosen_method is None: # Mode Otomatis
                chosen_method = 'median' if pd.api.types.is_numeric_dtype(X[col]) else 'mode'

            self._column_methods[col] = chosen_method
            
            # Metode yang tidak butuh fitting (hanya dicatat)
            if chosen_method in ['drop', 'ffill', 'bfill', 'knn']:
                self.imputers_[col] = chosen_method
                continue
            
            # Metode yang butuh fitting (SimpleImputer)
            strategy, fill_val = '', self.fill_value
            if chosen_method == 'mean': strategy = 'mean'
            elif chosen_method == 'median': strategy = 'median'
            elif chosen_method == 'mode': strategy = 'most_frequent'
            elif chosen_method == 'constant': strategy = 'constant'
            else: strategy = 'most_frequent' # Fallback
            
            imp = SimpleImputer(strategy=strategy, fill_value=fill_val)
            self.imputers_[col] = imp.fit(X[[col]])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Menerapkan imputasi yang telah di-fit ke data."""
        if not self.is_fitted_:
            raise RuntimeError("Imputer harus di-fit terlebih dahulu.")
        
        df_out = X.copy()
        
        if 'knn' in self.imputers_:
            knn_info = self.imputers_['knn']
            cols_to_transform = [c for c in knn_info['cols'] if c in df_out.columns]
            if cols_to_transform:
                df_out[cols_to_transform] = knn_info['imputer'].transform(df_out[cols_to_transform])

        for col, imputer in self.imputers_.items():
            if col == 'knn' or col not in df_out.columns: continue
            
            if imputer == 'drop': df_out.dropna(subset=[col], inplace=True)
            elif imputer in ['ffill', 'bfill']: df_out[col].fillna(method=imputer, inplace=True)
            else: df_out[[col]] = imputer.transform(df_out[[col]])
            
        return df_out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Melakukan fit dan transform dalam satu langkah."""
        return self.fit(X, y).transform(X)

    # --- Metode Laporan & Visualisasi ---

    def get_quality_report(self) -> Dict[str, Any]:
        """Menghitung dan mengembalikan laporan kualitas detail dari proses imputasi."""
        if not self.is_fitted_ or self._original_df_snapshot is None:
            print("Imputer belum di-fit.")
            return {}

        df_before = self._original_df_snapshot
        df_after = self.transform(df_before.copy())

        missing_before = df_before[self.columns_to_process_].isnull().sum().sum() if self.columns_to_process_ else 0
        processed_cols = [c for c in self.columns_to_process_ if c in df_after.columns]
        missing_after = df_after[processed_cols].isnull().sum().sum() if processed_cols else 0
        
        values_imputed = missing_before - missing_after
        score = (values_imputed / missing_before * 100) if missing_before > 0 else 100.0

        report = {'overall_summary': {}, 'column_details': {}}
        report['overall_summary'] = {
            'total_missing_before': int(missing_before),
            'total_missing_after': int(missing_after),
            'total_values_imputed': int(values_imputed),
            'completion_score': f"{score:.2f}%"
        }
        for col in self.columns_to_process_:
            report['column_details'][col] = {
                'missing_before': int(df_before[col].isnull().sum()),
                'method': self._column_methods.get(col, 'N/A')
            }
        
        self.quality_report_ = report
        return self.quality_report_

    def _print_summary(self):
        """Mencetak ringkasan yang mudah dibaca ke konsol."""
        report = self.get_quality_report()
        summary = report.get('overall_summary', {})
        
        print("\n📋" + "="*23 + " IMPUTATION SUMMARY " + "="*23 + "📋")
        method_str = self.method if isinstance(self.method, str) else 'CUSTOM MAP'
        print(f"{'Method':<25} | {method_str if self.method is not None else 'AUTO'}")
        print(f"{'Total Values Imputed':<25} | {summary.get('total_values_imputed', 'N/A')}")
        print(f"{'Completion Score':<25} | {summary.get('completion_score', 'N/A')}")
        print("="*68)

    def plot_comparison(self, max_cols: int = 3, color_before: str = '#FF6849', color_after: str = '#0F2CAB'):
        if not self.is_fitted_ or self._original_df_snapshot is None:
            print("Imputer belum di-fit.")
            return

        print("\n📈 Membuat visualisasi perbandingan untuk imputasi...")
        
        original_data = self._original_df_snapshot
        transformed_data = self.transform(original_data.copy())
        
        numeric_cols_processed = [
            col for col in self.columns_to_process_ 
            if pd.api.types.is_numeric_dtype(original_data[col]) and original_data[col].isnull().sum() > 0
        ][:max_cols]

        if not numeric_cols_processed:
            print("Tidak ada kolom numerik yang diimputasi untuk divisualisasikan.")
            return

        for col in numeric_cols_processed:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Perbandingan Distribusi untuk Kolom '{col}'", fontsize=16)
            
            # Plot Sebelum
            sns.histplot(original_data[col].dropna(), kde=True, ax=axes[0], color=color_before)
            axes[0].set_title("Sebelum (Distribusi Nilai Asli)")
            axes[0].axvline(original_data[col].mean(), color='r', linestyle='--', label='Mean')
            axes[0].axvline(original_data[col].median(), color='g', linestyle='-', label='Median')
            axes[0].legend()

            # Plot Sesudah
            sns.histplot(transformed_data[col], kde=True, ax=axes[1], color=color_after)
            axes[1].set_title("Sesudah (Distribusi Setelah Imputasi)")
            axes[1].axvline(transformed_data[col].mean(), color='r', linestyle='--', label='Mean')
            axes[1].axvline(transformed_data[col].median(), color='g', linestyle='-', label='Median')
            axes[1].legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()