import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns

class NoventisImputer:
    """
    Menangani nilai yang hilang (null/NaN) secara cerdas dengan metode class-based.
    
    Fitur:
    - Sesuai dengan API Scikit-learn (fit, transform).
    - Deteksi otomatis tipe kolom (numerik & kategorikal).
    - Mendukung metode: 'mean', 'median', 'mode', 'knn', 'constant', 'ffill', 'bfill', 'drop'.
    - Fleksibel: dapat menerima metode global (string) atau per kolom (dictionary).
    - Penanganan khusus untuk kolom integer (pembulatan otomatis).
    """

    def __init__(self, 
                 method: Optional[str] = None, 
                 columns: Optional[List[str]] = None,
                 fill_value: Any = None,
                 n_neighbors: int = 5):
        """
        Inisialisasi NoventisImputer.

        Args:
            method (str, dict, or None): 
                - str: 'mean', 'median', 'mode', 'knn', 'constant', 'ffill', 'bfill', 'drop' untuk semua kolom.
                - dict: Metode spesifik per kolom, misal: {'Usia': 'median'}.
                - None: Mode otomatis ('mean' untuk numerik, 'mode' untuk kategorikal).
            columns (list, optional): Daftar kolom spesifik yang ingin diproses. 
                                      Jika None, proses semua kolom yang relevan.
            fill_value (any, optional): Nilai yang digunakan jika method='constant'.
            n_neighbors (int): Jumlah tetangga untuk metode 'knn'.
        """
        self.method = method
        self.columns = columns
        self.fill_value = fill_value
        self.n_neighbors = n_neighbors
        
        self.is_fitted_ = False
        self.imputers_: Dict[str, Any] = {}
        self.columns_to_process_: List[str] = []
        self.quality_report_: Dict[str, Any] = {}
        self._original_df_snapshot: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y=None) -> 'NoventisImputer':
        """
        Mempelajari strategi imputasi dari data training X.
        """
        df = X.copy()
        self._original_df_snapshot = df
        self.imputers_ = {}
        
        # Tentukan kolom yang akan diproses
        cols_with_null = [col for col in df.columns if df[col].isnull().sum() > 0]
        if self.columns:
            self.columns_to_process_ = [col for col in self.columns if col in cols_with_null]
        else:
            self.columns_to_process_ = cols_with_null

        # Penanganan khusus dan efisien untuk KNN
        if self.method == 'knn':
            num_cols = [c for c in df.select_dtypes(include=np.number).columns if c in self.columns_to_process_]
            cat_cols = [c for c in df.select_dtypes(include='object').columns if c in self.columns_to_process_]
            
            if num_cols:
                knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
                knn_imputer.fit(df[num_cols])
                self.imputers_['knn'] = {'imputer': knn_imputer, 'cols': num_cols}

            # Untuk kolom kategorikal dalam mode KNN, gunakan mode sebagai fallback
            for col in cat_cols:
                imp = SimpleImputer(strategy='most_frequent')
                self.imputers_[col] = imp.fit(df[[col]])
        else:
            # Penanganan untuk metode lain (per kolom)
            for col in self.columns_to_process_:
                chosen_method = self.method
                if isinstance(self.method, dict):
                    chosen_method = self.method.get(col, None)
                
                # Fallback Otomatis
                if chosen_method is None:
                    chosen_method = 'mean' if pd.api.types.is_numeric_dtype(df[col]) else 'mode'
                
                # Metode yang tidak butuh fitting
                if chosen_method in ['drop', 'ffill', 'bfill']:
                    self.imputers_[col] = chosen_method
                    continue

                strategy = ''
                fill_val = None
                is_numeric = pd.api.types.is_numeric_dtype(df[col])

                if chosen_method == 'mean' and is_numeric:
                    # Cek apakah kolom berisi integer untuk pembulatan
                    if np.all(df[col].dropna() % 1 == 0):
                        strategy = 'constant'
                        fill_val = round(df[col].mean())
                    else:
                        strategy = 'mean'
                elif chosen_method == 'median' and is_numeric:
                    strategy = 'median'
                elif chosen_method == 'mode':
                    strategy = 'most_frequent'
                elif chosen_method == 'constant':
                    strategy = 'constant'
                    fill_val = self.fill_value # Menggunakan fill_value dari user
                else: # Fallback untuk kategorikal
                    strategy = 'most_frequent'
                
                imp = SimpleImputer(strategy=strategy, fill_value=fill_val)
                self.imputers_[col] = imp.fit(df[[col]])
        
        self.is_fitted_ = True

        cols_with_null = [col for col in df.columns if df[col].isnull().sum() > 0]
        if self.columns:
            cols_to_process = [col for col in self.columns if col in cols_with_null]
        else:
            cols_to_process = cols_with_null
            
        # Simpan kolom yang akan diproses untuk digunakan nanti
        self.columns_to_process_ = cols_to_process

        if not self.columns_to_process_:
            missing_before = 0
        else:
            missing_before = df[self.columns_to_process_].isnull().sum().sum()

        self.quality_report_ = {
            'missing_values_before': int(missing_before)
        }

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Menerapkan imputasi nilai hilang ke DataFrame.
        """
        if not self.is_fitted_:
            raise RuntimeError("Imputer harus di-fit terlebih dahulu sebelum transform.")
            
        df_out = X.copy()
        
        # Handle KNN
        if 'knn' in self.imputers_:
            knn_info = self.imputers_['knn']
            # Pastikan kolom ada di dataframe transform
            cols_to_transform = [c for c in knn_info['cols'] if c in df_out.columns]
            if cols_to_transform:
                df_out[cols_to_transform] = knn_info['imputer'].transform(df_out[cols_to_transform])
        
        # Handle metode lainnya
        for col, imputer in self.imputers_.items():
            if col == 'knn' or col not in df_out.columns:
                continue
            
            if imputer == 'drop':
                df_out.dropna(subset=[col], inplace=True)
            elif imputer in ['ffill', 'bfill']:
                df_out[col].fillna(method=imputer, inplace=True)
            else: # SimpleImputer
                df_out[[col]] = imputer.transform(df_out[[col]])
        
        if not self.columns_to_process_:
            missing_after = 0
        else:
            # Pastikan hanya menghitung di kolom yang ada di df_out
            processed_cols_in_df = [col for col in self.columns_to_process_ if col in df_out.columns]
            missing_after = df_out[processed_cols_in_df].isnull().sum().sum()

        missing_before = self.quality_report_.get('missing_values_before', 0)
        values_imputed = missing_before - missing_after
        completion_score = (values_imputed / missing_before * 100) if missing_before > 0 else 100.0

        self.quality_report_.update({
            'missing_values_after': int(missing_after),
            'values_imputed': int(values_imputed),
            'completion_score': f"{completion_score:.2f}%"
        })

        return df_out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Melakukan fit dan transform dalam satu langkah.
        """
        return self.fit(X, y).transform(X)
    
    def get_quality_report(self) -> Dict[str, Any]:
        """
        Mengembalikan ringkasan statistik dari proses imputasi nilai hilang.

        Returns:
            dict: Kamus berisi metrik sebelum dan sesudah proses imputasi.
        """
        if not self.is_fitted_:
            print("Imputer belum di-fit. Jalankan .fit() atau .fit_transform() terlebih dahulu.")
            return {}
        return self.quality_report_

    def plot_comparison(self, max_cols: int = 5):
        """
        Membuat visualisasi perbandingan distribusi data sebelum dan sesudah imputasi.
        Hanya akan memvisualisasikan kolom numerik yang diimputasi.

        Args:
            max_cols (int): Jumlah maksimum kolom yang akan divisualisasikan.
        """
        if not self.is_fitted_ or self._original_df_snapshot is None:
            print("Imputer belum di-fit. Jalankan .fit() atau .fit_transform() terlebih dahulu.")
            return

        print("Membuat visualisasi perbandingan untuk imputasi...")
        
        original_data = self._original_df_snapshot
        transformed_data = self.transform(original_data.copy())
        
        # Ambil hanya kolom numerik yang diproses untuk di-plot
        numeric_cols_processed = [
            col for col in self.columns_to_process_ 
            if pd.api.types.is_numeric_dtype(original_data[col])
        ][:max_cols]

        if not numeric_cols_processed:
            print("Tidak ada kolom numerik yang diimputasi untuk divisualisasikan.")
            return

        for col in numeric_cols_processed:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Perbandingan Distribusi untuk Kolom '{col}'", fontsize=16)

            # Plot Sebelum (hanya plot nilai yang tidak null)
            sns.histplot(original_data[col].dropna(), kde=True, ax=axes[0], color='skyblue')
            axes[0].set_title(f"Sebelum (Distribusi Nilai Asli)")
            axes[0].axvline(original_data[col].mean(), color='r', linestyle='--', label='Mean')
            axes[0].axvline(original_data[col].median(), color='g', linestyle='-', label='Median')
            axes[0].legend()

            # Plot Sesudah (plot semua nilai setelah imputasi)
            sns.histplot(transformed_data[col], kde=True, ax=axes[1], color='lightgreen')
            axes[1].set_title(f"Sesudah (Distribusi Setelah Imputasi)")
            axes[1].axvline(transformed_data[col].mean(), color='r', linestyle='--', label='Mean')
            axes[1].axvline(transformed_data[col].median(), color='g', linestyle='-', label='Median')
            axes[1].legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()