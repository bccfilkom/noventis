import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from typing import Dict, Any, Optional, List

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

    def fit(self, X: pd.DataFrame, y=None) -> 'NoventisImputer':
        """
        Mempelajari strategi imputasi dari data training X.
        """
        df = X.copy()
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
        
        return df_out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Melakukan fit dan transform dalam satu langkah.
        """
        return self.fit(X, y).transform(X)