import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from typing import Dict, Any, Optional, List

class NoventisImputer:
    """
    Menangani nilai yang hilang (null/NaN) secara cerdas dengan metode class-based.
    
    Fitur:
    - Deteksi otomatis tipe kolom (numerik & kategorikal).
    - Metode 'knn' diterapkan secara efisien untuk semua kolom numerik sekaligus.
    - Menangani kolom integer secara khusus dengan membulatkan nilai mean.
    """

    def __init__(self, method: Optional[str] = None, n_neighbors: int = 5):
        """
        Inisialisasi NoventisImputer.

        Args:
            method (str, dict, or None): 
                - str: 'mean', 'median', 'mode', 'knn', 'drop' untuk semua kolom.
                - dict: Metode spesifik per kolom, misal: {'Usia': 'median'}.
                - None: Mode otomatis ('mean' untuk numerik, 'mode' untuk kategorikal).
            n_neighbors (int): Jumlah tetangga untuk metode 'knn'.
        """
        self.method = method
        self.n_neighbors = n_neighbors
        
        self.is_fitted_ = False
        self.imputers_: Dict[str, Any] = {}
        self.columns_to_process_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> 'NoventisImputer':
        """
        Mempelajari strategi imputasi dari data training X.

        Args:
            X (pd.DataFrame): DataFrame input untuk dipelajari.

        Returns:
            self: instance yang sudah di-fit.
        """
        df = X.copy()
        self.imputers_ = {}
        self.columns_to_process_ = [col for col in df.columns if df[col].isnull().sum() > 0]

        # Penanganan khusus dan efisien untuk KNN
        if self.method == 'knn':
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df.select_dtypes(include='object').columns.tolist()
            
            if num_cols:
                knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
                knn_imputer.fit(df[num_cols])
                self.imputers_['knn'] = {'imputer': knn_imputer, 'cols': num_cols}

            for col in cat_cols:
                if col in self.columns_to_process_:
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
                
                if chosen_method == 'drop':
                    # 'drop' tidak memerlukan fitting, akan ditangani di transform
                    self.imputers_[col] = 'drop'
                    continue

                strategy = ''
                fill_val = None
                
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                if chosen_method == 'mean' and is_numeric:
                    is_int = np.all(df[col].dropna() % 1 == 0)
                    if is_int:
                        strategy = 'constant'
                        fill_val = round(df[col].mean())
                    else:
                        strategy = 'mean'
                elif chosen_method == 'median' and is_numeric:
                    strategy = 'median'
                else: # mode untuk kategorikal atau fallback
                    strategy = 'most_frequent'
                
                imp = SimpleImputer(strategy=strategy, fill_value=fill_val)
                self.imputers_[col] = imp.fit(df[[col]])
        
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Menerapkan imputasi nilai hilang ke DataFrame.

        Args:
            X (pd.DataFrame): DataFrame yang akan ditransformasi.

        Returns:
            pd.DataFrame: DataFrame dengan nilai hilang yang sudah diisi.
        """
        if not self.is_fitted_:
            raise RuntimeError("Imputer harus di-fit terlebih dahulu sebelum transform.")
            
        df_out = X.copy()
        
        if 'knn' in self.imputers_:
            knn_info = self.imputers_['knn']
            df_out[knn_info['cols']] = knn_info['imputer'].transform(df_out[knn_info['cols']])
        
        for col, imputer in self.imputers_.items():
            if col == 'knn': continue
            
            if imputer == 'drop':
                df_out.dropna(subset=[col], inplace=True)
            else:
                if col in df_out.columns:
                     df_out[[col]] = imputer.transform(df_out[[col]])
        
        return df_out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Melakukan fit dan transform dalam satu langkah.
        """
        return self.fit(X, y).transform(X)