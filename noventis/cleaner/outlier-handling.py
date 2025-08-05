import pandas as pd
import numpy as np
from scipy.stats import skew

def handle_outliers_revised(
    df: pd.DataFrame,
    feature_method_map: dict[str, str] = None,
    default_method: str = None, # 'auto', 'trim', 'winsorize', 'iqr', atau 'none'
    iqr_multiplier: float = 1.5,
    quantile_range: tuple[float, float] = (0.05, 0.95),
    min_data_threshold: int = 100,
    skew_threshold: float = 1.0
) -> pd.DataFrame:
    """
    Menangani outlier pada kolom numerik DataFrame dengan cerdas dan aman.

    Metode 'trim' dan 'iqr' mengidentifikasi semua baris outlier dari semua
    kolom terlebih dahulu, lalu menghapusnya dalam satu operasi untuk
    menjamin konsistensi hasil.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame input yang akan diproses.
    feature_method_map : dict[str, str], optional
        Peta untuk menerapkan metode spesifik per kolom, misal: {'Gaji': 'winsorize'}.
        Metode yang tersedia: 'trim', 'winsorize', 'iqr', 'auto', 'none'.
    default_method : str, optional
        Metode fallback jika tidak ada di peta. Jika None, akan disetel ke 'auto'.
    iqr_multiplier : float, default 1.5
        Pengali untuk metode 'iqr' dalam menentukan batas outlier.
    quantile_range : tuple[float, float], default (0.05, 0.95)
        Batas kuantil bawah dan atas untuk metode 'trim' dan 'winsorize'.
    min_data_threshold : int, default 100
        Batas jumlah data untuk penentuan metode otomatis. Di bawah ini, 'iqr' digunakan.
    skew_threshold : float, default 1.0
        Batas absolut skewness untuk penentuan metode otomatis.

    Returns:
    --------
    pd.DataFrame
        DataFrame baru yang telah ditangani outlier-nya.
    """
    df_out = df.copy()
    feature_method_map = feature_method_map or {}
    default_method = default_method or 'auto'
    
    indices_to_drop = set()

    # Fungsi helper untuk memilih metode otomatis
    def _choose_auto_method(col_data: pd.Series) -> str:
        if len(col_data.dropna()) < min_data_threshold:
            return 'iqr'
        elif abs(skew(col_data.dropna())) > skew_threshold:
            return 'winsorize'
        else:
            return 'trim'

    # Loop untuk menerapkan Winsorize dan mengumpulkan indeks untuk dihapus
    for col in df_out.select_dtypes(include=np.number).columns:
        method = feature_method_map.get(col, default_method)
        if method == 'auto':
            method = _choose_auto_method(df_out[col])
        
        if method == 'none':
            continue

        # Hitung batas berdasarkan metode
        lower_bound, upper_bound = None, None
        if method in ['trim', 'winsorize']:
            q_low, q_high = df_out[col].quantile(quantile_range)
            lower_bound, upper_bound = q_low, q_high
        elif method == 'iqr':
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR

        # Lakukan aksi
        if method == 'winsorize':
            df_out[col] = np.clip(df_out[col], lower_bound, upper_bound)
        elif method in ['trim', 'iqr']:
            # Kumpulkan indeks baris yang merupakan outlier
            outlier_indices = df_out.index[(df_out[col] < lower_bound) | (df_out[col] > upper_bound)]
            indices_to_drop.update(outlier_indices)
        elif method != 'none':
            raise ValueError(f"Metode '{method}' untuk kolom '{col}' tidak dikenali.")

    # Hapus semua baris outlier yang terkumpul dalam satu langkah
    if indices_to_drop:
        df_out.drop(index=list(indices_to_drop), inplace=True)
        
    return df_out