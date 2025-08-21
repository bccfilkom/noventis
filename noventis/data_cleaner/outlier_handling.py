import pandas as pd
import numpy as np
from scipy.stats import skew
from typing import Dict, Tuple, Optional, Set, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

class NoventisOutlierHandler:
    """
<<<<<<< HEAD
    Intelligently handles outliers in DataFrame numeric columns using
    a class-based method consistent with Scikit-Learn.

    Supports methods 'auto', 'quantile_trim', 'iqr_trim', 'winsorize', and 'none'.
    The 'quantile_trim' and 'iqr_trim' methods will first identify all outlier 
    rows from all relevant columns, then remove them in one operation during 
    the transform stage.
    """

    def __init__(self,
                feature_method_map: Optional[Dict[str, str]] = None,
                default_method: str = 'auto',
                iqr_multiplier: float = 1.5,
                quantile_range: Tuple[float, float] = (0.05, 0.95),
                min_data_threshold: int = 100,
                skew_threshold: float = 1.0,
                verbose: bool = True):
=======
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
>>>>>>> 86d65dfcddb2792662a9912630fee13880075784
        """
        Initialize NoventisOutlierHandler.

        Args:
<<<<<<< HEAD
            feature_method_map (dict, optional): Map for specific method per column.
            default_method (str): Fallback method ('auto', 'quantile_trim', 'iqr_trim', 'winsorize', 'none'). 
            iqr_multiplier (float): Multiplier for 'iqr_trim' method. 
            quantile_range (tuple): Quantile bounds for 'quantile_trim' and 'winsorize'. 
            min_data_threshold (int): Data threshold for 'auto' method to choose 'iqr_trim'. 
            skew_threshold (float): Skewness threshold for 'auto' method.
=======
            feature_method_map (dict, optional): Peta untuk metode spesifik per kolom.
            default_method (str): Metode fallback ('auto', 'quantile_trim', 'iqr_trim', 'winsorize', 'none').
            iqr_multiplier (float): Pengali untuk metode 'iqr_trim'.
            quantile_range (tuple): Batas kuantil untuk 'quantile_trim' dan 'winsorize'.
            verbose (bool, optional): Jika True, cetak ringkasan setelah fit. Defaults to True.
>>>>>>> 86d65dfcddb2792662a9912630fee13880075784
        """
        self.feature_method_map = feature_method_map or {}
        self.default_method = default_method or 'auto'
        self.iqr_multiplier = iqr_multiplier
        self.quantile_range = quantile_range
        self.min_data_threshold = min_data_threshold
        self.skew_threshold = skew_threshold
        self.verbose = verbose
<<<<<<< HEAD
        
=======

        # --- Atribut Internal ---
>>>>>>> 86d65dfcddb2792662a9912630fee13880075784
        self.is_fitted_ = False
        self.boundaries_: Dict[str, Tuple[float, float]] = {}
        self.methods_: Dict[str, str] = {}
        self.reasons_: Dict[str, str] = {}
        self.indices_to_drop_: Set[int] = set()
<<<<<<< HEAD

        # New attributes for reporting
        self.quality_report_: Dict[str, Any] = {}
        self._original_df_snapshot: Optional[pd.DataFrame] = None

    def _choose_auto_method(self, col_data: pd.Series) -> str:
        """Helper function to choose automatic method."""
=======
        
        # --- Atribut Laporan & Visualisasi ---
        self.quality_report_: Dict[str, Any] = {}
        self._original_df_snapshot: Optional[pd.DataFrame] = None

    # --- Metode Inti ---

    def _choose_auto_method(self, col_data: pd.Series) -> Tuple[str, str]:
        """Fungsi helper untuk memilih metode otomatis dan alasannya."""
>>>>>>> 86d65dfcddb2792662a9912630fee13880075784
        if len(col_data.dropna()) < self.min_data_threshold:
            return 'iqr_trim', f"Data < {self.min_data_threshold} samples"
        elif abs(skew(col_data.dropna())) > self.skew_threshold:
            return 'winsorize', f"High skewness ({abs(skew(col_data.dropna())):.2f})"
        else:
            return 'quantile_trim', "Default fallback"

    def fit(self, X: pd.DataFrame, y=None) -> 'NoventisOutlierHandler':
<<<<<<< HEAD
        """
        Learn outlier boundaries from training data X.
        """
        df = X.copy()
        self._original_df_snapshot = df # <-- Save original data snapshot
        self.boundaries_ = {}
        self.methods_ = {}
        self.indices_to_drop_ = set()

        for col in df.select_dtypes(include=np.number).columns:
            # Handle edge case if column has no variation
            if df[col].nunique() <= 1: 
                continue               
=======
        self._original_df_snapshot = X.copy()
        self.boundaries_, self.methods_, self.reasons_, self.indices_to_drop_ = {}, {}, {}, set()

        for col in X.select_dtypes(include=np.number).columns:
            if X[col].nunique() <= 1: continue
>>>>>>> 86d65dfcddb2792662a9912630fee13880075784

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
<<<<<<< HEAD

=======
        
        # Panggil ringkasan jika verbose, tapi laporan lengkap dibuat di transform
>>>>>>> 86d65dfcddb2792662a9912630fee13880075784
        if self.verbose:
            self._print_summary(X)
            
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
<<<<<<< HEAD
        """
        Apply outlier handling to DataFrame.
        """
        if not self.is_fitted_:
            raise RuntimeError("Handler must be fitted before transform.")
=======
        if not self.is_fitted_:
            raise RuntimeError("Handler harus di-fit terlebih dahulu.")
>>>>>>> 86d65dfcddb2792662a9912630fee13880075784
        
        df_out = X.copy()
        for col, method in self.methods_.items():
<<<<<<< HEAD
            # Robustness check if column doesn't exist in transform data
            if col not in df_out.columns: 
                continue                  
            
=======
            if col not in df_out.columns: continue
>>>>>>> 86d65dfcddb2792662a9912630fee13880075784
            if method == 'winsorize':
                lower_bound, upper_bound = self.boundaries_[col]
                df_out[col] = np.clip(df_out[col], lower_bound, upper_bound)

        if self.indices_to_drop_:
<<<<<<< HEAD
            # Ensure only drop indices that exist in current dataframe
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
        Perform fit and transform in one step.
        """
        return self.fit(X).transform(X)
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Return detailed quality report from outlier handling process."""
        if not self.is_fitted_:
            print("Handler has not been fitted.")
=======
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
>>>>>>> 86d65dfcddb2792662a9912630fee13880075784
            return {}
        return self.quality_report_

    def _print_summary(self, X: pd.DataFrame):
<<<<<<< HEAD
        """Print an easy-to-read summary to console."""
        # Need dummy .transform() to get final results
        df_after = self.transform(X.copy())
        report = self.get_quality_report()
        summary = report.get('overall_summary', {})
        
        print("\n📋" + "="*23 + " OUTLIER HANDLING SUMMARY " + "="*23 + "📋")
        print(f"{'Method':<25} | {self.default_method.upper() if self.default_method == 'auto' else 'CUSTOM MAP'}")
        print(f"{'Total Rows Removed':<25} | {summary.get('outliers_removed (rows)', 'N/A')}")
        print(f"{'Data Retained Score':<25} | {summary.get('data_retained_score', 'N/A')}")
        print("="*72)
    
    def get_summary_text(self) -> str:
        """Generates a formatted string summary for the HTML report."""
        if not self.is_fitted_: return "<p>Outlier Handler has not been fitted.</p>"

        report = self.quality_report_
        methods_html = "".join([f"<li><b>{col}:</b> '{method.upper()}'</li>" for col, method in self.methods_.items()])

        summary_html = f"""
            <div class="grid-item">
                <h4>Outlier Summary</h4>
                <p><b>Rows Before:</b> {report.get('rows_before', 0)}</p>
                <p><b>Rows After:</b> {report.get('rows_after', 0)}</p>
                <p><b>Outlier Rows Removed:</b> {report.get('outliers_removed', 0)}</p>
            </div>
            <div class="grid-item">
                <h4>Methodology per Column</h4>
                <ul>{methods_html if methods_html else "<li>No columns handled.</li>"}</ul>
            </div>
        """
        return summary_html

    def plot_comparison(self, max_cols: int = 1):
        """Plot before/after comparison of outlier handling results."""
        if not self.is_fitted_ or self._original_df_snapshot is None: return None
        cols_to_plot = [col for col, method in self.methods_.items() if method != 'none']
        if not cols_to_plot: return None
        col_to_plot = cols_to_plot[0]

=======
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
        
>>>>>>> 86d65dfcddb2792662a9912630fee13880075784
        original_data = self._original_df_snapshot
        transformed_data = self.transform(original_data.copy())
        cols_to_plot = [col for col in self.methods_ if self.methods_[col] != 'none'][:max_cols]

        if not cols_to_plot:
            print("Tidak ada kolom yang diproses untuk divisualisasikan.")
            return

<<<<<<< HEAD
        color_before, color_after = '#58A6FF', '#F78166'
        bg_color, text_color = '#0D1117', '#C9D1D9'

        fig = plt.figure(figsize=(16, 8), facecolor=bg_color)
        gs = fig.add_gridspec(2, 2, height_ratios=(3, 1), hspace=0.05)
        fig.suptitle(f"Outlier Handling Comparison for '{col_to_plot}' (Method: {self.methods_[col_to_plot].upper()})",
                    fontsize=20, color=text_color, weight='bold')

        # --- BEFORE ---
        ax_hist_before = fig.add_subplot(gs[0, 0])
        ax_box_before = fig.add_subplot(gs[1, 0], sharex=ax_hist_before)
        sns.histplot(data=original_data, x=col_to_plot, kde=True, ax=ax_hist_before, color=color_before)
        sns.boxplot(data=original_data, x=col_to_plot, ax=ax_box_before, color=color_before)
        ax_hist_before.set_title("Before", color=text_color, fontsize=14)
        plt.setp(ax_hist_before.get_xticklabels(), visible=False) # Hide x-axis labels on hist plot
=======
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
>>>>>>> 86d65dfcddb2792662a9912630fee13880075784

        # --- AFTER ---
        ax_hist_after = fig.add_subplot(gs[0, 1])
        ax_box_after = fig.add_subplot(gs[1, 1], sharex=ax_hist_after)
        sns.histplot(data=transformed_data, x=col_to_plot, kde=True, ax=ax_hist_after, color=color_after)
        sns.boxplot(data=transformed_data, x=col_to_plot, ax=ax_box_after, color=color_after)
        ax_hist_after.set_title("After", color=text_color, fontsize=14)
        plt.setp(ax_hist_after.get_xticklabels(), visible=False)

        # Style all axes
        for ax in [ax_hist_before, ax_box_before, ax_hist_after, ax_box_after]:
            ax.set_facecolor(bg_color)
            ax.tick_params(colors=text_color, which='both')
            for spine in ax.spines.values(): spine.set_edgecolor(text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.set_xlabel('') # Remove individual x-labels
            ax.set_ylabel('') # Remove individual y-labels

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig