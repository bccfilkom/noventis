import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple

from IPython.display import HTML

from .scaling import NoventisScaler
from .encoding import NoventisEncoder
from .imputing import NoventisImputer
from .outlier_handling import NoventisOutlierHandler
from .data_quality import assess_data_quality


import io
import base64

def plot_to_base64(fig):
    """Mengonversi figure Matplotlib menjadi string Base64 untuk disisipkan di HTML."""
    if fig is None:
        return ""
    # Membuat buffer di memori
    buf = io.BytesIO()
    # Menyimpan plot ke buffer dalam format PNG
    fig.savefig(buf, format='png', bbox_inches='tight')
    # Kembali ke awal buffer
    buf.seek(0)
    # Encode buffer ke Base64 dan ubah menjadi string
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    # Tutup buffer
    buf.close()
    # Kembalikan string dengan format yang bisa dibaca tag <img> HTML
    return f"data:image/png;base64,{img_str}"

class NoventisDataCleaner:
    """
    A wrapper (orchestrator) class to run a data cleaning pipeline
    consisting of Imputation, Outlier Handling, Encoding, and Scaling.

    This class implements the logic from the PRD, including:
    - A configurable pipeline
    - A data quality scoring system
    - A final summary report
    - Before-and-after comparison visualizations
    """
    def __init__(self,
                 pipeline_steps: list = ['impute', 'outlier', 'encode', 'scale'],
                 imputer_params: dict = None,
                 outlier_params: dict = None,
                 encoder_params: dict = None,
                 scaler_params: dict = None,
                 verbose: bool = True):
        """
        Initializes the NoventisDataCleaner.

        Args:
            pipeline_steps (list): The sequence of steps to execute.
            imputer_params (dict): Parameters for NoventisImputer.
            outlier_params (dict): Parameters for NoventisOutlierHandler.
            encoder_params (dict): Parameters for NoventisEncoder.
            scaler_params (dict): Parameters for NoventisScaler.
            verbose (bool): If True, prints process logs.
        """
        self.pipeline_steps = pipeline_steps
        self.imputer_params = imputer_params or {}
        self.outlier_params = outlier_params or {}
        self.encoder_params = encoder_params or {}
        self.scaler_params = scaler_params or {}
        self.verbose = verbose

        self.imputer_ = None
        self.outlier_handler_ = None
        self.encoder_ = None
        self.scaler_ = None

        self.is_fitted_ = False
        self.reports_ = {}
        self.quality_score_ = {}
        self._original_df = None


    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Executes the entire fit and transform pipeline on the data.

        Args:
            X (pd.DataFrame): The feature DataFrame.
            y (pd.Series, optional): The target Series, required for encoder 'auto'/'target' mode.

        Returns:
            pd.DataFrame: The cleaned and processed DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input 'X' must be a pandas DataFrame.")

        self._original_df = X.copy()
        df_processed = X.copy()

        if self.verbose:
            print("STARTING NOVENTIS DATA CLEANER PIPELINE")
            print("="*50)

        # Execute each step in the pipeline
        for step in self.pipeline_steps:
            if self.verbose:
                print(f"\nExecuting Step: {step.upper()}...")

            if step == 'impute':
                self.imputer_ = NoventisImputer(**self.imputer_params)
                df_processed = self.imputer_.fit_transform(df_processed)
                self.reports_['impute'] = self.imputer_.get_quality_report()

                if y is not None:
                    y = y.loc[df_processed.index] 

            elif step == 'outlier':
                self.outlier_handler_ = NoventisOutlierHandler(**self.outlier_params)
                df_processed = self.outlier_handler_.fit_transform(df_processed)
                self.reports_['outlier'] = self.outlier_handler_.get_quality_report()
                if y is not None:
                    y = y.loc[df_processed.index] 

            elif step == 'encode':
                if 'target_column' not in self.encoder_params and y is not None:
                    self.encoder_params['target_column'] = y.name
                self.encoder_ = NoventisEncoder(**self.encoder_params)
                df_processed = self.encoder_.fit_transform(df_processed, y)
                self.reports_['encode'] = self.encoder_.get_quality_report()

            elif step == 'scale':
                self.scaler_ = NoventisScaler(**self.scaler_params)
                df_processed = self.scaler_.fit_transform(df_processed)
                self.reports_['scale'] = self.scaler_.get_quality_report()

            if self.verbose:
                print(f"Step {step.upper()} Complete.")

        self.is_fitted_ = True
        self._calculate_quality_score()

        if self.verbose:
            print("\nPIPELINE FINISHED")
            print("="*50)
            self.display_summary_report()

        return df_processed

    def _calculate_quality_score(self):
        """
        Calculates the overall data quality score based on reports from each step.
        """
        scores = {}

        # 1. Completeness Score from Imputer
        if 'impute' in self.reports_ and self.reports_['impute']:
            comp_score_str = self.reports_['impute'].get('completion_score', '0%')
            scores['completeness'] = float(comp_score_str.replace('%', ''))
        else:
            scores['completeness'] = 100.0 # Assume data is already complete

        # 2. Consistency Score from Outlier Handler
        if 'outlier' in self.reports_ and self.reports_['outlier']:
            rows_before = self.reports_['outlier'].get('rows_before', 1)
            rows_after = self.reports_['outlier'].get('rows_after', 1)
            scores['consistency'] = (rows_after / rows_before) * 100 if rows_before > 0 else 100.0
        else:
            scores['consistency'] = 100.0 # Assume no outliers were removed

        # 3. Distribution Score from Scaler
        if 'scale' in self.reports_ and self.reports_['scale']:
            skew_changes = []
            if 'column_details' in self.reports_['scale']:
                for col_detail in self.reports_['scale']['column_details'].values():
                    try:
                        # Correction: Convert to float before subtraction
                        skew_before = float(col_detail.get('skewness_before', 0))
                        skew_after = float(col_detail.get('skewness_after', 0))
                        if skew_before > 0:
                            # Measure the percentage reduction in skewness
                            change = max(0, (abs(skew_before) - abs(skew_after)) / abs(skew_before))
                            skew_changes.append(change)
                    except (ValueError, TypeError):
                        continue
                avg_skew_reduction = np.mean(skew_changes) if skew_changes else 0
                scores['distribution'] = avg_skew_reduction * 100
            else:
                scores['distribution'] = 0.0
        else:
            scores['distribution'] = 0.0 # No scaling, no distribution improvement

        # 4. Dimensionality Efficiency Score from Encoder
        if 'encode' in self.reports_ and self.reports_['encode']:
            dim_change_str = self.reports_['encode']['overall_summary'].get('dimensionality_change', '+0.0%')
            dim_change = float(dim_change_str.replace('%', '').replace('+', ''))
            # Score is higher if dimensionality does not increase significantly
            scores['dimensionality'] = max(0, 100 - dim_change)
        else:
            scores['dimensionality'] = 100.0 # No encoding, dimensionality does not change

        # Weights for each score
        weights = {
            'completeness': 0.40,
            'consistency': 0.30,
            'distribution': 0.20,
            'dimensionality': 0.10
        }

        final_score = (scores['completeness'] * weights['completeness'] +
                       scores['consistency'] * weights['consistency'] +
                       scores['distribution'] * weights['distribution'] +
                       scores['dimensionality'] * weights['dimensionality'])

        self.quality_score_ = {
            'final_score': f"{final_score:.2f}/100",
            'details': {
                'Completeness Score': f"{scores['completeness']:.2f}",
                'Data Consistency Score': f"{scores['consistency']:.2f}",
                'Distribution Improvement Score': f"{scores['distribution']:.2f}",
                'Dimensionality Efficiency Score': f"{scores['dimensionality']:.2f}"
            },
            'weights': weights
        }

    def display_summary_report(self):
        """
        Displays the comprehensive final summary report.
        """
        if not self.is_fitted_:
            print("Cleaner has not been run. Execute .fit_transform() first.")
            return

        print("\n" + "="*22 + " DATA QUALITY REPORT " + "="*22)

        # Quality Score
        print("\n" + "--- OVERALL DATA QUALITY SCORE ---")
        print(f"  Final Quality Score: {self.quality_score_['final_score']}")
        for name, score in self.quality_score_['details'].items():
            weight_key = name.split(' ')[0].lower()
            weight = self.quality_score_['weights'].get(weight_key, 0) * 100
            print(f"     - {name:<35}: {score:<10} (Weight: {weight:.0f}%)")

        # Details per Step
        print("\n" + "--- PIPELINE PROCESS SUMMARY ---")
        if 'impute' in self.reports_ and self.reports_['impute']:
            imputed_count = self.reports_['impute'].get('values_imputed', 0)
            print(f"  - Imputation: Successfully filled {imputed_count} missing values.")

        if 'outlier' in self.reports_ and self.reports_['outlier']:
            removed_count = self.reports_['outlier'].get('outliers_removed', 0)
            print(f"  - Outliers: Removed {removed_count} rows identified as outliers.")

        if 'encode' in self.reports_ and self.reports_['encode']:
            summary = self.reports_['encode'].get('overall_summary', {})
            encoded_cols = summary.get('total_columns_encoded', 0)
            new_features = summary.get('total_features_created', 0)
            print(f"  - Encoding: Transformed {encoded_cols} categorical columns into {new_features} new features.")

        if 'scale' in self.reports_ and self.reports_['scale']:
            scaled_cols = len(self.reports_['scale'].get('column_details', {}))
            print(f"  - Scaling: Standardized the scale for {scaled_cols} numerical columns.")

        # Details of Automatic Decisions
        print("\n" + "--- AUTOMATIC DECISION DETAILS ---")
        if self.scaler_ and self.scaler_.method == 'auto' and hasattr(self.scaler_, 'reasons_'):
            print("  - SCALING:")
            for col, reason in self.scaler_.reasons_.items():
                method = self.scaler_.fitted_methods_.get(col, 'N/A')
                print(f"    - Column '{col}': Used '{method.upper()}' because: {reason}.")

        if self.encoder_ and self.encoder_.method == 'auto' and hasattr(self.encoder_, 'learned_cols'):
            print("  - ENCODING:")
            for col, method in self.encoder_.learned_cols.items():
                 print(f"    - Column '{col}': Selected method '{method.upper()}'.")

        print("\n" + "="*65)

    def plot_all_comparisons(self, max_cols: int = 3):
        """
        Generates all comparison plots from each pipeline step.
        """
        if not self.is_fitted_:
            print("Cleaner has not been run. Execute .fit_transform() first.")
            return

        print("\n" + "="*22 + " COMPARISON VISUALIZATIONS " + "="*22)

        if self.imputer_ and hasattr(self.imputer_, 'plot_comparison'):
            self.imputer_.plot_comparison(max_cols=max_cols)
        if self.outlier_handler_ and hasattr(self.outlier_handler_, 'plot_comparison'):
            self.outlier_handler_.plot_comparison(max_cols=max_cols)
        if self.encoder_ and hasattr(self.encoder_, 'plot_comparison'):
            # Encoder plot needs the original data before transformation
            self.encoder_.plot_comparison(self._original_df, max_cols=max_cols)
            if self.scaler_ and hasattr(self.scaler_, 'plot_comparison'):
                self.scaler_.plot_comparison(max_cols=max_cols)


    
    def generate_html_report(self):
        """
        Menghasilkan dan menampilkan laporan visual HTML interaktif yang lengkap,
        dengan JavaScript yang andal untuk lingkungan Jupyter Notebook.
        """
        if not self.is_fitted_ or self._original_df is None:
            print("Cleaner has not been run. Execute .fit_transform() first.")
            return

        # --- Bagian pengumpulan data dan plot tetap sama ---
        overview_stats = assess_data_quality(self._original_df)
        overview_html = f"""
            <div class="overview-grid">
                <div class="stat-card"><h3>Total Baris</h3><p>{len(self._original_df)}</p></div>
                <div class="stat-card"><h3>Total Kolom</h3><p>{len(self._original_df.columns)}</p></div>
                <div class="stat-card"><h3>Kelengkapan Data</h3><p>{overview_stats['completeness']['score']}</p></div>
                <div class="stat-card"><h3>Kualitas Outlier</h3><p>{overview_stats['outlier_quality']['score']}</p></div>
                <div class="stat-card"><h3>Kualitas Distribusi</h3><p>{overview_stats['distribution_quality']['score']}</p></div>
                <div class="stat-card"><h3>Kemurnian Tipe Data</h3><p>{overview_stats['datatype_purity']['score']}</p></div>
            </div>
            <div class="df-preview">
                <h4>Pratinjau 5 Baris Data Asli:</h4>
                {self._original_df.head().to_html(classes='styled-table')}
            </div>
        """
        imputer_fig = self.imputer_.plot_comparison(max_cols=1) if self.imputer_ else None
        outlier_fig = self.outlier_handler_.plot_comparison(max_cols=1) if self.outlier_handler_ else None
        scaler_fig = self.scaler_.plot_comparison(max_cols=1) if self.scaler_ else None
        encoder_fig = self.encoder_.plot_comparison(self._original_df, max_cols=1) if self.encoder_ else None
        imputer_plot_b64 = plot_to_base64(imputer_fig)
        outlier_plot_b64 = plot_to_base64(outlier_fig)
        scaler_plot_b64 = plot_to_base64(scaler_fig)
        encoder_plot_b64 = plot_to_base64(encoder_fig)

        # --- Template HTML Final dengan Perbaikan JavaScript ---
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Noventis - Data Cleaning Report</title>
            <style>
                /* CSS tetap sama seperti sebelumnya */
                @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&family=Exo+2:wght@700&display=swap');
                :root {{
                    --base-black: #121212; --base-gray: #807F8C; --base-white: #FFFFFF;
                    --primary-blue: #0F2CAB; --primary-orange: #FF6849;
                    --secondary-pink: #CF4BC0; --secondary-purple: #896DF3; --secondary-turquoise: #31B7AE;
                    --background-dark-1: #01010A; --background-dark-2: #04021F; --background-dark-3: #050329; --background-dark-4: #0B0848;
                    --gradient: linear-gradient(90deg, var(--primary-orange), var(--primary-blue));
                }}
                body {{ font-family: 'Roboto', sans-serif; background-color: var(--background-dark-3); color: var(--base-white); margin: 0; padding: 2rem; }}
                .container {{ width: 100%; max-width: 1200px; margin: auto; background-color: var(--background-dark-2); border-radius: 15px; padding: 2rem 3rem; border: 1px solid var(--background-dark-4); box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); }}
                header h1 {{ font-family: 'Exo 2', sans-serif; font-size: 3rem; text-align: center; margin-bottom: 2rem; background: var(--gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
                .navbar {{ display: flex; flex-wrap: wrap; justify-content: center; margin-bottom: 2.5rem; background-color: var(--background-dark-4); border-radius: 10px; padding: 0.5rem; }}
                .nav-btn {{ background: none; border: none; color: var(--base-gray); padding: 0.8rem 1.5rem; margin: 0.25rem; font-size: 1rem; font-weight: 700; cursor: pointer; border-radius: 7px; transition: all 0.3s ease; }}
                .nav-btn:hover {{ background-color: rgba(255, 255, 255, 0.1); color: var(--base-white); }}
                .nav-btn.active {{ background-color: var(--primary-blue); color: var(--base-white); box-shadow: 0 0 15px rgba(15, 44, 171, 0.5); }}
                .content-section {{ display: none; }} .content-section.active {{ display: block; }}
                .plot-container {{ background-color: var(--background-dark-3); padding: 1.5rem; border-radius: 10px; border: 1px solid var(--background-dark-4); text-align: center; margin-top: 1rem; }}
                .plot-container img {{ max-width: 100%; height: auto; border-radius: 5px; background-color: white; }}
                .plot-container p {{ color: var(--base-gray); font-style: italic; }}
                h2 {{ color: var(--primary-orange); border-bottom: 2px solid var(--primary-blue); padding-bottom: 0.5rem; }}
                .overview-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; }}
                .stat-card {{ background-color: var(--background-dark-4); padding: 1.5rem; border-radius: 10px; text-align: center; border-left: 5px solid var(--primary-orange); }}
                .stat-card h3 {{ margin-top: 0; color: var(--base-white); font-size: 1.1rem; }}
                .stat-card p {{ font-size: 2rem; font-weight: 700; color: var(--primary-blue); margin-bottom: 0; }}
                .df-preview {{ margin-top: 2rem; }}
                .styled-table {{ width: 100%; color: var(--base-white); background-color: var(--background-dark-4); border-collapse: collapse; }}
                .styled-table th, .styled-table td {{ border: 1px solid var(--background-dark-3); padding: 0.8rem; text-align: left; }}
                .styled-table thead {{ background-color: var(--primary-blue); }}
            </style>
        </head>
        <body>
            <div class="container">
                <header><h1>Noventis</h1></header>

                <nav class="navbar">
                    <button class="nav-btn active" onclick="showTab(event, 'overview')">Overview</button>
                    <button class="nav-btn" onclick="showTab(event, 'imputer')">Imputer</button>
                    <button class="nav-btn" onclick="showTab(event, 'outlier')">Outlier</button>
                    <button class="nav-btn" onclick="showTab(event, 'scaler')">Scaler</button>
                    <button class="nav-btn" onclick="showTab(event, 'encoder')">Encoder</button>
                </nav>

                <main>
                    <section id="overview" class="content-section active">
                        <h2>Ringkasan Kualitas Data Awal</h2>
                        {overview_html}
                    </section>
                    <section id="imputer" class="content-section"><div class="plot-container">{self._get_plot_html(imputer_plot_b64, 'Perbandingan Imputasi')}</div></section>
                    <section id="outlier" class="content-section"><div class="plot-container">{self._get_plot_html(outlier_plot_b64, 'Perbandingan Penanganan Outlier')}</div></section>
                    <section id="scaler" class="content-section"><div class="plot-container">{self._get_plot_html(scaler_plot_b64, 'Perbandingan Scaler')}</div></section>
                    <section id="encoder" class="content-section"><div class="plot-container">{self._get_plot_html(encoder_plot_b64, 'Perbandingan Encoder')}</div></section>
                </main>
            </div>

            <script>
                function showTab(event, tabName) {{
                    // Sembunyikan semua konten
                    const contentSections = document.querySelectorAll('.content-section');
                    contentSections.forEach(section => {{
                        section.style.display = 'none';
                        section.classList.remove('active');
                    }});

                    // Non-aktifkan semua tombol
                    const navButtons = document.querySelectorAll('.nav-btn');
                    navButtons.forEach(button => {{
                        button.classList.remove('active');
                    }});

                    // Tampilkan konten yang ditargetkan
                    document.getElementById(tabName).style.display = 'block';
                    document.getElementById(tabName).classList.add('active');

                    // Aktifkan tombol yang diklik
                    event.currentTarget.classList.add('active');
                }}

                // Secara default, tampilkan tab pertama saat halaman dimuat
                document.addEventListener('DOMContentLoaded', (event) => {{
                    document.querySelector('.nav-btn').click();
                }});
            </script>
        </body>
        </html>
        """
        return HTML(html_template)

    # Jangan lupa untuk juga menyertakan metode helper ini di dalam kelas
    def _get_plot_html(self, base64_str, title):
        """Helper untuk membuat tag img atau pesan jika plot tidak ada."""
        if base64_str:
            # Menambahkan background putih pada gambar agar plot terlihat jelas
            return f'<h2>{title}</h2><img src="{base64_str}" style="background-color: #FFFFFF;">'
        else:
            return f"<h2>{title}</h2><p>Visualisasi tidak tersedia atau tidak dapat dibuat untuk langkah ini.</p>"

def data_cleaner(
    data: Union[str, pd.DataFrame],
    target_column: Optional[str] = None,
    null_handling: str = 'auto',
    outlier_handling: str = 'auto',
    encoding: str = 'auto',
    scaling: str = 'auto',
    verbose: bool = True,
    return_instance: bool = False  
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, 'NoventisDataCleaner']]:
    """
    A high-level wrapper function to run the Noventis data cleaning pipeline.
    Provides a simplified interface. By default, returns only the cleaned DataFrame.

    Args:
        data (Union[str, pd.DataFrame]): Path to a CSV file or an existing DataFrame.
        target_column (str, optional): Name of the target column. Important for some 'auto' modes.
        null_handling (str): Method for handling nulls ('auto', 'mean', 'median', 'mode', 'drop', etc.).
        outlier_handling (str): Method for handling outliers ('auto', 'iqr_trim', 'winsorize', 'dropping').
        encoding (str): Method for encoding ('auto', 'ohe', 'label', 'target', etc.).
        scaling (str): Method for scaling ('auto', 'minmax', 'standard', 'robust').
        verbose (bool): If True, displays detailed reports during the process.
        return_instance (bool): If True, returns a tuple of (DataFrame, cleaner_instance). 
                                If False (default), returns only the cleaned DataFrame.

    Returns:
        Union[pd.DataFrame, Tuple[pd.DataFrame, NoventisDataCleaner]]:
            - The cleaned DataFrame (default).
            - A tuple of (cleaned DataFrame, NoventisDataCleaner instance) if return_instance is True.
    """

    # 1. Load Data
    try:
        if isinstance(data, str):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("Unsupported 'data' format. Please provide a CSV file path or a pandas DataFrame.")
    except FileNotFoundError:
        print(f"ERROR: File not found at path: {data}")
        return None, None

    # 2. Separate Features and Target
    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        if verbose:
            print(f"Target column '{target_column}' identified.")
    else:
        X = df
        y = None
        if target_column and verbose:
            print(f"WARNING: Target column '{target_column}' not found. Proceeding without a target.")

    # 3. Map Function Arguments to Class Parameters
    imputer_method = None if null_handling == 'auto' else null_handling
    outlier_method = 'iqr_trim' if outlier_handling == 'dropping' else outlier_handling

    imputer_params = {'method': imputer_method}
    outlier_params = {'default_method': outlier_method}
    encoder_params = {'method': encoding, 'target_column': target_column}
    scaler_params = {'method': scaling}

    # 4. Initialize and Run the Main Cleaner
    cleaner_instance = NoventisDataCleaner(
        imputer_params=imputer_params,
        outlier_params=outlier_params,
        encoder_params=encoder_params,
        scaler_params=scaler_params,
        verbose=verbose
    )

    cleaned_df = cleaner_instance.fit_transform(X, y)

    # --- BAGIAN RETURN YANG DIMODIFIKASI ---
    if return_instance:
        return cleaned_df, cleaner_instance
    else:
        return cleaned_df
    

