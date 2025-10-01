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
import matplotlib.pyplot as plt

def plot_to_base64(fig):
    """Converts a Matplotlib figure to a Base64 string for HTML embedding."""
    if fig is None:
        return ""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig) # Close the figure to free up memory
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
                 verbose: bool = False):
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
        self.verbose = False

        self.imputer_ = None
        self.outlier_handler_ = None
        self.encoder_ = None
        self.scaler_ = None

        self.is_fitted_ = False
        self.reports_ = {}
        self.quality_score_ = {}
        self._original_df = None
        self._processed_df = None


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
            # print("="*50)

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
                # Pass a copy of the dataframe state *before* this step to the encoder for its internal snapshot
                self.encoder_ = NoventisEncoder(**self.encoder_params)
                df_processed = self.encoder_.fit_transform(df_processed.copy(), y)
                self.reports_['encode'] = self.encoder_.get_quality_report()


            elif step == 'scale':
                self.scaler_ = NoventisScaler(**self.scaler_params)
                df_processed = self.scaler_.fit_transform(df_processed)
                self.reports_['scale'] = self.scaler_.get_quality_report()

            if self.verbose:
                print(f"Step {step.upper()} Complete.")

        self.is_fitted_ = True
        self._processed_df = df_processed
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
        initial_quality = assess_data_quality(self._original_df)
        final_quality = assess_data_quality(self._processed_df)

        scores['completeness'] = float(final_quality['completeness']['score'].replace('%',''))
        scores['consistency'] = float(final_quality['outlier_quality']['score'].replace('%',''))
        scores['distribution'] = float(final_quality['distribution_quality']['score'].replace('/100',''))
        
        # Dimensionality score
        if 'encode' in self.reports_ and self.reports_['encode']:
            dim_change_str = self.reports_['encode']['overall_summary'].get('dimensionality_change', '+0.0%')
            dim_change = float(dim_change_str.replace('%', '').replace('+', ''))
            scores['dimensionality'] = max(0, 100 - max(0, dim_change * 2)) # Penalize increase more
        else:
            scores['dimensionality'] = 100.0

        weights = {'completeness': 0.40, 'consistency': 0.30, 'distribution': 0.20, 'dimensionality': 0.10}

        final_score = sum(scores[key] * weights[key] for key in scores)

        self.quality_score_ = {
            'final_score': f"{final_score:.2f}/100",
            'details': {
                'Completeness Score': f"{scores['completeness']:.2f}",
                'Data Consistency Score': f"{scores['consistency']:.2f}",
                'Distribution Quality Score': f"{scores['distribution']:.2f}",
                'Dimensionality Efficiency Score': f"{scores['dimensionality']:.2f}"
            },
            'weights': weights
        }


    def display_summary_report(self):
        """
        Displays the comprehensive final summary report in the console.
        """
        if not self.is_fitted_:
            print("Cleaner has not been run. Execute .fit_transform() first.")
            return

        print("\n" + "="*22 + " DATA QUALITY REPORT " + "="*22)
        print(f"  Final Quality Score: {self.quality_score_['final_score']}")
        for name, score in self.quality_score_['details'].items():
            weight_key = name.split(' ')[0].lower()
            weight = self.quality_score_['weights'].get(weight_key, 0) * 100
            print(f"     - {name:<35}: {score:<10} (Weight: {weight:.0f}%)")

        print("\n" + "PIPELINE PROCESS SUMMARY")
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

        print("\n" + "="*65)

    def _get_plot_html(self, base64_str: str, title: str, description: str) -> str:
        """Helper to generate HTML for a plot or a fallback message."""
        if base64_str:
            return f'<h3>{title}</h3><p>{description}</p><div class="plot-container"><img src="{base64_str}"></div>'
        return f"<h3>{title}</h3><p>Visualization was not generated for this step.</p>"

    def generate_html_report(self) -> HTML:
        """
        Generates and displays a complete, visually appealing, and interactive HTML report.
        """
        if not self.is_fitted_ or self._original_df is None:
            return HTML("<h3>Report cannot be generated.</h3><p>Please run the `.fit_transform()` method first.</p>")
    
        # --- Overview Tab Content (Tidak berubah) ---
        imputed_values_report = self.reports_.get('impute', {}).get('overall_summary', {})
        outlier_report = self.reports_.get('outlier', {})
        encoder_report = self.reports_.get('encode', {}).get('overall_summary', {})
        scaler_report = self.reports_.get('scale', {})

        # --- Overview Tab Content ---
        overview_html = f"""
            <div class="grid-container">
                <div class="grid-item score-card">
                    <h4>Final Quality Score</h4>
                    <p class="score">{self.quality_score_['final_score']}</p>
                    <div class="score-details">
                        <span>Completeness: {self.quality_score_['details']['Completeness Score']}</span>
                        <span>Consistency: {self.quality_score_['details']['Data Consistency Score']}</span>
                        <span>Distribution: {self.quality_score_['details']['Distribution Quality Score']}</span>
                        <span>Dimensionality: {self.quality_score_['details']['Dimensionality Efficiency Score']}</span>
                    </div>
                </div>
                <div class="grid-item">
                    <h4>Initial Data Profile</h4>
                    <p><b>Rows:</b> {self._original_df.shape[0]}</p>
                    <p><b>Columns:</b> {self._original_df.shape[1]}</p>
                    <p><b>Missing Cells:</b> {self._original_df.isnull().sum().sum()}</p>
                    <p><b>Categorical Columns:</b> {len(self._original_df.select_dtypes(include=['object', 'category']).columns)}</p>
                </div>
                <div class="grid-item">
                    <h4>Processing Summary</h4>
                    <p><b>Imputed Values:</b> {self.reports_.get('impute', {}).get('values_imputed', 'N/A')}</p>
                    <p><b>Outlier Rows Removed:</b> {self.reports_.get('outlier', {}).get('outliers_removed', 'N/A')}</p>
                    <p><b>Categorical Features Encoded:</b> {self.reports_.get('encode', {}).get('overall_summary', {}).get('total_columns_encoded', 'N/A')}</p>
                    <p><b>Numeric Features Scaled:</b> {len(self.reports_.get('scale', {}).get('column_details', {})) if self.scaler_ else 'N/A'}</p>
                </div>
            </div>
            <div class="df-preview">
                <h4>Data Preview (First 5 Rows of Original Data)</h4>
                {self._original_df.head().to_html(classes='styled-table')}
            </div>
        """

        imputer_html = "<h4>This step was not run.</h4>"
        if self.imputer_:
            # --- PERUBAHAN ---
            # Mengambil plot dan ringkasan dari kelasnya masing-masing
            plot_b64 = plot_to_base64(self.imputer_.plot_comparison(max_cols=1))
            desc = "Membandingkan distribusi data sebelum dan sesudah penanganan nilai kosong."
            plot_html = self._get_plot_html(plot_b64, "Distribution & Missingness Comparison", desc)
            summary_html = self.imputer_.get_summary_text()
            
            # Menggabungkan ringkasan dan plot
            imputer_html = f'<div class="grid-container">{summary_html}</div>{plot_html}'

        # --- Outlier Tab Content ---
        outlier_html = "<h4>This step was not run.</h4>"
        if self.outlier_handler_:
            # --- PERUBAHAN ---
            # Pola yang sama diterapkan di sini
            plot_b64 = plot_to_base64(self.outlier_handler_.plot_comparison(max_cols=1))
            desc = "Visualisasi ini menunjukkan distribusi data dan boxplot sebelum dan sesudah penghapusan outlier."
            plot_html = self._get_plot_html(plot_b64, "Outlier Handling Comparison", desc)
            summary_html = self.outlier_handler_.get_summary_text()
            
            outlier_html = f'<div class="grid-container">{summary_html}</div>{plot_html}'

        # --- Scaler Tab Content ---
        scaler_html = "<h4>This step was not run.</h4>"
        if self.scaler_:
            # --- PERUBAHAN ---
            # Dan juga di sini untuk konsistensi
            plot_b64 = plot_to_base64(self.scaler_.plot_comparison(max_cols=1))
            desc = "Membandingkan distribusi dan Q-Q plot sebelum dan sesudah scaling."
            plot_html = self._get_plot_html(plot_b64, "Feature Scaling Comparison", desc)
            summary_html = self.scaler_.get_summary_text()
            
            scaler_html = f'<div class="grid-container">{summary_html}</div>{plot_html}'

        # --- Encoder Tab Content ---
        encoder_html = "<h4>This step was not run.</h4>"
        if self.encoder_:
            # Bagian ini sudah menggunakan pola yang benar
            report = self.reports_.get('encode', {}).get('overall_summary', {})
            plot_b64 = plot_to_base64(self.encoder_.plot_comparison(max_cols=1))
            desc = "Plot 'before' menunjukkan frekuensi kategori asli. Plot 'after' menunjukkan hasilnya."
            plot_html = self._get_plot_html(plot_b64, "Categorical Encoding Comparison", desc)
            
            analysis_summary = self.encoder_.get_summary_text()

            encoder_html = f"""
                 <div class="grid-container">
                    <div class="grid-item">
                        <h4>Encoding Summary</h4>
                        <p><b>Columns Encoded:</b> {report.get('total_columns_encoded', 0)}</p>
                        <p><b>New Features Created:</b> {report.get('total_features_created', 0)}</p>
                        <p><b>Dimensionality Change:</b> {report.get('dimensionality_change', '+0.0%')}</p>
                    </div>
                    <div class="grid-item">
                        {analysis_summary}
                    </div>
                </div>{plot_html}"""
                    # --- 2. BUILD THE HTML STRING ---
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Noventis Data Cleaning Report</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Exo+2:wght@600;800&display=swap');
                :root {{
                    --bg-dark-1: #0D1117; --bg-dark-2: #161B22; --bg-dark-3: #010409;
                    --border-color: #30363D; --text-light: #C9D1D9; --text-muted: #8B949E;
                    --primary-blue: #58A6FF; --primary-orange: #F78166;
                    --font-main: 'Roboto', sans-serif; --font-header: 'Exo 2', sans-serif;
                }}
                body {{ font-family: var(--font-main); background-color: var(--bg-dark-3); color: var(--text-light); margin: 0; padding: 1.5rem; }}
                .container {{ width: 100%; max-width: 1400px; margin: auto; background-color: var(--bg-dark-1); border-radius: 10px; border: 1px solid var(--border-color); }}
                header {{ padding: 1.5rem 2.5rem; border-bottom: 1px solid var(--border-color); background-color: var(--bg-dark-2); border-radius: 10px 10px 0 0; }}
                header h1 {{ font-family: var(--font-header); font-size: 2.5rem; margin: 0; color: var(--primary-blue); }}
                header p {{ margin: 0.25rem 0 0; color: var(--text-muted); font-size: 1rem; }}
                .navbar {{ display: flex; background-color: var(--bg-dark-2); padding: 0 2.5rem; border-bottom: 1px solid var(--border-color); }}
                .nav-btn {{ background: none; border: none; color: var(--text-muted); padding: 1rem 1.5rem; font-size: 1rem; cursor: pointer; border-bottom: 3px solid transparent; transition: all 0.2s ease-in-out; }}
                .nav-btn:hover {{ color: var(--text-light); }}
                .nav-btn.active {{ color: var(--primary-orange); border-bottom-color: var(--primary-orange); font-weight: 700; }}
                main {{ padding: 2.5rem; }}
                .content-section {{ display: none; }} .content-section.active {{ display: block; animation: fadeIn 0.5s; }}
                @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
                h2 {{ font-family: var(--font-header); font-size: 2rem; color: var(--primary-orange); border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem; margin-top: 0; }}
                h3 {{ font-family: var(--font-header); color: var(--primary-blue); font-size: 1.5rem; margin-top: 2rem; }}
                p {{ line-height: 1.6; }}
                .grid-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
                .grid-item {{ background-color: var(--bg-dark-2); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--border-color); }}
                .grid-item h4 {{ margin-top: 0; color: var(--text-light); border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem; }}
                .grid-item p, .grid-item li {{ color: var(--text-muted); }}
                .grid-item ul {{ padding-left: 20px; margin: 0; }}
                .score-card .score {{ font-size: 4rem; font-weight: 800; color: var(--primary-orange); margin: 1rem 0; text-align: center; }}
                .score-card .score-details {{ display: flex; justify-content: space-around; font-size: 0.9rem; text-align: center; color: var(--text-muted); flex-wrap: wrap; }}
                .df-preview {{ margin-top: 2rem; }}
                .styled-table {{ width: 100%; color: var(--text-muted); background-color: var(--bg-dark-2); border-collapse: collapse; border-radius: 8px; overflow: hidden; }}
                .styled-table th, .styled-table td {{ border-bottom: 1px solid var(--border-color); padding: 0.8rem 1rem; text-align: left; }}
                .styled-table thead {{ background-color: var(--bg-dark-3); color: var(--text-light); }}
                .plot-container {{ background-color: var(--bg-dark-2); padding: 1rem; margin-top: 1rem; border-radius: 8px; border: 1px solid var(--border-color); text-align: center; }}
                .plot-container img {{ max-width: 80%; height: auto; border-radius: 5px; background-color: #FFFFFF; }}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>Noventis Data Cleaning Report</h1>
                    <p>An automated summary of the data preparation process.</p>
                </header>
                <nav class="navbar">
                    <button class="nav-btn active" onclick="showTab(event, 'overview')">📊 Overview</button>
                    <button class="nav-btn" onclick="showTab(event, 'imputer')">💧 Imputer</button>
                    <button class="nav-btn" onclick="showTab(event, 'outlier')">📈 Outlier</button>
                    <button class="nav-btn" onclick="showTab(event, 'scaler')">⚖️ Scaler</button>
                    <button class="nav-btn" onclick="showTab(event, 'encoder')">🔠 Encoder</button>
                </nav>
                <main>
                    <section id="overview" class="content-section active">
                        <h2>Pipeline Overview & Final Score</h2>
                        {overview_html}
                    </section>
                    <section id="imputer" class="content-section">
                        <h2>Missing Value Imputation</h2>
                        {imputer_html}
                    </section>
                    <section id="outlier" class="content-section">
                        <h2>Outlier Handling</h2>
                        {outlier_html}
                    </section>
                    <section id="scaler" class="content-section">
                        <h2>Feature Scaling</h2>
                        {scaler_html}
                    </section>
                    <section id="encoder" class="content-section">
                        <h2>Categorical Encoding</h2>
                        {encoder_html}
                    </section>
                </main>
            </div>
            <script>
                function showTab(event, tabName) {{
                    document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
                    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
                    document.getElementById(tabName).classList.add('active');
                    event.currentTarget.classList.add('active');
                }}
                // Set the first tab as active on load
                document.addEventListener('DOMContentLoaded', () => {{
                    document.querySelector('.nav-btn').click();
                }});
            </script>
        </body>
        </html>
        """
        return HTML(html_template)
    

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

    if return_instance:
        return cleaned_df, cleaner_instance
    else:
        return cleaned_df
    

