import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
from IPython.display import HTML
import io
import base64
import matplotlib.pyplot as plt
import uuid

def plot_to_base64(fig):
    """Converts a Matplotlib figure to a Base64 string for HTML embedding."""
    if fig is None:
        return ""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

class NoventisDataCleaner:
    """
    A wrapper (orchestrator) class to run a data cleaning pipeline
    consisting of Imputation, Outlier Handling, Encoding, and Scaling.
    """
    def __init__(self,
                 pipeline_steps: list = ['impute', 'outlier', 'encode', 'scale'],
                 imputer_params: dict = None,
                 outlier_params: dict = None,
                 encoder_params: dict = None,
                 scaler_params: dict = None,
                 verbose: bool = False):
        """Initializes the NoventisDataCleaner."""
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
        self._processed_df = None
        self.report_id = f"noventis-report-{uuid.uuid4().hex[:8]}"

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Executes the entire fit and transform pipeline on the data."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input 'X' must be a pandas DataFrame.")

        self._original_df = X.copy()
        df_processed = X.copy()

        if self.verbose:
            print("STARTING NOVENTIS DATA CLEANER PIPELINE")

        for step in self.pipeline_steps:
            if self.verbose:
                print(f"\nExecuting Step: {step.upper()}...")

            if step == 'impute':
                from .imputing import NoventisImputer
                self.imputer_ = NoventisImputer(**self.imputer_params)
                df_processed = self.imputer_.fit_transform(df_processed)
                self.reports_['impute'] = self.imputer_.get_quality_report()
                if y is not None:
                    y = y.loc[df_processed.index]

            elif step == 'outlier':
                from .outlier_handling import NoventisOutlierHandler
                self.outlier_handler_ = NoventisOutlierHandler(**self.outlier_params)
                df_processed = self.outlier_handler_.fit_transform(df_processed)
                self.reports_['outlier'] = self.outlier_handler_.get_quality_report()
                if y is not None:
                    y = y.loc[df_processed.index]

            elif step == 'encode':
                from .encoding import NoventisEncoder
                self.encoder_ = NoventisEncoder(**self.encoder_params)
                df_processed = self.encoder_.fit_transform(df_processed.copy(), y)
                self.reports_['encode'] = self.encoder_.get_quality_report()

            elif step == 'scale':
                from .scaling import NoventisScaler
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
        Calculates the overall data quality score with improved methodology.
        Focuses on data readiness for ML rather than penalizing legitimate transformations.
        """
        from .data_quality import assess_data_quality
        
        scores = {}
        initial_quality = assess_data_quality(self._original_df)
        final_quality = assess_data_quality(self._processed_df)

        # 1. Completeness Score (40%) - Higher is better
        scores['completeness'] = float(final_quality['completeness']['score'].replace('%',''))
        
        # 2. Consistency Score (30%) - Based on outlier removal effectiveness
        scores['consistency'] = float(final_quality['outlier_quality']['score'].replace('%',''))
        
        # 3. Distribution Quality (20%) - Based on skewness improvement
        initial_dist = float(initial_quality['distribution_quality']['score'].replace('/100',''))
        final_dist = float(final_quality['distribution_quality']['score'].replace('/100',''))
        # Reward improvement in distribution
        dist_improvement = final_dist - initial_dist
        scores['distribution'] = min(100, max(0, final_dist + (dist_improvement * 0.5)))
        
        
        # 4. Feature Engineering Quality (10%) - Reward useful encoding
        if 'encode' in self.reports_ and self.reports_['encode']:
            report = self.reports_['encode'].get('overall_summary', {})
            cols_encoded = report.get('total_columns_encoded', 0)
            features_created = report.get('total_features_created', 0)
            
            # Calculate encoding efficiency
            if cols_encoded > 0:
                # Average features per encoded column
                avg_expansion = features_created / cols_encoded
                # Reward moderate expansion (2-5 features per column is ideal)
                if avg_expansion <= 5:
                    encoding_score = 100
                elif avg_expansion <= 10:
                    encoding_score = 90
                elif avg_expansion <= 20:
                    encoding_score = 80
                else:
                    encoding_score = max(60, 100 - (avg_expansion - 20) * 2)
            else:
                encoding_score = 100  # No encoding needed
            
            scores['feature_engineering'] = encoding_score
        else:
            scores['feature_engineering'] = 100.0

        # Weights for final score calculation
        weights = {
            'completeness': 0.40,
            'consistency': 0.30, 
            'distribution': 0.20,
            'feature_engineering': 0.10
        }

        final_score = sum(scores[key] * weights[key] for key in scores)

        self.quality_score_ = {
            'final_score': f"{final_score:.2f}/100",
            'final_score_numeric': final_score,
            'details': {
                'Completeness Score': f"{scores['completeness']:.2f}",
                'Data Consistency Score': f"{scores['consistency']:.2f}",
                'Distribution Quality Score': f"{scores['distribution']:.2f}",
                'Feature Engineering Score': f"{scores['feature_engineering']:.2f}"
            },
            'weights': weights
        }

    def display_summary_report(self):
        """Displays the comprehensive final summary report in the console."""
        if not self.is_fitted_:
            print("Cleaner has not been run. Execute .fit_transform() first.")
            return

        print("\n" + "="*22 + " DATA QUALITY REPORT " + "="*22)
        print(f"  Final Quality Score: {self.quality_score_['final_score']}")
        for name, score in self.quality_score_['details'].items():
            weight_key = name.split(' ')[0].lower()
            if weight_key == 'feature':
                weight_key = 'feature_engineering'
            weight = self.quality_score_['weights'].get(weight_key, 0) * 100
            print(f"     - {name:<35}: {score:<10} (Weight: {weight:.0f}%)")

        print("\n" + "PIPELINE PROCESS SUMMARY")
        if 'impute' in self.reports_ and self.reports_['impute']:
            imputed_count = self.reports_['impute'].get('overall_summary', {}).get('total_values_imputed', 0)
            print(f"  - Imputation: Successfully filled {imputed_count} missing values.")
        if 'outlier' in self.reports_ and self.reports_['outlier']:
            removed_count = self.reports_['outlier'].get('outliers_removed', 0)
            print(self.reports_['outlier']['outliers_removed'])
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
            return f'<h3>{title}</h3><p class="plot-desc">{description}</p><div class="plot-container"><img src="{base64_str}"></div>'
        return f"<h3>{title}</h3><p>Visualization was not generated for this step.</p>"

    def generate_html_report(self) -> HTML:
        """Generates and displays a complete, visually appealing, and interactive HTML report."""
        if not self.is_fitted_ or self._original_df is None:
            return HTML("<h3>Report cannot be generated.</h3><p>Please run the `.fit_transform()` method first.</p>")
    
        # Get quality score with color
        score_numeric = self.quality_score_.get('final_score_numeric', 0)
        if score_numeric >= 90:
            score_color = '#3FB950'  # Green
            score_label = 'Excellent'
        elif score_numeric >= 80:
            score_color = '#58A6FF'  # Blue
            score_label = 'Very Good'
        elif score_numeric >= 70:
            score_color = '#D29922'  # Yellow
            score_label = 'Good'
        else:
            score_color = '#F78166'  # Orange
            score_label = 'Needs Improvement'

        # Overview Tab Content
        overview_html = f"""
            <div class="stats-grid">
                <div class="stat-card score-highlight">
                    <div class="stat-icon">📊</div>
                    <h4>Final Quality Score</h4>
                    <div class="score-large" style="color: {score_color};">{self.quality_score_['final_score']}</div>
                    <div class="score-label" style="color: {score_color};">{score_label}</div>
                    <div class="score-breakdown">
                        <div class="score-item">
                            <span class="label">Completeness</span>
                            <span class="value">{self.quality_score_['details']['Completeness Score']}</span>
                            <span class="weight">40%</span>
                        </div>
                        <div class="score-item">
                            <span class="label">Consistency</span>
                            <span class="value">{self.quality_score_['details']['Data Consistency Score']}</span>
                            <span class="weight">30%</span>
                        </div>
                        <div class="score-item">
                            <span class="label">Distribution</span>
                            <span class="value">{self.quality_score_['details']['Distribution Quality Score']}</span>
                            <span class="weight">20%</span>
                        </div>
                        <div class="score-item">
                            <span class="label">Feature Eng.</span>
                            <span class="value">{self.quality_score_['details']['Feature Engineering Score']}</span>
                            <span class="weight">10%</span>
                        </div>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">📋</div>
                    <h4>Initial Data Profile</h4>
                    <div class="stat-row">
                        <span class="stat-label">Rows:</span>
                        <span class="stat-value">{self._original_df.shape[0]:,}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Columns:</span>
                        <span class="stat-value">{self._original_df.shape[1]}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Missing Cells:</span>
                        <span class="stat-value">{self._original_df.isnull().sum().sum():,}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Categorical:</span>
                        <span class="stat-value">{len(self._original_df.select_dtypes(include=['object', 'category']).columns)}</span>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">⚙️</div>
                    <h4>Processing Summary</h4>
                    <div class="stat-row">
                        <span class="stat-label">Imputed Values:</span>
                        <span class="stat-value">{self.reports_.get('impute', {}).get('overall_summary').get('total_values_imputed', 'N/A')}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Outliers Removed:</span>
                        <span class="stat-value">{self.reports_.get('outlier', {}).get('outliers_removed', 'N/A')}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Features Encoded:</span>
                        <span class="stat-value">{self.reports_.get('encode', {}).get('overall_summary', {}).get('total_columns_encoded', 'N/A')}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Features Scaled:</span>
                        <span class="stat-value">{len(self.reports_.get('scale', {}).get('column_details', {})) if self.scaler_ else 'N/A'}</span>
                    </div>
                </div>
            </div>
            <div class="data-preview">
                <h4>📄 Data Preview (First 5 Rows of Original Data)</h4>
                <div class="table-wrapper">
                    {self._original_df.head().to_html(classes='preview-table', index=False)}
                </div>
            </div>
        """

        # Imputer Tab Content
        imputer_html = "<div class='empty-state'>❌ This step was not run.</div>"
        if self.imputer_:
            plot_b64 = plot_to_base64(self.imputer_.plot_comparison(max_cols=1))
            desc = "Membandingkan distribusi data sebelum dan sesudah penanganan nilai kosong."
            plot_html = self._get_plot_html(plot_b64, "Distribution & Missingness Comparison", desc)
            summary_html = self.imputer_.get_summary_text()
            imputer_html = f'<div class="summary-grid">{summary_html}</div>{plot_html}'

        # Outlier Tab Content
        outlier_html = "<div class='empty-state'>❌ This step was not run.</div>"
        if self.outlier_handler_:
            plot_b64 = plot_to_base64(self.outlier_handler_.plot_comparison(max_cols=1))
            desc = "Visualisasi ini menunjukkan distribusi data dan boxplot sebelum dan sesudah penghapusan outlier."
            plot_html = self._get_plot_html(plot_b64, "Outlier Handling Comparison", desc)
            summary_html = self.outlier_handler_.get_summary_text()
            outlier_html = f'<div class="summary-grid">{summary_html}</div>{plot_html}'

        # Scaler Tab Content
        scaler_html = "<div class='empty-state'>❌ This step was not run.</div>"
        if self.scaler_:
            plot_b64 = plot_to_base64(self.scaler_.plot_comparison(max_cols=1))
            desc = "Membandingkan distribusi dan Q-Q plot sebelum dan sesudah scaling."
            plot_html = self._get_plot_html(plot_b64, "Feature Scaling Comparison", desc)
            summary_html = self.scaler_.get_summary_text()
            scaler_html = f'<div class="summary-grid">{summary_html}</div>{plot_html}'

        # Encoder Tab Content
        encoder_html = "<div class='empty-state'>❌ This step was not run.</div>"
        if self.encoder_:
            report = self.reports_.get('encode', {}).get('overall_summary', {})
            plot_b64 = plot_to_base64(self.encoder_.plot_comparison(max_cols=1))
            desc = "Plot 'before' menunjukkan frekuensi kategori asli. Plot 'after' menunjukkan hasilnya."
            plot_html = self._get_plot_html(plot_b64, "Categorical Encoding Comparison", desc)
            analysis_summary = self.encoder_.get_summary_text()

            encoder_html = f"""
                <div class="summary-grid">
                    <div class="summary-card">
                        <h4>Encoding Summary</h4>
                        <div class="stat-row">
                            <span class="stat-label">Columns Encoded:</span>
                            <span class="stat-value">{report.get('total_columns_encoded', 0)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">New Features:</span>
                            <span class="stat-value">{report.get('total_features_created', 0)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Dimensionality Change:</span>
                            <span class="stat-value">{report.get('dimensionality_change', '+0.0%')}</span>
                        </div>
                    </div>
                    <div class="summary-card">
                        {analysis_summary}
                    </div>
                </div>{plot_html}"""

        # Build the HTML
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Noventis Data Cleaning Report</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
                
                #{self.report_id} {{
                    --bg-dark-1: #0D1117;
                    --bg-dark-2: #161B22;
                    --bg-dark-3: #010409;
                    --border-color: #30363D;
                    --text-light: #C9D1D9;
                    --text-muted: #8B949E;
                    --primary-blue: #58A6FF;
                    --primary-orange: #F78166;
                    --success-green: #3FB950;
                    --warning-yellow: #D29922;
                    font-family: 'Inter', sans-serif;
                    background-color: var(--bg-dark-3);
                    color: var(--text-light);
                    margin: 0;
                    padding: 0;
                    line-height: 1.6;
                }}
                
                #{self.report_id} * {{
                    box-sizing: border-box;
                }}
                
                #{self.report_id} .report-wrapper {{
                    width: 100%;
                    max-width: 1400px;
                    margin: 2rem auto;
                    background-color: var(--bg-dark-1);
                    border-radius: 12px;
                    border: 1px solid var(--border-color);
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
                    overflow: hidden;
                }}
                
                #{self.report_id} .report-header {{
                    padding: 2.5rem;
                    background: linear-gradient(135deg, #1A2D40 0%, #0D1117 100%);
                    text-align: center;
                    border-bottom: 2px solid var(--border-color);
                }}
                
                #{self.report_id} .report-header h1 {{
                    font-size: 2.5rem;
                    font-weight: 800;
                    color: var(--primary-blue);
                    margin: 0 0 0.5rem 0;
                    text-shadow: 0 2px 10px rgba(88, 166, 255, 0.3);
                }}
                
                #{self.report_id} .report-header p {{
                    margin: 0;
                    color: var(--text-muted);
                    font-size: 1.1rem;
                }}
                
                #{self.report_id} .navbar {{
                    display: flex;
                    background-color: var(--bg-dark-2);
                    padding: 0;
                    border-bottom: 1px solid var(--border-color);
                    overflow-x: auto;
                }}
                
                #{self.report_id} .nav-btn {{
                    background: none;
                    border: none;
                    color: var(--text-muted);
                    padding: 1rem 2rem;
                    font-size: 1rem;
                    font-weight: 600;
                    cursor: pointer;
                    border-bottom: 3px solid transparent;
                    transition: all 0.3s ease;
                    white-space: nowrap;
                    font-family: 'Inter', sans-serif;
                }}
                
                #{self.report_id} .nav-btn:hover {{
                    color: var(--text-light);
                    background-color: rgba(88, 166, 255, 0.1);
                }}
                
                #{self.report_id} .nav-btn.active {{
                    color: var(--primary-orange);
                    border-bottom-color: var(--primary-orange);
                    background-color: rgba(247, 129, 102, 0.05);
                }}
                
                #{self.report_id} .main-content {{
                    padding: 2.5rem;
                }}
                
                #{self.report_id} .content-section {{
                    display: none;
                }}
                
                #{self.report_id} .content-section.active {{
                    display: block;
                    animation: fadeIn 0.4s ease-in;
                }}
                
                @keyframes fadeIn {{
                    from {{ opacity: 0; transform: translateY(10px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}
                
                #{self.report_id} h2 {{
                    font-size: 1.8rem;
                    color: var(--primary-orange);
                    border-bottom: 2px solid var(--border-color);
                    padding-bottom: 0.75rem;
                    margin: 0 0 2rem 0;
                    font-weight: 700;
                }}
                
                #{self.report_id} h3 {{
                    color: var(--primary-blue);
                    font-size: 1.4rem;
                    margin: 2rem 0 1rem 0;
                    font-weight: 700;
                }}
                
                #{self.report_id} h4 {{
                    margin: 0 0 1rem 0;
                    color: var(--text-light);
                    font-size: 1.1rem;
                    font-weight: 600;
                }}
                
                #{self.report_id} .plot-desc {{
                    color: var(--text-muted);
                    margin-bottom: 1.5rem;
                    line-height: 1.6;
                }}
                
                #{self.report_id} .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                    gap: 1.5rem;
                    margin-bottom: 2.5rem;
                }}
                
                #{self.report_id} .stat-card {{
                    background: linear-gradient(145deg, var(--bg-dark-2) 0%, #1a1f28 100%);
                    padding: 2rem;
                    border-radius: 12px;
                    border: 1px solid var(--border-color);
                    transition: all 0.3s ease;
                }}
                
                #{self.report_id} .stat-card:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 4px 20px rgba(88, 166, 255, 0.15);
                    border-color: var(--primary-blue);
                }}
                
                #{self.report_id} .stat-icon {{
                    font-size: 2.5rem;
                    margin-bottom: 1rem;
                }}
                
                #{self.report_id} .score-highlight {{
                    background: linear-gradient(145deg, #1a2330 0%, #0f1419 100%);
                    border: 2px solid var(--primary-orange);
                }}
                
                #{self.report_id} .score-large {{
                    font-size: 3.5rem;
                    font-weight: 800;
                    text-align: center;
                    margin: 1rem 0;
                    text-shadow: 0 2px 10px rgba(247, 129, 102, 0.4);
                }}
                
                #{self.report_id} .score-label {{
                    text-align: center;
                    font-size: 1.2rem;
                    font-weight: 600;
                    margin-bottom: 1.5rem;
                }}
                
                #{self.report_id} .score-breakdown {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 1rem;
                    margin-top: 1.5rem;
                }}
                
                #{self.report_id} .score-item {{
                    display: flex;
                    flex-direction: column;
                    gap: 0.3rem;
                    padding: 0.75rem;
                    background-color: rgba(88, 166, 255, 0.05);
                    border-radius: 8px;
                }}
                
                #{self.report_id} .score-item .label {{
                    font-size: 0.85rem;
                    color: var(--text-muted);
                    font-weight: 500;
                }}
                
                #{self.report_id} .score-item .value {{
                    font-size: 1.3rem;
                    color: var(--primary-blue);
                    font-weight: 700;
                }}
                
                #{self.report_id} .score-item .weight {{
                    font-size: 0.75rem;
                    color: var(--text-muted);
                    font-style: italic;
                }}
                
                #{self.report_id} .stat-row {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 0.75rem 0;
                    border-bottom: 1px solid var(--border-color);
                }}
                
                #{self.report_id} .stat-row:last-child {{
                    border-bottom: none;
                }}
                
                #{self.report_id} .stat-label {{
                    color: var(--text-muted);
                    font-weight: 500;
                }}
                
                #{self.report_id} .stat-value {{
                    color: var(--primary-blue);
                    font-weight: 700;
                    font-size: 1.1rem;
                }}
                
                #{self.report_id} .data-preview {{
                    margin-top: 2.5rem;
                    background-color: var(--bg-dark-2);
                    padding: 2rem;
                    border-radius: 12px;
                    border: 1px solid var(--border-color);
                }}
                
                #{self.report_id} .table-wrapper {{
                    overflow-x: auto;
                    margin-top: 1rem;
                    border-radius: 8px;
                    border: 1px solid var(--border-color);
                }}
                
                #{self.report_id} .preview-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.9rem;
                    min-width: 600px;
                }}
                
                #{self.report_id} .preview-table thead {{
                    background-color: var(--bg-dark-3);
                    position: sticky;
                    top: 0;
                    z-index: 10;
                }}
                
                #{self.report_id} .preview-table th {{
                    color: var(--primary-blue);
                    font-weight: 700;
                    padding: 1rem;
                    text-align: left;
                    border-bottom: 2px solid var(--border-color);
                    white-space: nowrap;
                }}
                
                #{self.report_id} .preview-table td {{
                    color: var(--text-muted);
                    padding: 1rem;
                    border-bottom: 1px solid var(--border-color);
                }}
                
                #{self.report_id} .preview-table tbody tr:hover {{
                    background-color: rgba(88, 166, 255, 0.05);
                }}
                
                #{self.report_id} .plot-container {{
                    background-color: var(--bg-dark-2);
                    padding: 2rem;
                    margin-top: 2rem;
                    border-radius: 12px;
                    border: 1px solid var(--border-color);
                    text-align: center;
                }}
                
                #{self.report_id} .plot-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
                }}
                
                #{self.report_id} .summary-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 1.5rem;
                    margin-bottom: 2rem;
                }}
                
                #{self.report_id} .summary-card {{
                    background-color: var(--bg-dark-2);
                    padding: 1.5rem;
                    border-radius: 12px;
                    border: 1px solid var(--border-color);
                }}
                
                #{self.report_id} .empty-state {{
                    text-align: center;
                    padding: 4rem 2rem;
                    color: var(--text-muted);
                    font-size: 1.2rem;
                    background-color: var(--bg-dark-2);
                    border-radius: 12px;
                    border: 1px dashed var(--border-color);
                }}
                
                /* Scrollbar Styling */
                #{self.report_id} .table-wrapper::-webkit-scrollbar {{
                    height: 8px;
                }}
                
                #{self.report_id} .table-wrapper::-webkit-scrollbar-track {{
                    background: var(--bg-dark-3);
                    border-radius: 4px;
                }}
                
                #{self.report_id} .table-wrapper::-webkit-scrollbar-thumb {{
                    background: var(--border-color);
                    border-radius: 4px;
                }}
                
                #{self.report_id} .table-wrapper::-webkit-scrollbar-thumb:hover {{
                    background: var(--primary-blue);
                }}
            </style>
        </head>
        <body>
            <div id="{self.report_id}">
                <div class="report-wrapper">
                    <header class="report-header">
                        <h1>🚀 Noventis Data Cleaning Report</h1>
                        <p>An automated summary of the data preparation process</p>
                    </header>
                    <nav class="navbar">
                        <button class="nav-btn active" onclick="showTab_{self.report_id}(event, 'overview')">📊 Overview</button>
                        <button class="nav-btn" onclick="showTab_{self.report_id}(event, 'imputer')">💧 Imputer</button>
                        <button class="nav-btn" onclick="showTab_{self.report_id}(event, 'outlier')">📉 Outlier</button>
                        <button class="nav-btn" onclick="showTab_{self.report_id}(event, 'scaler')">⚖️ Scaler</button>
                        <button class="nav-btn" onclick="showTab_{self.report_id}(event, 'encoder')">🔤 Encoder</button>
                    </nav>
                    <main class="main-content">
                        <section id="overview-{self.report_id}" class="content-section active">
                            <h2>Pipeline Overview & Final Score</h2>
                            {overview_html}
                        </section>
                        <section id="imputer-{self.report_id}" class="content-section">
                            <h2>Missing Value Imputation</h2>
                            {imputer_html}
                        </section>
                        <section id="outlier-{self.report_id}" class="content-section">
                            <h2>Outlier Handling</h2>
                            {outlier_html}
                        </section>
                        <section id="scaler-{self.report_id}" class="content-section">
                            <h2>Feature Scaling</h2>
                            {scaler_html}
                        </section>
                        <section id="encoder-{self.report_id}" class="content-section">
                            <h2>Categorical Encoding</h2>
                            {encoder_html}
                        </section>
                    </main>
                </div>
            </div>
            <script>
                (function() {{
                    const reportId = '{self.report_id}';
                    
                    function showTab(event, tabName) {{
                        // Hide all content sections in this report
                        const sections = document.querySelectorAll('#' + reportId + ' .content-section');
                        sections.forEach(function(section) {{
                            section.classList.remove('active');
                        }});
                        
                        // Remove active class from all buttons in this report
                        const buttons = document.querySelectorAll('#' + reportId + ' .nav-btn');
                        buttons.forEach(function(btn) {{
                            btn.classList.remove('active');
                        }});
                        
                        // Show selected content section
                        const selectedSection = document.getElementById(tabName + '-' + reportId);
                        if (selectedSection) {{
                            selectedSection.classList.add('active');
                        }}
                        
                        // Add active class to clicked button
                        if (event && event.currentTarget) {{
                            event.currentTarget.classList.add('active');
                        }}
                    }}
                    
                    // Attach event listeners to buttons
                    document.addEventListener('DOMContentLoaded', function() {{
                        const navButtons = document.querySelectorAll('#' + reportId + ' .nav-btn');
                        navButtons.forEach(function(btn, index) {{
                            btn.addEventListener('click', function(e) {{
                                const tabs = ['overview', 'imputer', 'outlier', 'scaler', 'encoder'];
                                showTab(e, tabs[index]);
                            }});
                        }});
                    }});
                    
                    // For immediate execution if DOM is already loaded
                    if (document.readyState === 'loading') {{
                        // Do nothing, DOMContentLoaded will fire
                    }} else {{
                        // DOM is already loaded, attach listeners immediately
                        setTimeout(function() {{
                            const navButtons = document.querySelectorAll('#' + reportId + ' .nav-btn');
                            navButtons.forEach(function(btn, index) {{
                                btn.addEventListener('click', function(e) {{
                                    const tabs = ['overview', 'imputer', 'outlier', 'scaler', 'encoder'];
                                    showTab(e, tabs[index]);
                                }});
                            }});
                        }}, 100);
                    }}
                }})();
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
    # Load Data
    try:
        if isinstance(data, str):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("Unsupported 'data' format. Please provide a CSV file path or a pandas DataFrame.")
    except FileNotFoundError:
        print(f"ERROR: File not found at path: {data}")
        return None if not return_instance else (None, None)

    # Separate Features and Target
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

    # Map Function Arguments to Class Parameters
    imputer_method = None if null_handling == 'auto' else null_handling
    outlier_method = 'iqr_trim' if outlier_handling == 'dropping' else outlier_handling

    imputer_params = {'method': imputer_method}
    outlier_params = {'default_method': outlier_method}
    encoder_params = {'method': encoding, 'target_column': target_column}
    scaler_params = {'method': scaling}

    # Initialize and Run the Main Cleaner
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