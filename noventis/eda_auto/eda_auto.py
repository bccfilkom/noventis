import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import io
import base64
import tempfile
from pathlib import Path
import uuid

# --- Helper Function ---
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

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.titlesize': 12, 'figure.titlesize': 14, 'legend.fontsize': 10,
    'figure.facecolor': '#010409', 'axes.facecolor': '#161B22', 'text.color': '#C9D1D9',
    'axes.labelcolor': '#C9D1D9', 'xtick.color': '#8B949E', 'ytick.color': '#8B949E',
    'grid.color': '#30363D', 'patch.edgecolor': '#30363D', 'figure.edgecolor': '#010409',
})


class EDAAnalyzer:
    """
    A class to perform automated Exploratory Data Analysis (EDA).
    It generates statistics, visualizations, and an interactive HTML report.
    """
    def __init__(self, df: pd.DataFrame, target: str = None, personality: str = 'default'):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
        
        self.df = df.copy()
        self.target = target
        if self.target and self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' not found in the DataFrame.")
            
        self.personality = personality
        self._report_content = {} 

        self.numeric_cols_ = self.df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols_ = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.CORRELATION_COL_THRESHOLD = 30

    def _generate_overview(self) -> str:
        overview_html = f"""<div class="grid-container"><div class="grid-item"><h4>Dataset Shape</h4><p><b>Rows:</b> {self.df.shape[0]}</p><p><b>Columns:</b> {self.df.shape[1]}</p></div><div class="grid-item"><h4>Column Types</h4><p><b>Numeric:</b> {len(self.numeric_cols_)}</p><p><b>Categorical:</b> {len(self.categorical_cols_)}</p></div><div class="grid-item"><h4>Memory Usage</h4><p>{(self.df.memory_usage(deep=True).sum() / 1024**2):.2f} MB</p></div></div><h3>Data Preview (First 10 Rows)</h3><div class='table-scroll-wrapper'>{self.df.head(10).to_html(classes='styled-table')}</div>"""
        return overview_html

    def _generate_descriptive_stats(self) -> str:
        try: return f"<div class='table-scroll-wrapper'>{self.df.describe(include='all').transpose().to_html(classes='styled-table')}</div>"
        except Exception as e: return f"<p>Could not generate descriptive statistics: {e}</p>"

    def _analyze_missing_values(self) -> tuple[str, str]:
        missing_counts = self.df.isnull().sum()
        if missing_counts.sum() == 0: return "<p>No missing values found.</p>", ""
        missing_percentage = (missing_counts / len(self.df)) * 100
        missing_df = pd.DataFrame({'missing_count': missing_counts, 'missing_percentage': missing_percentage})
        missing_df = missing_df[missing_df['missing_count'] > 0].sort_values(by='missing_percentage', ascending=False)
        summary_html = f"<div class='table-scroll-wrapper'>{missing_df.to_html(classes='styled-table')}</div>"
        fig, ax = plt.subplots(figsize=(15, 8)); sns.heatmap(self.df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax); ax.set_title('Heatmap of Missing Values', fontsize=plt.rcParams['axes.titlesize'] * 1.2)
        return summary_html, plot_to_base64(fig)

    def _analyze_outliers(self) -> str:
        if not self.numeric_cols_: return "<p>No numeric columns to analyze for outliers.</p>"
        all_panels_html = ""; outlier_found = False
        for col in self.numeric_cols_:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75); IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            if not outliers.empty:
                outlier_found = True
                fig, ax = plt.subplots(figsize=(8, 5)); sns.boxplot(x=self.df[col], ax=ax)
                ax.set_title(f'Outliers for {col}', fontsize=plt.rcParams['axes.titlesize'] * 1.2); ax.set_xlabel(col, fontsize=plt.rcParams['axes.labelsize'] * 1.1)
                ax.tick_params(axis='x', labelsize=plt.rcParams['xtick.labelsize'] * 1.1); plot_b64 = plot_to_base64(fig)
                info_html = f"""<div class="panel-info"><h4>Outlier Information</h4><p><b>Count:</b> {len(outliers)} ({len(outliers)/len(self.df)*100:.2f}%)</p><p><b>IQR Lower:</b> {lower_bound:.2f}</p><p><b>IQR Upper:</b> {upper_bound:.2f}</p></div>"""
                all_panels_html += f"""<div class="panel-container"><div class="panel-title"><h3>{col}</h3></div><div class="panel-plot"><img src="{plot_b64}"></div>{info_html}</div>"""
        if not outlier_found: return "<p>No outliers were detected in any numeric columns based on the IQR method.</p>"
        return all_panels_html

    def _analyze_numerical_distributions(self) -> str:
        if not self.numeric_cols_: return "<p>No numeric columns to analyze for distribution.</p>"
        all_panels_html = ""
        for col in self.numeric_cols_:
            skew_val = self.df[col].skew()
            skew_label, label_class = ("Skewed", "label-skewed") if abs(skew_val) > 0.5 else ("Normal", "label-normal")
            fig, ax = plt.subplots(figsize=(8, 5)); sns.histplot(self.df[col], kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}', fontsize=plt.rcParams['axes.titlesize'] * 1.2); ax.set_xlabel(col, fontsize=plt.rcParams['axes.labelsize'] * 1.1)
            ax.set_ylabel('Count', fontsize=plt.rcParams['axes.labelsize'] * 1.1); ax.tick_params(axis='x', labelsize=plt.rcParams['xtick.labelsize'] * 1.1); ax.tick_params(axis='y', labelsize=plt.rcParams['ytick.labelsize'] * 1.1)
            plot_b64 = plot_to_base64(fig)
            info_html = f"""<div class="panel-info"><h4>Distribution Analysis</h4><p><b>Skewness:</b> {skew_val:.2f}</p><span class="label {label_class}">{skew_label}</span></div>"""
            all_panels_html += f"""<div class="panel-container"><div class="panel-title"><h3>{col}</h3></div><div class="panel-plot"><img src="{plot_b64}"></div>{info_html}</div>"""
        return all_panels_html

    def _plot_correlation_report(self) -> tuple[str, str, str]:
        if len(self.numeric_cols_) < 2: return "<p>Not enough numeric features for correlation.</p>", "", ""
        correlation_matrix = self.df[self.numeric_cols_].corr()
        correlation_content_html = ""
        if len(self.numeric_cols_) > self.CORRELATION_COL_THRESHOLD:
            correlation_content_html = f"""<h3>Correlation Matrix Table</h3><div class='table-scroll-wrapper-large'>{correlation_matrix.round(2).to_html(classes='styled-table sticky-index-table')}</div>"""
        else:
            fig, ax = plt.subplots(figsize=(len(self.numeric_cols_)*0.8, len(self.numeric_cols_)*0.7)); sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax, annot_kws={"size": 6})
            ax.set_title('Correlation Matrix', fontsize=plt.rcParams['axes.titlesize'] * 1.2)
            ax.tick_params(axis='x', labelsize=plt.rcParams['xtick.labelsize']); ax.tick_params(axis='y', labelsize=plt.rcParams['ytick.labelsize'], rotation=0); heatmap_b64 = plot_to_base64(fig)
            correlation_content_html = f"<h3>Heatmap</h3><div class='plot-container'><img src='{heatmap_b64}'></div>"
        corr_pairs = correlation_matrix.unstack().sort_values(kind="quicksort", ascending=False); corr_pairs = corr_pairs[corr_pairs != 1.0]
        top_positive = corr_pairs[corr_pairs > 0.7].drop_duplicates().head(5); top_negative = corr_pairs[corr_pairs < -0.7].drop_duplicates().head(5)
        pos_corr_html = top_positive.to_frame('correlation').to_html(classes='styled-table') if not top_positive.empty else "<p>No strong positive correlations (> 0.7) found.</p>"
        neg_corr_html = top_negative.to_frame('correlation').to_html(classes='styled-table') if not top_negative.empty else "<p>No strong negative correlations (< -0.7) found.</p>"
        return correlation_content_html, pos_corr_html, neg_corr_html
        
    def _analyze_target_variable(self) -> str:
        target_series = self.df[self.target]; dtype = target_series.dtype; n_unique = target_series.nunique()
        CLASSIFICATION_THRESHOLD = 25; problem_type = "Unknown"
        if dtype in ['object', 'category', 'bool']: problem_type = "Classification"
        elif pd.api.types.is_numeric_dtype(dtype):
            if n_unique == 2: problem_type = "Binary Classification"
            elif 2 < n_unique <= CLASSIFICATION_THRESHOLD: problem_type = "Multiclass Classification"
            else: problem_type = "Regression"
        if "Classification" in problem_type:
            counts = target_series.value_counts(); percentages = target_series.value_counts(normalize=True) * 100
            dist_df = pd.DataFrame({'Counts': counts, 'Percentage (%)': percentages.round(2)})
            fig, ax = plt.subplots(figsize=(10, 6)); sns.countplot(x=target_series, ax=ax, order=counts.index)
            ax.set_title(f'Distribution of Target: {self.target}', fontsize=plt.rcParams['axes.titlesize'] * 1.2); ax.set_xlabel(self.target, fontsize=plt.rcParams['axes.labelsize'] * 1.1)
            ax.set_ylabel('Count', fontsize=plt.rcParams['axes.labelsize'] * 1.1); ax.tick_params(axis='x', labelsize=plt.rcParams['xtick.labelsize'] * 1.1); ax.tick_params(axis='y', labelsize=plt.rcParams['ytick.labelsize'] * 1.1)
            plot_b64 = plot_to_base64(fig)
            return f"""<div class="grid-container"><div class="grid-item"><h4>Detected Problem Type</h4><p>{problem_type}</p></div><div class="grid-item"><h4>Number of Classes</h4><p>{n_unique}</p></div></div><h3>Class Distribution</h3><div class='table-scroll-wrapper'>{dist_df.to_html(classes='styled-table')}</div><div class="plot-container" style="margin-top: 2rem;">{f'<img src="{plot_b64}">' if plot_b64 else ''}</div>"""
        elif problem_type == "Regression":
            stats_df = target_series.describe().to_frame().round(2)
            fig1, ax1 = plt.subplots(figsize=(10, 6)); sns.histplot(target_series, kde=True, ax=ax1)
            ax1.set_title(f'Distribution of Target: {self.target}', fontsize=plt.rcParams['axes.titlesize'] * 1.2); ax1.set_xlabel(self.target, fontsize=plt.rcParams['axes.labelsize'] * 1.1)
            ax1.set_ylabel('Count', fontsize=plt.rcParams['axes.labelsize'] * 1.1); ax1.tick_params(axis='x', labelsize=plt.rcParams['xtick.labelsize'] * 1.1); ax1.tick_params(axis='y', labelsize=plt.rcParams['ytick.labelsize'] * 1.1); plot1_b64 = plot_to_base64(fig1)
            fig2, ax2 = plt.subplots(figsize=(10, 4)); sns.boxplot(x=target_series, ax=ax2)
            ax2.set_title(f'Boxplot of Target: {self.target}', fontsize=plt.rcParams['axes.titlesize'] * 1.2); ax2.set_xlabel(self.target, fontsize=plt.rcParams['axes.labelsize'] * 1.1)
            ax2.tick_params(axis='x', labelsize=plt.rcParams['xtick.labelsize'] * 1.1); plot2_b64 = plot_to_base64(fig2)
            return f"""<div class="grid-container"><div class="grid-item"><h4>Detected Problem Type</h4><p>{problem_type}</p></div><div class="grid-item"><h4>Unique Values</h4><p>{n_unique}</p></div></div><h3>Descriptive Statistics</h3><div class='table-scroll-wrapper'>{stats_df.to_html(classes='styled-table')}</div><div style="display: flex; gap: 2rem; margin-top: 2rem; flex-wrap: wrap;"><div class="plot-container" style="flex: 1; min-width: 400px;"><h4>Distribution Plot</h4>{f'<img src="{plot1_b64}">' if plot1_b64 else ''}</div><div class="plot-container" style="flex: 1; min-width: 400px;"><h4>Box Plot</h4>{f'<img src="{plot2_b64}">' if plot2_b64 else ''}</div></div>"""
        else: return f"<p>Could not determine problem type for target '{self.target}'. Dtype: {dtype}, Unique Values: {n_unique}.</p>"
    
    def generate_html_report(self, report_height: int = 1000) -> HTML:
        """Assembles all generated content into a final interactive HTML report."""
        overview_html = self._generate_overview()
        stats_html = self._generate_descriptive_stats()
        missing_summary_html, missing_plot_b64 = self._analyze_missing_values()
        outliers_html = self._analyze_outliers()
        num_dist_html = self._analyze_numerical_distributions()
        correlation_main_content_html, pos_corr_html, neg_corr_html = self._plot_correlation_report()
        target_nav_button_html, target_content_section_html = "", ""
        if self.target:
            target_analysis_html = self._analyze_target_variable()
            target_nav_button_html = f"""<button class="nav-btn" onclick="showTab(event, 'target')">🎯 Target Analysis</button>"""
            target_content_section_html = f"""<section id="target" class="content-section"><h2>Target Variable: {self.target}</h2>{target_analysis_html}</section>"""
        missing_values_html = f"<h3>Summary</h3>{missing_summary_html}<h3>Pattern</h3><div class='plot-container'><img src='{missing_plot_b64}'></div>" if missing_plot_b64 else missing_summary_html
        correlation_html = f"{correlation_main_content_html}<h3>Top Positive Correlations (> 0.7)</h3>{pos_corr_html}<h3>Top Negative Correlations (< -0.7)</h3>{neg_corr_html}"
        
        html_template = f"""
        <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Noventis Automated EDA Report</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Exo+2:wght@600;800&display=swap');
            :root {{
                --bg-dark-1: #0D1117; --bg-dark-2: #161B22; --bg-dark-3: #010409; --border-color: #30363D;
                --text-light: #C9D1D9; --text-muted: #8B949E; --primary-blue: #58A6FF; --primary-orange: #F78166;
                --font-main: 'Roboto', sans-serif; --font-header: 'Exo 2', sans-serif;
            }}
            body {{ font-family: var(--font-main); background-color: transparent; color: var(--text-light); margin: 0; padding: 0; }}
            
            /* CSS BARU: 'Jendela' atau frame untuk laporan */
            .report-frame {{
                height: {report_height}px;
                width: 100%;
                border: 1px solid var(--border-color);
                border-radius: 10px;
                overflow: hidden; /* Sembunyikan scrollbar dari frame itu sendiri */
            }}
            
            /* Modifikasi .container agar bisa di-scroll di dalam frame */
            .container {{
                width: 100%; max-width: 1400px; margin: auto; background-color: var(--bg-dark-1);
                height: 100%; /* Penting: isi penuh tinggi frame */
                overflow: auto; /* Penting: scrollbar utama ada di sini */
            }}
            header {{ position: sticky; top: 0; z-index: 10; padding: 1.5rem 2.5rem; border-bottom: 1px solid var(--border-color); background-color: var(--bg-dark-2); }}
            header h1 {{ font-family: var(--font-header); font-size: 2.5rem; margin: 0; color: var(--primary-blue); }} header p {{ margin: 0.25rem 0 0; color: var(--text-muted); font-size: 1rem; }}
            .navbar {{ position: sticky; top: 118px; z-index: 10; display: flex; flex-wrap: wrap; background-color: var(--bg-dark-2); padding: 0 2.5rem; border-bottom: 1px solid var(--border-color); }}
            .nav-btn {{ background: none; border: none; color: var(--text-muted); padding: 1rem 1.5rem; font-size: 1rem; cursor: pointer; border-bottom: 3px solid transparent; transition: all 0.2s ease-in-out; }}
            .nav-btn:hover {{ color: var(--text-light); }} .nav-btn.active {{ color: var(--primary-orange); border-bottom-color: var(--primary-orange); font-weight: 700; }}
            main {{ padding: 2.5rem; }} .content-section {{ display: none; }} .content-section.active {{ display: block; animation: fadeIn 0.5s; }}
            @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
            h2 {{ font-family: var(--font-header); font-size: 2rem; color: var(--primary-orange); border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem; margin-top: 0; }}
            h3 {{ font-family: var(--font-header); color: var(--primary-blue); font-size: 1.5rem; margin-top: 2rem; }}
            .grid-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
            .grid-item {{ background-color: var(--bg-dark-2); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--border-color); }}
            .table-scroll-wrapper {{ margin-top: 1rem; overflow: auto; }}
            .table-scroll-wrapper-large {{ margin-top: 1rem; overflow: auto; max-height: 600px; }}
            .styled-table {{ width: 100%; color: var(--text-muted); background-color: var(--bg-dark-2); border-collapse: collapse; border-radius: 8px; overflow: hidden; font-size: 0.9rem; }}
            .styled-table th, .styled-table td {{ border-bottom: 1px solid var(--border-color); padding: 0.8rem 1rem; text-align: left; white-space: nowrap; }}
            .styled-table thead th {{ background-color: var(--bg-dark-3); }}
            .sticky-index-table thead th {{ position: sticky; top: 0; z-index: 2; }}
            .sticky-index-table tbody th {{ position: sticky; left: 0; z-index: 1; background-color: var(--bg-dark-3); }}
            .sticky-index-table thead th:first-child {{ left: 0; z-index: 3; }}
            .plot-container {{ background-color: var(--bg-dark-2); padding: 1rem; margin-top: 1rem; border-radius: 8px; border: 1px solid var(--border-color); text-align: center; }}
            .panel-container {{ display: grid; grid-template-columns: 200px 1fr 250px; align-items: center; gap: 1.5rem; background-color: var(--bg-dark-2); border: 1px solid var(--border-color); padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; }}
            .label {{ padding: 0.3rem 0.8rem; border-radius: 15px; font-weight: bold; font-size: 0.9rem; color: #FFFFFF; }}
            .label-skewed {{ background-color: #D73A49; }}
            .label-normal {{ background-color: #28A745; }}
        </style>
        </head><body>
            <div class="report-frame">
                <div class="container">
                    <header><h1>Noventis Automated EDA Report</h1><p>A comprehensive overview of the dataset's characteristics.</p></header>
                    <nav class="navbar">
                        <button class="nav-btn active" onclick="showTab(event, 'overview')">📊 Overview</button>
                        {target_nav_button_html}
                        <button class="nav-btn" onclick="showTab(event, 'stats')">🔢 Descriptive Stats</button>
                        <button class="nav-btn" onclick="showTab(event, 'missing')">💧 Missing Values</button>
                        <button class="nav-btn" onclick="showTab(event, 'outliers')">📈 Outlier Distribution</button>
                        <button class="nav-btn" onclick="showTab(event, 'num_dist')">📉 Numerical Distribution</button>
                        <button class="nav-btn" onclick="showTab(event, 'correlation')">🔗 Correlation</button>
                    </nav>
                    <main>
                        <section id="overview" class="content-section active"><h2>Dataset Overview</h2>{overview_html}</section>
                        {target_content_section_html}
                        <section id="stats" class="content-section"><h2>Descriptive Statistics</h2>{stats_html}</section>
                        <section id="missing" class="content-section"><h2>Missing Values Analysis</h2>{missing_values_html}</section>
                        <section id="outliers" class="content-section"><h2>Outlier Analysis (IQR Method)</h2>{outliers_html}</section>
                        <section id="num_dist" class="content-section"><h2>Numerical Feature Skewness</h2>{num_dist_html}</section>
                        <section id="correlation" class="content-section"><h2>Correlation Analysis</h2>{correlation_html}</section>
                    </main>
                </div>
            </div>
        <script>
            function showTab(event, tabName) {{
                // Javascript tidak perlu tahu tentang frame, ia bekerja di dalam container
                const container = document.querySelector('.container');
                container.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
                container.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
                container.querySelector('#' + tabName).classList.add('active');
                event.currentTarget.classList.add('active');
            }}
        </script>
        </body></html>
        """
        return HTML(html_template)

    def run(self, report_height: int = 1000) -> HTML:
        """
        Runs the entire EDA workflow and displays the report in a contained, scrollable frame.
        
        Args:
            report_height (int): The height of the report window in pixels.
        """
        print("Generating EDA report, please wait...")
        
        report_html = self.generate_html_report(report_height=report_height)
        
        print("Report generated successfully. Displaying below.")
        return report_html