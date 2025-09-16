import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import io
import base64
import uuid
import scipy.stats as stats
from typing import Union

# --- Helper Function ---
def plot_to_base64(fig):
    if fig is None: return ""
    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor()); buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8'); buf.close(); plt.close(fig)
    return f"data:image/png;base64,{img_str}"

# --- Tema Plot Standar Tunggal (Gelap) ---
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.titlesize': 12, 
    'figure.titlesize': 14, 'legend.fontsize': 10, 'figure.facecolor': '#161B22', 'axes.facecolor': '#161B22', 
    'text.color': '#C9D1D9', 'axes.labelcolor': '#C9D1D9', 'xtick.color': '#8B949E', 'ytick.color': '#8B949E',
    'grid.color': '#30363D', 'patch.edgecolor': '#30363D', 'figure.edgecolor': '#161B22',
})

class EDAAnalyzer:
    def __init__(self, data: Union[pd.DataFrame, str], target: str = None, personality: str = 'default'):
        if isinstance(data, str):
            try: df = pd.read_csv(data)
            except FileNotFoundError: raise FileNotFoundError(f"File tidak ditemukan di '{data}'")
        elif isinstance(data, pd.DataFrame): df = data.copy()
        else: raise TypeError("Input 'data' harus berupa path file CSV atau pandas DataFrame.")
        
        allowed_personalities = ['default', 'academic', 'business', 'all'];
        if personality not in allowed_personalities: raise ValueError(f"Personality '{personality}' is not supported. Please choose from {allowed_personalities}.")
        
        self.df = df; self.target = target
        if self.target and self.target not in self.df.columns: raise ValueError(f"Target column '{self.target}' not found in the DataFrame.")
        self.personality = personality; self.numeric_cols_ = self.df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols_ = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.report_id = f"report-{uuid.uuid4().hex[:8]}"
        self.CORRELATION_COL_THRESHOLD = 30
    
    def _count_total_outliers(self):
        total_outliers = 0
        for col in self.numeric_cols_:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75); IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]; total_outliers += len(outliers)
        return total_outliers

    def _find_most_impactful_categorical(self):
        if not self.target or self.target not in self.numeric_cols_ or not self.categorical_cols_: return self.categorical_cols_[0] if self.categorical_cols_ else None
        f_values = {}
        for col in self.categorical_cols_:
            if self.df[col].nunique() > 1 and self.df[col].nunique() < 50:
                groups = [self.df[self.target][self.df[col] == category].dropna() for category in self.df[col].unique() if not self.df[self.target][self.df[col] == category].dropna().empty]
                if len(groups) < 2: continue
                try:
                    f_stat, p_val = stats.f_oneway(*groups)
                    if not np.isnan(f_stat): f_values[col] = f_stat
                except: continue
        return max(f_values, key=f_values.get) if f_values else (self.categorical_cols_[0] if self.categorical_cols_ else None)

    def _business_panel_data_quality_roi(self) -> str:
        missing_cells = self.df.isnull().sum().sum(); missing_pct = (missing_cells / self.df.size) * 100
        outliers_count = self._count_total_outliers(); duplicates_count = self.df.duplicated().sum()
        quality_score = max(0, 100 - (missing_pct * 1.5) - (duplicates_count / len(self.df) * 100) - (outliers_count / self.df.size * 100 * 2.0))
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'aspect': 'equal'})
        wedges, texts = ax.pie([quality_score, 100 - quality_score], wedgeprops=dict(width=0.45), startangle=90, colors=['#58A6FF', '#30363D'])
        ax.text(0, 0, f'{quality_score:.0f}%', ha='center', va='center', fontsize=32, weight="bold", color='#FFFFFF')
        plot_b64 = plot_to_base64(fig)
        kpi_html = f"""<div class="biz-kpi-container"><div class="biz-kpi-item"><span class="kpi-label">Missing Cells</span><span class="kpi-value-small">{missing_cells:,} ({missing_pct:.2f}%)</span></div><div class="biz-kpi-item"><span class="kpi-label">Outliers Detected</span><span class="kpi-value-small">{outliers_count:,}</span></div><div class="biz-kpi-item"><span class="kpi-label">Duplicate Rows</span><span class="kpi-value-small">{duplicates_count:,}</span></div></div>"""
        top_section_html = f"<div class='biz-panel-split'><div class='biz-gauge-container'><img src='{plot_b64}'></div><div class='biz-kpi-wrapper'>{kpi_html}</div></div>"
        
        missing_pct_col = (self.df.isnull().sum() / len(self.df)) * 100
        top_missing = missing_pct_col[missing_pct_col > 0].nlargest(5)
        missing_plot_html = ""
        if not top_missing.empty:
            fig_miss, ax_miss = plt.subplots(figsize=(8, 4)); sns.barplot(x=top_missing.values, y=top_missing.index, ax=ax_miss, color='#58A6FF', orient='h')
            ax_miss.set_title('Top 5 Columns with Missing Data', fontsize=12); ax_miss.set_xlabel('Percentage Missing (%)', fontsize=10)
            ax_miss.tick_params(axis='x', labelsize=10); ax_miss.tick_params(axis='y', labelsize=10) # PERBAIKAN FONT
            missing_plot_html = plot_to_base64(fig_miss)
        
        outlier_counts = {}
        for col in self.numeric_cols_:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75); IQR = Q3 - Q1; lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]; outlier_counts[col] = len(outliers)
        top_outliers = pd.Series(outlier_counts).nlargest(5)
        top_outliers = top_outliers[top_outliers > 0]
        outlier_plot_html = ""
        if not top_outliers.empty:
            fig_out, ax_out = plt.subplots(figsize=(8, 4)); sns.barplot(x=top_outliers.values, y=top_outliers.index, ax=ax_out, color='#F78166', orient='h')
            ax_out.set_title('Top 5 Columns with Most Outliers', fontsize=12); ax_out.set_xlabel('Number of Outliers', fontsize=10)
            ax_out.tick_params(axis='x', labelsize=10); ax_out.tick_params(axis='y', labelsize=10) # PERBAIKAN FONT
            outlier_plot_html = plot_to_base64(fig_out)
        
        bottom_section_html = f"""
            <h3 class="detail-header">Detail Permasalahan Kualitas Data</h3>
            <div class="biz-details-grid">
                <div class="grid-item"><h4>Top Missing Values</h4>{f"<img src='{missing_plot_html}'>" if missing_plot_html else "<p>Tidak ada data hilang.</p>"}</div>
                <div class="grid-item"><h4>Top Outliers</h4>{f"<img src='{outlier_plot_html}'>" if outlier_plot_html else "<p>Tidak ada outlier terdeteksi.</p>"}</div>
            </div>
        """
        return top_section_html + bottom_section_html

    def _business_panel_customer_intelligence(self) -> str:
        if not self.target or self.target not in self.numeric_cols_ or not self.categorical_cols_: return "<div class='biz-panel-placeholder'><h4>Customer Intelligence</h4><p>Requires a numeric target and categorical features.</p></div>"
        segment_col = self._find_most_impactful_categorical()
        if not segment_col: return "<div class='biz-panel-placeholder'><h4>Customer Intelligence</h4><p>No suitable categorical feature found for segmentation.</p></div>"
        segment_impact = self.df.groupby(segment_col)[self.target].sum().nlargest(5)
        fig, ax = plt.subplots(figsize=(6, 5), subplot_kw={'aspect': 'equal'})
        wedges, texts, autotexts = ax.pie(segment_impact, labels=segment_impact.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Blues_r", len(segment_impact)))
        plt.setp(autotexts, size=10, weight="bold", color="white"); ax.set_title(f"Impact by '{segment_col}'")
        plot_b64 = plot_to_base64(fig)
        summary_html = "<h4>Revenue Impact:</h4>" + segment_impact.to_frame().map('{:,.0f}'.format).to_html(classes='styled-table-small')
        return f"<div class='biz-panel'><img src='{plot_b64}'>{summary_html}</div>"

    def _business_panel_priority_matrix(self) -> str:
        if not self.target or self.target not in self.numeric_cols_ or len(self.numeric_cols_) < 2: return "<div class='biz-panel-placeholder'><h4>Priority Matrix</h4><p>Requires a numeric target to analyze feature impact.</p></div>"
        impact = self.df[self.numeric_cols_].corrwith(self.df[self.target]).abs().drop(self.target, errors='ignore')
        quality = (1 - self.df[self.numeric_cols_].isnull().sum() / len(self.df)) * 100
        matrix_df = pd.DataFrame({'Impact': impact, 'Quality (%)': quality.round(2)}).dropna().sort_values(by="Impact", ascending=False)
        if matrix_df.empty: return "<div class='biz-panel-placeholder'><h4>Priority Matrix</h4><p>Could not compute matrix.</p></div>"
        median_impact = matrix_df['Impact'].median(); median_quality = matrix_df['Quality (%)'].median()
        def assign_quadrant(row):
            if row['Impact'] >= median_impact and row['Quality (%)'] >= median_quality: return 'Focus Here'
            if row['Impact'] >= median_impact and row['Quality (%)'] < median_quality: return 'Strategic Fix'
            if row['Impact'] < median_impact and row['Quality (%)'] >= median_quality: return 'Easy Win'
            return 'Low Priority'
        matrix_df['Priority Quadrant'] = matrix_df.apply(assign_quadrant, axis=1)
        def style_quadrant(quadrant):
            colors = {'Focus Here': '#28A745', 'Strategic Fix': '#FD7E14', 'Easy Win': '#007BFF', 'Low Priority': '#6C757D'}
            return f"background-color: {colors.get(quadrant, '')}; color: white; font-weight: bold;"
        styler = matrix_df.style.map(style_quadrant, subset=['Priority Quadrant'])\
                               .format({'Impact': "{:.2f}", 'Quality (%)': "{:.2f}"})
        table_html = styler.to_html(classes='styled-table')
        return f"<div class='table-scroll-wrapper' style='max-height: 500px;'>{table_html}</div>"

    def _generate_business_impact_dashboard(self) -> str:
        panel1 = self._business_panel_data_quality_roi()
        panel2 = self._business_panel_customer_intelligence()
        panel3 = self._business_panel_priority_matrix()
        return f"""<div class="biz-dashboard-grid"><div class="grid-item"><h2>Data Quality ROI</h2>{panel1}</div><div class="grid-item"><h2>Customer Intelligence</h2>{panel2}</div><div class="grid-item"><h2>Priority Matrix</h2>{panel3}</div></div>"""
        
    def _generate_overview(self) -> str:
        base_html = f"""<div class="grid-container"><div class="grid-item"><h4>Dataset Shape</h4><p><b>Rows:</b> {self.df.shape[0]:,}</p><p><b>Columns:</b> {self.df.shape[1]}</p></div><div class="grid-item"><h4>Column Types</h4><p><b>Numeric:</b> {len(self.numeric_cols_)}</p><p><b>Categorical:</b> {len(self.categorical_cols_)}</p></div><div class="grid-item"><h4>Memory Usage</h4><p>{(self.df.memory_usage(deep=True).sum() / 1024**2):.2f} MB</p></div></div>"""
        if self.personality == 'academic':
            high_cardinality = [col for col in self.categorical_cols_ if self.df[col].nunique() > 50]; constant_cols = [col for col in self.df.columns if self.df[col].nunique() == 1]
            base_html += f"""<div class="grid-container"><div class="grid-item"><h4>Data Quality Warnings</h4><p><b>High Cardinality (>50):</b> {len(high_cardinality)}</p><p><b>Constant Columns:</b> {len(constant_cols)}</p><p><b>Duplicate Rows:</b> {self.df.duplicated().sum()}</p></div></div>"""
        preview_html = f"<h3>Data Preview (First 5 Rows)</h3><div class='table-scroll-wrapper'>{self.df.head().to_html(classes='styled-table')}</div>"
        return base_html + preview_html

    def _generate_descriptive_stats(self) -> str:
        try:
            stats_df = self.df.describe(include='all').transpose()
            if self.personality == 'academic' and len(self.numeric_cols_) > 0:
                numeric_stats = self.df[self.numeric_cols_]; stats_df['variance'] = numeric_stats.var(); stats_df['skewness'] = numeric_stats.skew(); stats_df['kurtosis'] = numeric_stats.kurt()
            return f"<div class='table-scroll-wrapper'>{stats_df.to_html(classes='styled-table')}</div>"
        except Exception as e: return f"<p>Could not generate descriptive statistics: {e}</p>"

    def _analyze_missing_values(self) -> str:
        missing_counts = self.df.isnull().sum()
        if missing_counts.sum() == 0: return "<p>No missing values found.</p>"
        missing_percentage = (missing_counts / len(self.df)) * 100; missing_df = pd.DataFrame({'missing_count': missing_counts, 'missing_percentage': missing_percentage})
        missing_df = missing_df[missing_df['missing_count'] > 0].sort_values(by='missing_percentage', ascending=False)
        summary_html = f"<div class='table-scroll-wrapper'>{missing_df.to_html(classes='styled-table')}</div>"
        fig, ax = plt.subplots(figsize=(15, 8)); sns.heatmap(self.df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax); ax.set_title('Heatmap of Missing Values')
        plot_b64 = plot_to_base64(fig)
        return f"<h3>Summary</h3>{summary_html}<h3>Pattern</h3><div class='plot-container'><img src='{plot_b64}'></div>"

    def _analyze_outliers(self) -> str:
        if not self.numeric_cols_: return "<p>No numeric columns to analyze for outliers.</p>"
        all_panels_html = ""; outlier_found = False
        for col in self.numeric_cols_:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75); IQR = Q3 - Q1; lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            if not outliers.empty:
                outlier_found = True
                fig, ax = plt.subplots(figsize=(8, 5)); sns.boxplot(x=self.df[col], ax=ax); ax.set_title(f'Outliers for {col}'); ax.set_xlabel(col); plot_b64 = plot_to_base64(fig)
                info_html = f"""<div class="panel-info"><h4>Outlier Information</h4><p><b>Count:</b> {len(outliers)} ({len(outliers)/len(self.df)*100:.2f}%)</p><p><b>IQR Lower:</b> {lower_bound:.2f}</p><p><b>IQR Upper:</b> {upper_bound:.2f}</p></div>"""
                all_panels_html += f"""<div class="panel-container"><div class="panel-title"><h3>{col}</h3></div><div class="panel-plot"><img src="{plot_b64}"></div>{info_html}</div>"""
        if not outlier_found: return "<p>No outliers were detected in any numeric columns based on the IQR method.</p>"
        return all_panels_html

    def _analyze_numerical_distributions(self) -> str:
        if not self.numeric_cols_: return "<p>No numeric columns to analyze for distribution.</p>"
        all_panels_html = ""
        for col in self.numeric_cols_:
            col_data = self.df[col].dropna()
            if col_data.empty: continue
            skew_val = col_data.skew(); skew_label, label_class = ("Skewed", "label-skewed") if abs(skew_val) > 0.5 else ("Normal", "label-normal")
            fig, ax = plt.subplots(figsize=(8, 5)); sns.histplot(col_data, kde=True, ax=ax); ax.set_title(f'Distribution of {col}'); ax.set_xlabel(col); ax.set_ylabel('Count')
            plot_b64_main = plot_to_base64(fig)
            info_html = f"""<div class="panel-info"><h4>Distribution Analysis</h4><p><b>Skewness:</b> {skew_val:.2f}</p><span class="label {label_class}">{skew_label}</span></div>"""
            if self.personality == 'academic':
                fig_qq, ax_qq = plt.subplots(figsize=(6, 4)); stats.probplot(col_data, dist="norm", plot=ax_qq); ax_qq.set_title(f'Q-Q Plot for {col}'); ax_qq.get_lines()[0].set_markerfacecolor('#58A6FF'); ax_qq.get_lines()[0].set_markeredgecolor('#58A6FF'); ax_qq.get_lines()[1].set_color('#F78166'); qq_plot_b64 = plot_to_base64(fig_qq)
                if len(col_data) >= 5000: normality_text = "N > 5000, please assess visually."
                else:
                    try: shapiro_test = stats.shapiro(col_data); p_value = shapiro_test.pvalue; normality_text = f"Normal (p={p_value:.3f})" if p_value > 0.05 else f"Not Normal (p={p_value:.3f})"
                    except: normality_text = "Test could not be performed."
                info_html += f"""<div class="panel-info" style="margin-top:1rem;"><h4>Normality Test (Shapiro-Wilk)</h4><p>{normality_text}</p></div>"""; plot_html = f"""<div style="display:flex; gap:1rem;"><div style="flex:1;"><img src="{plot_b64_main}"></div><div style="flex:1;"><img src="{qq_plot_b64}"></div></div>"""
            else: plot_html = f'<img src="{plot_b64_main}">'
            all_panels_html += f"""<div class="panel-container"><div class="panel-title"><h3>{col}</h3></div><div class="panel-plot-wide">{plot_html}</div><div class="panel-info">{info_html}</div></div>"""
        return all_panels_html

    def _plot_correlation_report(self) -> str:
        if len(self.numeric_cols_) < 2: return "<p>Not enough numeric features for correlation.</p>"
        correlation_matrix = self.df[self.numeric_cols_].corr(); corr_pairs = correlation_matrix.unstack().sort_values(kind="quicksort", ascending=False); corr_pairs = corr_pairs[corr_pairs != 1.0]
        correlation_main_content_html = ""
        dropdown_html = """<div class="filter-container"><label for="corr-filter-{id}">Highlight correlations (absolute value):</label><select id="corr-filter-{id}" onchange="filterCorrelationTable(this.value, '{id}')"><option value="0.0">Show All</option><option value="0.5">&gt; 0.5</option><option value="0.7">&gt; 0.7</option></select></div>""".format(id=self.report_id)
        if len(self.numeric_cols_) > self.CORRELATION_COL_THRESHOLD:
            table_html = f"<table id='corr-matrix-{self.report_id}' class='styled-table sticky-index-table'><thead><tr><th></th>";
            for col in correlation_matrix.columns: table_html += f"<th>{col}</th>"
            table_html += "</tr></thead><tbody>"
            for index, row in correlation_matrix.iterrows():
                table_html += f"<tr><th>{index}</th>"
                for col, value in row.items(): table_html += f"<td data-corr='{value}'>{value:.2f}</td>"
                table_html += "</tr>"
            table_html += "</tbody></table>"
            correlation_main_content_html = f"<h3>Correlation Matrix Table</h3>{dropdown_html}<div class='table-scroll-wrapper-large'>{table_html}</div>"
        else:
            fig, ax = plt.subplots(figsize=(len(self.numeric_cols_)*0.8, len(self.numeric_cols_)*0.7)); sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax, annot_kws={"size": 6});
            ax.set_title('Correlation Matrix'); ax.tick_params(axis='x'); ax.tick_params(axis='y', rotation=0); heatmap_b64 = plot_to_base64(fig)
            correlation_main_content_html = f"<h3>Heatmap</h3><div class='plot-container'><img src='{heatmap_b64}'></div>"
        all_summary_tables_html = ""
        top_positive_07 = corr_pairs[corr_pairs > 0.7].drop_duplicates().head(5); top_negative_07 = corr_pairs[corr_pairs < -0.7].drop_duplicates().head(5)
        pos_corr_html_07 = top_positive_07.to_frame('correlation').to_html(classes='styled-table') if not top_positive_07.empty else "<p>No strong positive correlations (> 0.7) found.</p>"
        neg_corr_html_07 = top_negative_07.to_frame('correlation').to_html(classes='styled-table') if not top_negative_07.empty else "<p>No strong negative correlations (< -0.7) found.</p>"
        all_summary_tables_html += f'<div id="corr-summary-0.7-{self.report_id}" class="corr-summary-table" style="display: none;"><h3>Top Positive/Negative Correlations (> 0.7)</h3>{pos_corr_html_07}{neg_corr_html_07}</div>'
        top_positive_05 = corr_pairs[corr_pairs > 0.5].drop_duplicates().head(5); top_negative_05 = corr_pairs[corr_pairs < -0.5].drop_duplicates().head(5)
        pos_corr_html_05 = top_positive_05.to_frame('correlation').to_html(classes='styled-table') if not top_positive_05.empty else "<p>No strong positive correlations (> 0.5) found.</p>"
        neg_corr_html_05 = top_negative_05.to_frame('correlation').to_html(classes='styled-table') if not top_negative_05.empty else "<p>No strong negative correlations (< -0.5) found.</p>"
        all_summary_tables_html += f'<div id="corr-summary-0.5-{self.report_id}" class="corr-summary-table" style="display: none;"><h3>Top Positive/Negative Correlations (> 0.5)</h3>{pos_corr_html_05}{neg_corr_html_05}</div>'
        return f"{correlation_main_content_html}{all_summary_tables_html}"

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
            fig, ax = plt.subplots(figsize=(10, 6)); sns.countplot(x=target_series, ax=ax, order=counts.index); ax.set_title(f'Distribution of Target: {self.target}'); ax.set_xlabel(self.target); ax.set_ylabel('Count'); plot_b64 = plot_to_base64(fig)
            return f"""<div class="grid-container"><div class="grid-item"><h4>Detected Problem Type</h4><p>{problem_type}</p></div><div class="grid-item"><h4>Number of Classes</h4><p>{n_unique}</p></div></div><h3>Class Distribution</h3><div class='table-scroll-wrapper'>{dist_df.to_html(classes='styled-table')}</div><div class="plot-container" style="margin-top: 2rem;"><img src='{plot_b64}'></div>"""
        elif problem_type == "Regression":
            stats_df = target_series.describe().to_frame().round(2)
            fig1, ax1 = plt.subplots(figsize=(10, 6)); sns.histplot(target_series, kde=True, ax=ax1); ax1.set_title(f'Distribution of Target: {self.target}'); ax1.set_xlabel(self.target); ax1.set_ylabel('Count'); plot1_b64 = plot_to_base64(fig1)
            fig2, ax2 = plt.subplots(figsize=(10, 4)); sns.boxplot(x=target_series, ax=ax2); ax2.set_title(f'Boxplot of Target: {self.target}'); ax2.set_xlabel(self.target); plot2_b64 = plot_to_base64(fig2)
            return f"""<div class="grid-container"><div class="grid-item"><h4>Detected Problem Type</h4><p>{problem_type}</p></div><div class="grid-item"><h4>Unique Values</h4><p>{n_unique}</p></div></div><h3>Descriptive Statistics</h3><div class='table-scroll-wrapper'>{stats_df.to_html(classes='styled-table')}</div><div style="display: flex; gap: 2rem; margin-top: 2rem; flex-wrap: wrap;"><div class="plot-container" style="flex: 1; min-width: 400px;"><h4>Distribution Plot</h4><img src='{plot1_b64}'></div><div class="plot-container" style="flex: 1; min-width: 400px;"><h4>Box Plot</h4><img src='{plot2_b64}'></div></div>"""
        else: return f"<p>Could not determine problem type for target '{self.target}'. Dtype: {dtype}, Unique Values: {n_unique}.</p>"
    
    def generate_html_report(self, report_height: int, show_base_viz: bool) -> HTML:
        tabs_config = []; report_title = "Noventis Automated EDA Report"
        if self.personality in ['business', 'all']:
            report_title = "Noventis Business Intelligence Report"
            tabs_config.append({'id': 'business_impact', 'title': '🚀 Business Impact', 'content_func': self._generate_business_impact_dashboard})
        if self.personality in ['academic', 'all']:
            report_title = "Noventis Academic Diagnostic Dashboard"
            tabs_config.append({'id': 'academic_diag', 'title': '✨ Statistical Validation', 'content_func': lambda: "<h2>Konten Academic Dashboard akan ada di sini.</h2>"})
        if show_base_viz:
            base_tabs = [
                {'id': 'overview', 'title': '📊 Overview', 'content_func': self._generate_overview},
                {'id': 'stats', 'title': '🔢 Descriptive Stats', 'content_func': self._generate_descriptive_stats},
                {'id': 'missing', 'title': '💧 Missing Values', 'content_func': self._analyze_missing_values},
                {'id': 'outliers', 'title': '📈 Outlier Distribution', 'content_func': self._analyze_outliers},
                {'id': 'num_dist', 'title': '📉 Numerical Distribution', 'content_func': self._analyze_numerical_distributions},
                {'id': 'correlation', 'title': '🔗 Correlation', 'content_func': self._plot_correlation_report},
            ]
            if self.target: base_tabs.insert(1, {'id': 'target', 'title': '🎯 Target Analysis', 'content_func': self._analyze_target_variable})
            tabs_config.extend(base_tabs)
        if not tabs_config:
             tabs_config.append({'id': 'overview', 'title': '📊 Overview', 'content_func': self._generate_overview})
        navbar_html = ""; main_content_html = ""
        for i, tab in enumerate(tabs_config):
            active_class = 'active' if i == 0 else ''
            navbar_html += f"""<button class="nav-btn {active_class}" onclick="showTab(event, '{tab['id']}', '{self.report_id}')">{tab['title']}</button>"""
            content = tab['content_func'](); main_content_html += f"""<section id="{tab['id']}-{self.report_id}" class="content-section {active_class}"><h2>{tab['title'].split(' ', 1)[1]}</h2>{content}</section>"""
        
        html_template = f"""
        <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>{report_title}</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Exo+2:wght@600;800&display=swap');
            :root {{ --bg-dark-1: #0D1117; --bg-dark-2: #161B22; --bg-dark-3: #010409; --border-color: #30363D; --text-light: #C9D1D9; --text-muted: #8B949E; --primary-blue: #58A6FF; --primary-orange: #F78166; --font-main: 'Roboto', sans-serif; --font-header: 'Exo 2', sans-serif; }}
            body {{ font-family: var(--font-main); background-color: transparent; color: var(--text-light); margin: 0; padding: 0; }}
            .report-frame {{ height: {report_height}px; width: 100%; border: 1px solid var(--border-color); border-radius: 10px; overflow: hidden; background-color: var(--bg-dark-1); }}
            .container {{ width: 100%; max-width: 1600px; margin: auto; background-color: var(--bg-dark-1); height: 100%; overflow: auto; }}
            header {{ position: sticky; top: 0; z-index: 10; padding: 1.5rem 2.5rem; border-bottom: 1px solid var(--border-color); background-color: var(--bg-dark-2); }}
            header h1 {{ font-family: var(--font-header); font-size: 2.5rem; margin: 0; color: var(--primary-blue); }}
            header p {{ margin: 0.25rem 0 0; color: var(--text-muted); font-size: 1rem; }}
            .navbar {{ position: sticky; top: 118px; z-index: 10; display: flex; flex-wrap: wrap; background-color: var(--bg-dark-2); padding: 0 2.5rem; border-bottom: 1px solid var(--border-color); }}
            .nav-btn {{ background: none; border: none; color: var(--text-muted); padding: 1rem 1.5rem; font-size: 1rem; cursor: pointer; border-bottom: 3px solid transparent; transition: all 0.2s ease-in-out; }}
            .nav-btn:hover {{ color: var(--text-light); }}
            .nav-btn.active {{ color: var(--primary-orange); border-bottom-color: var(--primary-orange); font-weight: 700; }}
            main {{ padding: 2.5rem; }} .content-section {{ display: none; }} .content-section.active {{ display: block; }}
            h2, h3, h4 {{ font-family: var(--font-header); }}
            h2 {{ font-size: 2rem; color: var(--primary-orange); border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem; margin-top: 0; }}
            h3 {{ color: var(--primary-blue); font-size: 1.5rem; margin-top: 2rem; }}
            .grid-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
            .grid-item {{ background-color: var(--bg-dark-2); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--border-color); }}
            .table-scroll-wrapper, .table-scroll-wrapper-large {{ margin-top: 1rem; overflow: auto; }}
            .table-scroll-wrapper-large {{ max-height: 600px; }}
            .styled-table, .styled-table-small {{ width: 100%; color: var(--text-muted); background-color: var(--bg-dark-2); border-collapse: collapse; border-radius: 8px; overflow: hidden; font-size: 0.9rem; }}
            .styled-table th, .styled-table td, .styled-table-small th, .styled-table-small td {{ border-bottom: 1px solid var(--border-color); padding: 0.8rem 1rem; text-align: left; white-space: nowrap; transition: opacity 0.3s ease, background-color 0.3s ease; }}
            .styled-table thead th, .styled-table-small thead th {{ background-color: var(--bg-dark-3); }}
            .sticky-index-table thead th {{ position: sticky; top: 0; z-index: 2; }}
            .sticky-index-table tbody th {{ position: sticky; left: 0; z-index: 1; background-color: var(--bg-dark-3); }}
            .sticky-index-table thead th:first-child {{ left: 0; z-index: 3; }}
            .panel-container {{ display: grid; grid-template-columns: 250px 1fr 250px; align-items: center; gap: 1.5rem; background-color: var(--bg-dark-2); border: 1px solid var(--border-color); padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; }}
            .panel-container .panel-title {{ grid-column: 1; }} .panel-container .panel-plot-wide, .panel-container .panel-plot {{ grid-column: 2; }} .panel-container .panel-info {{ grid-column: 3; }}
            .panel-plot-wide div img, .panel-plot img, .panel-plot-wide > img {{ max-width: 100%; height: auto; }}
            .label {{ padding: 0.3rem 0.8rem; border-radius: 15px; font-weight: bold; font-size: 0.9rem; color: #FFFFFF; }}
            .label-skewed {{ background-color: #D73A49; }}
            .label-normal {{ background-color: #28A745; }}
            .filter-container {{ margin: 1rem 0; }} .filter-container select {{ background-color: var(--bg-dark-2); color: var(--text-light); border: 1px solid var(--border-color); border-radius: 5px; padding: 0.3rem; }}
            .biz-dashboard-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 1.5rem; }}
            .biz-panel {{ display: flex; flex-direction: column; align-items: center; text-align: center; height: 100%; }}
            .biz-panel-split {{ display: grid; grid-template-columns: 40% 60%; gap: 1.5rem; align-items: center; width: 100%; }}
            .biz-gauge-container {{ justify-self: center; }} .biz-panel img {{ max-width: 100%; height: auto; max-height: 280px; margin-bottom: 1rem; }}
            .biz-kpi-item {{ background-color: var(--bg-dark-3); padding: 0.75rem; border-radius: 5px; text-align: left; margin-bottom: 0.5rem;}}
            .kpi-label {{ display: block; font-size: 0.8rem; color: var(--text-muted); }}
            .kpi-value-small {{ display: block; font-size: 1.2rem; font-weight: bold; color: var(--text-light); }}
            .clean-list {{ list-style: none; padding: 0; text-align: left;}}
            .detail-header {{ margin-top: 2rem; margin-bottom: 1rem; text-align: center; }}
            .biz-details-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; width: 100%; }}
        </style>
        </head><body><div id="{self.report_id}" class="report-frame"><div class="container">
            <header><h1>{report_title}</h1><p>A comprehensive overview of the dataset's characteristics.</p></header>
            <nav class="navbar">{navbar_html}</nav>
            <main>{main_content_html}</main>
        </div></div>
        <script>
            function showTab(event, tabName, reportId) {{
                const reportFrame = document.getElementById(reportId);
                if (!reportFrame) return;
                reportFrame.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
                reportFrame.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
                const sectionId = `#${{tabName}}-${{reportId}}`;
                const sectionToShow = reportFrame.querySelector(sectionId);
                if (sectionToShow) sectionToShow.classList.add('active');
                event.currentTarget.classList.add('active');
            }}
            function filterCorrelationTable(threshold, reportId) {{
                const thresholdNum = parseFloat(threshold);
                const reportFrame = document.getElementById(reportId);
                if (!reportFrame) return;
                const table = reportFrame.querySelector(`#corr-matrix-${{reportId}}`);
                if (table) {{
                    const cells = table.getElementsByTagName('td');
                    for (let i = 0; i < cells.length; i++) {{
                        const cell = cells[i];
                        const corrVal = parseFloat(cell.getAttribute('data-corr'));
                        if (corrVal === 1.0) {{ cell.style.backgroundColor = 'var(--bg-dark-3)'; cell.style.fontWeight = 'bold'; cell.style.opacity = '1'; continue; }}
                        if (Math.abs(corrVal) >= thresholdNum) {{
                            cell.style.opacity = '1';
                            if (thresholdNum > 0) {{ cell.style.backgroundColor = corrVal > 0 ? 'rgba(88, 166, 255, 0.2)' : 'rgba(247, 129, 102, 0.2)'; }} 
                            else {{ cell.style.backgroundColor = 'transparent'; }}
                        }} else {{ cell.style.opacity = '0.3'; cell.style.backgroundColor = 'transparent'; }}
                    }}
                }}
                reportFrame.querySelectorAll('.corr-summary-table').forEach(tableDiv => {{ tableDiv.style.display = 'none'; }});
                let idToShow = thresholdNum >= 0.7 ? `corr-summary-0.7-${{reportId}}` : (thresholdNum >= 0.5 ? `corr-summary-0.5-${{reportId}}` : `corr-summary-0.7-${{reportId}}`);
                const tableToShow = reportFrame.querySelector(`#${{idToShow}}`);
                if (tableToShow) tableToShow.style.display = 'block';
            }}
        </script>
        </body></html>
        """
        return HTML(html_template)

    def run(self, report_height: int = 800, show_base_viz: bool = True) -> HTML:
        if not show_base_viz and self.personality == 'default':
            raise ValueError("Jika base_viz=False, Anda harus memilih 'personality' ('academic', 'business', atau 'all').")
        print(f"Generating EDA report with '{self.personality}' personality, please wait...")
        report_html = self.generate_html_report(report_height=report_height, show_base_viz=show_base_viz)
        print("Report generated successfully. Displaying below.")
        return report_html