import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, TargetEncoder , OneHotEncoder
from scipy.stats import chi2_contingency
from typing import Dict, List, Optional, Union, Tuple
import warnings
from collections import defaultdict
import logging
from category_encoders import OrdinalEncoder, BinaryEncoder, HashingEncoder
import matplotlib.pyplot as plt
import seaborn as sns
# from data_quality import assess_data_quality 

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoventisEncoder:
    """
    Advanced class for encoding categorical columns using automatic or manual methods.
    
    Features:
    - Intelligent automatic encoding selection
    - Multiple encoding methods (Label, OHE, Target, Ordinal, Binary, Hashing)
    - Detailed logging and recommendations
    - Cross-validation for target encoding (using scikit-learn's implementation)
    - Memory optimization
    - Handling of unseen categories
    """

    def __init__(self, 
                 method: str = 'auto', 
                 target_column: Optional[str] = None, 
                 columns_to_encode: Optional[List[str]] = None, 
                 category_mapping: Optional[Dict[str, Dict]] = None,
                 cv: int = 5,  # Changed from cv_folds to cv
                 smooth: Union[float, str] = 'auto',
                 target_type: str = 'auto',  # Added target_type parameter
                 verbose: bool = True):
        """
        Initializes the Advanced NoventisEncoder.

        Args:
            method (str): Encoding method ('auto', 'label', 'ohe', 'target', 'ordinal', 'binary', 'hashing')
            target_column (str, optional): Target variable column name
            columns_to_encode (list, optional): Specific columns to encode
            category_mapping (dict, optional): Custom mapping for ordinal encoding
            cv (int): Number of cross-validation folds for target encoding
            smooth (float or 'auto'): Smoothing parameter for target encoding
            target_type (str): Type of target ('auto', 'binary', 'continuous')
            verbose (bool): Whether to print detailed information
        """
        self.method = method
        self.target_column = target_column
        self.columns_to_encode = columns_to_encode
        self.category_mapping = category_mapping
        self.cv = cv
        self.smooth = smooth
        self.target_type = target_type
        self.verbose = verbose
        
        # Internal state
        self.encoders: Dict[str, object] = {}
        self.learned_cols: Dict[str, str] = {}
        self.encoding_stats: Dict[str, Dict] = {}
        self.column_info: Dict[str, Dict] = {}
        self.is_fitted = False
        self._original_df_snapshot = None 
        
        # Validation
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate initialization parameters."""
        valid_methods = ['auto', 'label', 'ohe', 'target', 'ordinal', 'binary', 'hashing']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        
        if self.method in ['auto', 'target'] and self.target_column is None:
            raise ValueError(f"target_column is required for method '{self.method}'")
        
        if self.cv < 2:
            raise ValueError("cv must be at least 2")

    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """
        Calculate Cramér's V statistic for categorical-categorical association.
        """
        try:
            mask = ~(x.isna() | y.isna())
            if mask.sum() < 5:
                return 0.0
                
            x_clean = x[mask]
            y_clean = y[mask]
            
            confusion_matrix = pd.crosstab(x_clean, y_clean)
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            
            if n == 0:
                return 0.0
                
            r, k = confusion_matrix.shape
            phi2 = chi2 / n
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            
            if min((kcorr-1), (rcorr-1)) <= 0:
                return 0.0
                
            return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
        except Exception as e:
            if self.verbose:
                logger.warning(f"Error calculating Cramér's V: {e}")
            return 0.0

    def _calculate_encoding_priority(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
        """
        Calculate encoding priority and statistics for each categorical column.
        """
        stats = {}
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
            y_binned = pd.qcut(y, q=min(4, y.nunique()), duplicates='drop', labels=False)
        else:
            y_binned = y
        
        for col in categorical_cols:
            col_data = X[col].dropna()
            if len(col_data) == 0:
                continue
                
            unique_count = col_data.nunique()
            missing_ratio = X[col].isna().sum() / len(X)
            
            try:
                correlation = self._cramers_v(X[col], y_binned)
            except:
                correlation = 0.0
            
            ohe_memory_impact = unique_count * len(X) * 8 / (1024**2)
            
            recommended_encoding = self._recommend_encoding(
                unique_count, correlation, missing_ratio, ohe_memory_impact
            )
            
            stats[col] = {
                'unique_count': unique_count,
                'missing_ratio': missing_ratio,
                'correlation_with_target': correlation,
                'ohe_memory_mb': ohe_memory_impact,
                'recommended_encoding': recommended_encoding,
                'sample_values': col_data.value_counts().head(3).to_dict()
            }
        
        return stats

    def _recommend_encoding(self, unique_count: int, correlation: float, 
                          missing_ratio: float, memory_impact: float) -> str:
        
        if unique_count == 2: return 'label'
        if unique_count > 50: return 'target' if correlation > 0.3 else 'hashing'
        if unique_count > 15:
            if correlation > 0.25: return 'target'
            elif memory_impact < 100: return 'binary'
            else: return 'hashing'
        if 3 <= unique_count <= 15:
            if correlation > 0.3: return 'ordinal_suggest'
            elif correlation > 0.15: return 'target'
            elif memory_impact < 50: return 'ohe'
            else: return 'binary'
        return 'label'
    
    def _determine_target_type(self, y: pd.Series) -> str:
        """Determine the target type for TargetEncoder."""
        if self.target_type != 'auto':
            return self.target_type
        
        # Auto-detect target type
        if pd.api.types.is_numeric_dtype(y):
            if y.nunique() <= 2:
                return 'binary'
            else:
                return 'continuous'
        else:
            # For categorical targets, treat as binary if 2 classes, otherwise continuous
            if y.nunique() <= 2:
                return 'binary'
            else:
                return 'continuous'

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self._original_df_snapshot = X.copy()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input 'X' must be a pandas DataFrame.")
        
        if self.method in ['auto', 'target'] and y is None:
            raise ValueError(f"Parameter 'y' is required for method '{self.method}'.")
        
        self.encoders, self.learned_cols, self.encoding_stats, self.column_info = {}, {}, {}, {}
        
        if self.verbose:
            print("=" * 60 + "\n🚀 NOVENTIS ENCODER - ANALYSIS REPORT\n" + "=" * 60)
        
        if self.method == 'auto':
            self._fit_auto_mode(X, y)
        else:
            self._fit_manual_mode(X, y)
        
        self.is_fitted = True
        
        if self.verbose: self._print_encoding_summary()
        return self

    def _fit_auto_mode(self, X: pd.DataFrame, y: pd.Series):
        self.column_info = self._calculate_encoding_priority(X, y)
        if self.verbose: print(f"📊 Analyzed {len(self.column_info)} categorical columns\n")
        
        for col, info in self.column_info.items():
            encoding_method = info['recommended_encoding']
            
            if encoding_method == 'ordinal_suggest':
                if self.verbose:
                    print(f"⚠️  MANUAL INTERVENTION RECOMMENDED for '{col}':\n"
                          f"   - High correlation with target ({info['correlation_with_target']:.3f})\n"
                          f"   - Consider using ordinal encoding with proper ordering\n"
                          f"   - Sample values: {list(info['sample_values'].keys())}\n"
                          f"   - Falling back to target encoding for now\n")
                encoding_method = 'target'
            
            self.learned_cols[col] = encoding_method
            
            if encoding_method == 'label':
                encoder = LabelEncoder()

                encoder.fit(X[col].astype(str).fillna('missing'))
            elif encoding_method == 'ohe':
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                encoder.fit(X[[col]].fillna('missing'))

            elif encoding_method == 'target':
                # Use scikit-learn's TargetEncoder with correct parameters
                target_type = self._determine_target_type(y)
                encoder = TargetEncoder(
                    cv=self.cv,
                    smooth=self.smooth,
                    target_type=target_type
                )
                encoder.fit(X[[col]], y)

            elif encoding_method == 'binary':
                encoder = BinaryEncoder()
                encoder.fit(X[col])

            elif encoding_method == 'hashing':
                n_components = min(8, max(4, int(np.log2(info['unique_count']))))
                encoder = HashingEncoder(n_components=n_components)
                encoder.fit(X[col])
            
            self.encoders[col] = encoder
            self.encoding_stats[col] = {
                'method': encoding_method, 'original_cardinality': info['unique_count'],
                'correlation': info['correlation_with_target'], 'memory_impact': info['ohe_memory_mb']
            }

    def _fit_manual_mode(self, X: pd.DataFrame, y: Optional[pd.Series]):
        columns_to_process = self.columns_to_encode or X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in columns_to_process:
            if col not in X.columns:
                if self.verbose: print(f"⚠️  Column '{col}' not found. Skipping.")
                continue
            
            if not (pd.api.types.is_categorical_dtype(X[col]) or pd.api.types.is_object_dtype(X[col])):
                if self.verbose: print(f"⚠️  Column '{col}' is not categorical. Skipping.")
                continue

            self.learned_cols[col] = self.method
            
            if self.method == 'label':
                encoder = LabelEncoder()
                encoder.fit(X[col].astype(str).fillna('missing'))

            elif self.method == 'ohe':
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                encoder.fit(X[[col]].fillna('missing'))

            elif self.method == 'target':
                # Use scikit-learn's TargetEncoder with correct parameters
                target_type = self._determine_target_type(y)
                encoder = TargetEncoder(
                    cv=self.cv,
                    smooth=self.smooth,
                    target_type=target_type
                )
                encoder.fit(X[[col]], y)

            elif self.method == 'ordinal':
                if self.category_mapping is None or col not in self.category_mapping:
                    raise ValueError(f"Mapping for '{col}' not found.")
                encoder = OrdinalEncoder(mapping=[{'col': col, 'mapping': self.category_mapping[col]}])
                encoder.fit(X[[col]])

            elif self.method == 'binary':
                encoder = BinaryEncoder()
                encoder.fit(X[col])
                
            elif self.method == 'hashing':
                encoder = HashingEncoder(n_components=min(8, max(4, int(np.log2(X[col].nunique())))))
                encoder.fit(X[col])
            
            self.encoders[col] = encoder

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform.")
        
        df = X.copy()
        if self.verbose: print("🔄 Transforming data...")
        transformed_cols = []
        
        for col, method in self.learned_cols.items():
            if col not in df.columns:
                if self.verbose: print(f"⚠️  Column '{col}' not in transform data. Skipping.")
                continue
            
            try:
                if method == 'label':
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str).fillna('missing'))
                    transformed_cols.append(f'{col}_encoded')
                elif method == 'ohe':
                    encoder_instance = self.encoders[col] 

                    encoded_array = encoder_instance.transform(df[[col]].fillna('missing'))
                    new_cols = encoder_instance.get_feature_names_out([col])
                    ohe_df = pd.DataFrame(encoded_array, columns=new_cols, index=df.index)
                    df = pd.concat([df, ohe_df], axis=1)
                elif method == 'target':
                    # Use the fitted TargetEncoder
                    encoded_values = self.encoders[col].transform(df[[col]])
                    df[f'{col}_target_encoded'] = encoded_values
                    transformed_cols.append(f'{col}_target_encoded')
                elif method == 'ordinal':
                    encoded_df = self.encoders[col].transform(df[[col]])
                    df[f'{col}_ordinal_encoded'] = encoded_df[col]
                    transformed_cols.append(f'{col}_ordinal_encoded')
                elif method in ['binary', 'hashing']:
                    encoded_df = self.encoders[col].transform(df[col])
                    encoded_df.columns = [f'{col}_{method}_{i}' for i in range(encoded_df.shape[1])]
                    df = pd.concat([df, encoded_df], axis=1)
                    transformed_cols.extend(encoded_df.columns.tolist())
                
                df.drop(col, axis=1, inplace=True)
            except Exception as e:
                if self.verbose: print(f"❌ Error encoding column '{col}': {e}")
                continue
        
        if self.verbose: print(f"✅ Successfully transformed {len(transformed_cols)} columns")
        return df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _print_encoding_summary(self):
        print("\n" + "📋 ENCODING SUMMARY" + "\n" + "-" * 40)
        method_counts = defaultdict(int)
        for method in self.learned_cols.values(): method_counts[method] += 1
        for method, count in method_counts.items(): print(f"   {method.upper()}: {count} columns")
        
        print("\n" + "📊 DETAILED COLUMN ANALYSIS" + "\n" + "-" * 40)
        for col, method in self.learned_cols.items():
            info = self.column_info.get(col, {})
            print(f"   {col}:\n"
                  f"      Method: {method.upper()}\n"
                  f"      Unique values: {info.get('unique_count', 'N/A')}\n"
                  f"      Target correlation: {info.get('correlation_with_target', 0):.3f}")
            if method == 'ohe': print(f"      Memory impact: {info.get('ohe_memory_mb', 0):.1f} MB")
            print()
        print("=" * 60)

    def get_encoding_info(self) -> Dict:
        return {
            'method': self.method, 'learned_columns': self.learned_cols,
            'encoding_stats': self.encoding_stats, 'column_info': self.column_info,
            'is_fitted': self.is_fitted
        }
    
    def get_summary_text(self) -> str:
        """Generates a formatted string summary for the HTML report."""
        if not self.is_fitted: return "<p>Encoder has not been fitted.</p>"
        
        summary_lines = []
        method_counts = defaultdict(int)
        for method in self.learned_cols.values(): method_counts[method] += 1
        
        summary_lines.append("<h4>Methods Used</h4><ul>")
        for method, count in method_counts.items():
            summary_lines.append(f"<li><b>{method.upper()}:</b> {count} columns</li>")
        summary_lines.append("</ul>")

        summary_lines.append("<h4>Detailed Analysis</h4><ul>")
        for col, method in self.learned_cols.items():
            info = self.column_info.get(col, {})
            corr_text = f" | Correlation: {info.get('correlation_with_target', 0):.3f}" if self.method == 'auto' else ""
            summary_lines.append(f"<li><b>{col}:</b> Method '{method.upper()}' (Unique Values: {info.get('unique_count', 'N/A')}{corr_text})</li>")
        summary_lines.append("</ul>")
        
        return "".join(summary_lines)
    
    def get_quality_report(self) -> Dict[str, any]:

        if not self.is_fitted:
            raise RuntimeError("Encoder belum di-fit. Jalankan .fit() atau .fit_transform() terlebih dahulu.")
        
        report = {
            'method_summary': dict(self.learned_cols),
            'column_details': {}
        }
        
        total_original_cols = len(self.learned_cols)
        total_new_cols = 0

        for col, method in self.learned_cols.items():
            info = self.column_info.get(col, {})
            original_cardinality = info.get('unique_count', 'N/A')
            
            new_cols_count = 0
            encoder = self.encoders.get(col)
            if not encoder:
                continue

            if method == 'ohe':
                if hasattr(encoder, 'get_feature_names_out'):
                    new_cols_count = len(encoder.get_feature_names_out([col]))
                else:
                    new_cols_count = original_cardinality
            elif method == 'binary':
                 new_cols_count = len(encoder.get_feature_names())
            elif method == 'hashing':
                 new_cols_count = encoder.n_components
            else: # label, target, ordinal
                new_cols_count = 1
            
            total_new_cols += new_cols_count

            report['column_details'][col] = {
                'method': method.upper(),
                'original_cardinality': original_cardinality,
                'new_features_created': new_cols_count,
            }

        efficiency_score = (total_new_cols - total_original_cols) / total_original_cols if total_original_cols > 0 else 0
        
        report['overall_summary'] = {
            'total_columns_encoded': total_original_cols,
            'total_features_created': total_new_cols,
            'dimensionality_change': f"{efficiency_score:+.2%}"
        }
        
        return report

    def plot_comparison(self, max_cols: int = 1):
        if not self.is_fitted or not self.learned_cols: return None
        
        col_to_plot = list(self.learned_cols.keys())[0]
        method_used = self.learned_cols[col_to_plot]
        
        df_before = self._original_df_snapshot
        df_after = self.transform(df_before.copy())

        color_before, color_after = '#58A6FF', '#F78166'
        bg_color, text_color = '#0D1117', '#C9D1D9'

        fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=bg_color)
        fig.suptitle(f"Encoding Comparison for '{col_to_plot}' (Method: {method_used.upper()})",
                     fontsize=20, color=text_color, weight='bold')

        # --- BEFORE: Horizontal Count Plot ---
        order = df_before[col_to_plot].value_counts().nlargest(20).index
        sns.countplot(y=df_before[col_to_plot], order=order, ax=axes[0], color=color_before)
        axes[0].set_title("Before: Top 20 Category Frequencies", color=text_color, fontsize=14)

        # --- AFTER: Adaptive Visualization ---
        axes[1].set_title("After: Encoded Result", color=text_color, fontsize=14)
        if method_used in ['ohe', 'binary', 'hashing']:
            new_cols = [c for c in df_after.columns if c.startswith(f'{col_to_plot}_')]
            if new_cols:
                sns.heatmap(df_after[new_cols].head(30), cmap="vlag", ax=axes[1], cbar=False)
                axes[1].set_ylabel("Sample Rows")
            else:
                axes[1].text(0.5, 0.5, "No new columns found for heatmap.", ha='center', va='center', color=text_color)
        else: # label, target, ordinal
            new_col_name = next((c for c in df_after.columns if col_to_plot in c), None)
            if new_col_name:
                sns.histplot(df_after[new_col_name], kde=True, ax=axes[1], color=color_after)
            else:
                axes[1].text(0.5, 0.5, "Encoded column not found.", ha='center', va='center', color=text_color)

        for ax in axes:
            ax.set_facecolor(bg_color)
            ax.tick_params(colors=text_color, which='both')
            for spine in ax.spines.values(): spine.set_edgecolor(text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig


    
    