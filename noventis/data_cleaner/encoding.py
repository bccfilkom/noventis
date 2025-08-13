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

    def plot_comparison(self, X: pd.DataFrame, max_cols: int = 3):

        if not self.is_fitted:
            raise RuntimeError("Encoder harus di-fit terlebih dahulu sebelum transform.")

        print("Membuat visualisasi perbandingan untuk encoding...")
        
        transformed_data = self.transform(X.copy())

        cols_to_plot = [col for col, method in self.learned_cols.items() if method in ['label', 'ohe', 'ordinal']][:max_cols]

        if not cols_to_plot:
            print("Tidak ada kolom dengan metode encoding yang cocok untuk divisualisasikan (misal: label, ohe).")
            return

        for col in cols_to_plot:
            plt.figure(figsize=(10, 5))
            
            X[col].value_counts().nlargest(15).plot(kind='bar', color='skyblue', alpha=0.7)
            
            plt.title(f"Distribusi 15 Kategori Teratas untuk '{col}' (Sebelum Encoding)")
            plt.ylabel("Frekuensi")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()


    
    