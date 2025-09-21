import pandas as pd
import numpy as np
import pickle
import time
import logging
import warnings
from typing import Dict, Any, List, Union, Optional

# Impor untuk Laporan HTML & Mengelola Peringatan
import io
import base64
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# --- IMPOR BARU UNTUK MENAMPILKAN DI NOTEBOOK ---
from IPython.display import display, HTML

import optuna
import shap
optuna.logging.set_verbosity(optuna.logging.ERROR)

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# Model Libraries
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Integrasi Cleaner (opsional)
try:
    from noventis_beta.data_cleaner import NoventisDataCleaner
except ImportError:
    NoventisDataCleaner = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# --- Konfigurasi Model & Hyperparameter (TIDAK BERUBAH) ---
def get_rf_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
    }

def get_xgb_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
    }

def get_lgbm_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'is_unbalance': trial.suggest_categorical('is_unbalance', [True, False]),
    }

def get_dt_params(trial):
    return {
        'max_depth': trial.suggest_int('max_depth', 3, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
    }

def get_gb_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
    }
    
def get_catboost_params(trial):
    return {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
    }

def get_rf_reg_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
    }

def get_xgb_reg_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }

def get_lgbm_reg_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
    }

MODEL_CONFIG = {
    'classification': {
        'logistic_regression': {'model': LogisticRegression, 'params': None},
        'decision_tree': {'model': DecisionTreeClassifier, 'params': get_dt_params},
        'random_forest': {'model': RandomForestClassifier, 'params': get_rf_params},
        'gradient_boosting': {'model': GradientBoostingClassifier, 'params': get_gb_params},
        'xgboost': {'model': xgb.XGBClassifier, 'params': get_xgb_params},
        'lightgbm': {'model': lgb.LGBMClassifier, 'params': get_lgbm_params},
        'catboost': {'model': cb.CatBoostClassifier, 'params': get_catboost_params}
    },
    'regression': {
        'linear_regression': {'model': LinearRegression, 'params': None},
        'decision_tree': {'model': DecisionTreeRegressor, 'params': get_dt_params},
        'random_forest': {'model': RandomForestRegressor, 'params': get_rf_reg_params},
        'gradient_boosting': {'model': GradientBoostingRegressor, 'params': get_gb_params},
        'xgboost': {'model': xgb.XGBRegressor, 'params': get_xgb_reg_params},
        'lightgbm': {'model': lgb.LGBMRegressor, 'params': get_lgbm_reg_params},
        'catboost': {'model': cb.CatBoostRegressor, 'params': get_catboost_params}
    }
}


class ManualPredictor:
    def __init__(
        self,
        model_name: Union[str, List[str]],
        task: str,
        random_state: int = 42,
        data_cleaner: Optional[Any] = None,
        tune_hyperparameters: bool = False,
        n_trials: int = 50,
        cv_folds: int = 3,
        enable_feature_engineering: bool = False,
        cv_strategy: str = 'repeated',
        show_tuning_plots: bool = False
    ):
        self.model_name = model_name
        self.task = task.lower()
        self.random_state = random_state
        self.data_cleaner = data_cleaner
        self.tune_hyperparameters = tune_hyperparameters
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.enable_feature_engineering = enable_feature_engineering
        self.cv_strategy = cv_strategy
        self.show_tuning_plots = show_tuning_plots

        self.best_model_info = {}
        self.all_results = []
        self.X_train_final, self.X_test_final, self.y_test_final = None, None, None
        self.preprocessor = None # Atribut untuk menyimpan preprocessor

        if self.task not in ['classification', 'regression']:
            raise ValueError("Task must be 'classification' or 'regression'.")

    def _apply_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        logging.info("🔧 Menerapkan feature engineering (Polynomial & Interaction)...")
        numeric_cols = X.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            logging.warning("Tidak ada kolom numerik untuk feature engineering. Melewati...")
            return X
        
        X_numeric_clean = X[numeric_cols].copy()
        X_numeric_clean.fillna(X_numeric_clean.median(), inplace=True)
            
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        X_poly = poly.fit_transform(X_numeric_clean)
        
        poly_feature_names = poly.get_feature_names_out(numeric_cols)
        X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)
        
        X_final = X.drop(columns=numeric_cols).join(X_poly_df)
        logging.info(f"✅ Feature engineering selesai. Shape data baru: {X_final.shape}")
        return X_final

    def _show_tuning_insights(self, study: optuna.Study, model_name: str):
        logging.info(f"Menampilkan visualisasi tuning untuk {model_name.upper()}...")
        try:
            fig1 = optuna.visualization.plot_optimization_history(study)
            fig1.update_layout(title=f'Optimization History for {model_name}')
            fig1.show()

            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.update_layout(title=f'Parameter Importances for {model_name}')
            fig2.show()
        except (ValueError, ImportError) as e:
            logging.warning(f"Gagal membuat visualisasi tuning: {e}")

    def _load_single_model(self, name: str) -> Any:
        name = name.lower()
        config = MODEL_CONFIG[self.task].get(name)
        if config is None:
            raise ValueError(f"Model '{name}' tidak dikenali untuk task '{self.task}'.")
        
        model_class = config['model']
        params = {'random_state': self.random_state} if 'random_state' in model_class().get_params() else {}
        
        if name == 'catboost':
            params['verbose'] = 0
        if name == 'xgboost':
            params.update({
                'use_label_encoder': False,
                'eval_metric': 'logloss' if self.task == 'classification' else 'rmse',
                'verbosity': 0
            })
        if name == 'lightgbm':
            params['verbose'] = -1
            
        return model_class(**params)

    def _calculate_all_metrics(self, y_true, y_pred) -> Dict:
        if self.task == 'classification':
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0)
            }
        else:
            mse = mean_squared_error(y_true, y_pred)
            return {
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2_score': r2_score(y_true, y_pred)
            }

    def _tune_with_optuna(self, model_name: str, X_train, y_train) -> Dict:
        logging.info(f"🔬 Memulai hyperparameter tuning untuk {model_name.upper()}...")
        
        param_func = MODEL_CONFIG[self.task][model_name.lower()].get('params')
        if not param_func:
            logging.warning(f"Tidak ada search space untuk '{model_name}'. Menggunakan parameter default.")
            return {}

        def objective(trial):
            params = param_func(trial)
            model = self._load_single_model(model_name)
            model.set_params(**params)
            
            if self.cv_strategy == 'repeated':
                cv = RepeatedStratifiedKFold(n_splits=self.cv_folds, n_repeats=2, random_state=self.random_state)
            else: # stratified
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

            scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                preds = model.predict(X_val_fold)
                metric = 'f1_score' if self.task == 'classification' else 'r2_score'
                scores.append(self._calculate_all_metrics(y_val_fold, preds)[metric])
            return np.mean(scores)

        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(direction='maximize', pruner=pruner)
        study.optimize(objective, n_trials=self.n_trials, timeout=600, n_jobs=-1, show_progress_bar=False)
        
        if self.show_tuning_plots:
            self._show_tuning_insights(study, model_name)
        
        logging.info(f"✅ Tuning selesai. Parameter terbaik: {study.best_params}")
        return study.best_params

    def _run_single_model_pipeline(self, model_name: str, X_train, y_train, X_test, y_test) -> Dict:
        logging.info(f"--- Memproses model: {model_name.upper()} ---")
        
        best_params = {}
        if self.tune_hyperparameters:
            best_params = self._tune_with_optuna(model_name, X_train, y_train)

        model = self._load_single_model(model_name)
        if best_params:
            model.set_params(**best_params)

        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        logging.info(f"Pelatihan selesai dalam {training_time:.2f} detik.")
        
        predictions = model.predict(X_test)
        metrics = self._calculate_all_metrics(y_test, predictions)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        return {
            'model_name': model_name,
            'model_object': model,
            'predictions': predictions,
            'prediction_proba': y_pred_proba,
            'actual': y_test,
            'metrics': metrics,
            'training_time_seconds': training_time,
            'best_params': best_params
        }

    def run_pipeline(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        compare: bool = False,
        explain: bool = False,
        chosen_metric: Optional[str] = None,
        display_report: bool = False
    ) -> Dict:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        stratify = y if self.task == 'classification' else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify
        )
        
        if self.data_cleaner and NoventisDataCleaner is not None:
            logging.info("DataCleaner eksternal terdeteksi. Fitting dan transforming data...")
            self.data_cleaner.fit(X_train, y_train)
            X_train = self.data_cleaner.transform(X_train)
            X_test = self.data_cleaner.transform(X_test)
            logging.info("✅ Proses DataCleaner eksternal selesai.")
        
        logging.info("Menjalankan preprocessor internal untuk memastikan tipe data...")
        numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )

        self.preprocessor.fit(X_train)
        X_train_transformed = self.preprocessor.transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)
        logging.info("✅ Preprocessing internal selesai. Semua data sudah numerik.")
        
        if self.enable_feature_engineering:
            logging.warning("Feature engineering dinonaktifkan sementara untuk stabilitas.")

        self.X_train_final, self.X_test_final, self.y_test_final = X_train_transformed, X_test_transformed, y_test

        model_list = self.model_name if isinstance(self.model_name, list) else [self.model_name]
        self.all_results = []
        
        for name in model_list:
            try:
                result = self._run_single_model_pipeline(name, self.X_train_final, y_train, self.X_test_final, self.y_test_final)
                self.all_results.append(result)
            except Exception as e:
                logging.error(f"Gagal memproses model {name}: {e}")
                self.all_results.append({'model_name': name, 'metrics': {}, 'error': str(e)})

        successful_results = [res for res in self.all_results if 'error' not in res]
        if not successful_results:
            raise RuntimeError("Tidak ada model yang berhasil dilatih. Periksa kembali data atau konfigurasi.")

        primary_metric = 'f1_score' if self.task == 'classification' else 'r2_score'
        self.best_model_info = max(successful_results, key=lambda x: x['metrics'].get(primary_metric, -1))
        
        logging.info(f"\n--- Proses Selesai ---")
        logging.info(f"🏆 Model Terbaik: {self.best_model_info['model_name'].upper()} dengan {primary_metric} = {self.best_model_info['metrics'][primary_metric]:.4f}")

        if compare:
            self._print_comparison()
        if explain:
            self._create_metric_plot(chosen_metric)

        if display_report:
            self.display_report()

        return {
            'best_model_details': self.best_model_info,
            'all_model_results': self.all_results
        }

    def get_results_dataframe(self) -> pd.DataFrame:
        if not self.all_results:
            logging.warning("Tidak ada hasil. Jalankan .run_pipeline() dahulu.")
            return pd.DataFrame()
        
        records = [
            {'model': res['model_name'], **res['metrics']}
            for res in self.all_results if 'error' not in res
        ]
        df = pd.DataFrame(records).set_index('model')
        primary_metric = 'f1_score' if self.task == 'classification' else 'r2_score'
        is_higher_better = primary_metric in ['f1_score', 'r2_score', 'accuracy', 'precision', 'recall']
        return df.sort_values(by=primary_metric, ascending=not is_higher_better)

    def save_model(self, filepath: str):
        if not self.best_model_info:
            raise ValueError("Tidak ada model terbaik untuk disimpan.")
        
        model_to_save = self.best_model_info.get('model_object')
        logging.info(f"Menyimpan model '{self.best_model_info['model_name']}' ke {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(model_to_save, f)
        logging.info("✅ Model berhasil disimpan.")

    @staticmethod
    def load_model(filepath: str) -> Any:
        logging.info(f"Memuat model dari {filepath}...")
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logging.info("✅ Model berhasil dimuat.")
        return model
        
    def _print_comparison(self):
        if not self.all_results:
            return logging.warning("No results to compare.")
        
        print("\n" + "="*80 + "\n📊 MODEL COMPARISON - ALL METRICS\n" + "="*80)
        print(self.get_results_dataframe())
        print("="*80)

    def _create_metric_plot(self, chosen_metric: str = None):
        if not self.all_results:
            return logging.warning("No results to plot.")
        
        metric = chosen_metric or ('f1_score' if self.task == 'classification' else 'r2_score')
        
        df_results = self.get_results_dataframe().reset_index()
        if metric not in df_results.columns:
            return logging.error(f"Metric '{metric}' not available. Choices: {list(df_results.columns)}")
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(df_results['model'], df_results[metric], color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)
        plt.title(f'Model Comparison - {metric.upper()}', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Models', fontsize=12, fontweight='bold')
        plt.ylabel(metric.upper(), fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + (df_results[metric].max()*0.01), f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
        
        is_higher_better = metric in ['accuracy', 'precision', 'recall', 'f1_score', 'r2_score']
        best_val = df_results[metric].max() if is_higher_better else df_results[metric].min()
        best_bar = bars[df_results[df_results[metric] == best_val].index[0]]
        best_bar.set_color('gold')
        best_bar.set_edgecolor('darkorange')
        best_bar.set_linewidth(2)
        
        plt.tight_layout()
        plt.show()

    def explain_model(self, model_object=None, plot_type='summary', feature: Optional[str] = None):
        if self.X_test_final is None:
            raise RuntimeError("Jalankan .run_pipeline() terlebih dahulu.")
        
        model_to_explain = model_object or self.best_model_info.get('model_object')
        if not model_to_explain:
            raise ValueError("Tidak ada model untuk dijelaskan.")

        model_name = self.best_model_info.get('model_name', 'Model').upper()
        logging.info(f"Membuat SHAP Explainer untuk {model_name}...")

        try:
            X_test_df = pd.DataFrame(self.X_test_final.toarray() if hasattr(self.X_test_final, 'toarray') else self.X_test_final)
            if self.preprocessor:
                try:
                    feature_names = self.preprocessor.get_feature_names_out()
                    X_test_df.columns = feature_names
                except Exception:
                    logging.warning("Gagal mendapatkan nama fitur dari preprocessor.")
            
            explainer = shap.Explainer(model_to_explain, self.X_train_final)
            shap_values = explainer(X_test_df)
        except Exception as e:
            return logging.error(f"Gagal membuat SHAP explainer: {e}")
        
        logging.info(f"Membuat SHAP '{plot_type}' plot...")
        plt.figure()
        title = f"SHAP {plot_type.title()} Plot for {model_name}"
        
        if plot_type == 'summary':
            shap.summary_plot(shap_values, X_test_df, show=False)
        elif plot_type == 'beeswarm':
            shap.plots.beeswarm(shap_values, show=False)
        elif plot_type == 'dependence':
            if not feature:
                raise ValueError("Mohon sediakan argumen 'feature' untuk dependence plot.")
            shap.dependence_plot(feature, shap_values.values, X_test_df, interaction_index=None, show=False)
        else:
            logging.warning(f"Plot tipe '{plot_type}' belum didukung.")
        
        plt.title(title)
        plt.tight_layout()
        plt.show()

    # =======================================================================
    # ================== METODE BARU & MODIFIKASI LAPORAN ===================
    # =======================================================================

    def _get_summary_html(self) -> str:
        if not self.best_model_info:
            return "<p>Belum ada hasil untuk ditampilkan.</p>"
            
        primary_metric = 'f1_score' if self.task == 'classification' else 'r2_score'
        best_score = self.best_model_info['metrics'].get(primary_metric, 0)
        training_time = self.best_model_info.get('training_time_seconds', 0)
        best_params_str = str(self.best_model_info.get('best_params', 'Default'))
        
        return f"""
        <div class="grid-container">
            <div class="grid-item">
                <h4>Ringkasan Proses</h4>
                <p><strong>Tipe Task:</strong> {self.task.title()}</p>
                <p><strong>Jumlah Model Diuji:</strong> {len([res for res in self.all_results if 'error' not in res])}</p>
                <p><strong>Strategi Validasi Silang:</strong> {self.cv_strategy.title()}</p>
                <p><strong>Tuning Hyperparameters:</strong> {'Aktif' if self.tune_hyperparameters else 'Nonaktif'}</p>
            </div>
            <div class="grid-item score-card">
                <h4>🏆 Model Terbaik</h4>
                <p class="model-name">{self.best_model_info['model_name'].upper()}</p>
                <p class="metric-score">{primary_metric.replace('_', ' ').title()}: {best_score:.4f}</p>
            </div>
            <div class="grid-item">
                <h4>Detail Model Terbaik</h4>
                <p><strong>Waktu Pelatihan:</strong> {training_time:.2f} detik</p>
                <p><strong>Parameter Terbaik:</strong></p>
                <pre class="params-box">{best_params_str}</pre>
            </div>
        </div>
        """

    def _get_comparison_table_html(self) -> str:
        df_results = self.get_results_dataframe()
        if df_results.empty:
            return "<p>Tidak ada data perbandingan model.</p>"
        return df_results.to_html(classes='styled-table', border=0, float_format='{:.4f}'.format)

    def _create_metric_plot_for_html(self) -> plt.Figure:
        metric = 'f1_score' if self.task == 'classification' else 'r2_score'
        df_results = self.get_results_dataframe().reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(df_results['model'], df_results[metric], color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_title(f'Perbandingan Model - {metric.upper()}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + (df_results[metric].max()*0.01), f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
        
        best_val = df_results[metric].max()
        best_bar = bars[df_results[df_results[metric] == best_val].index[0]]
        best_bar.set_color('gold')
        best_bar.set_edgecolor('darkorange')
        best_bar.set_linewidth(2)
        
        plt.tight_layout()
        return fig

    def _get_plots_html(self) -> str:
        if self.X_test_final is None:
            return "<p>Plot tidak dapat dibuat karena pipeline belum dijalankan.</p>"
        
        plots_html = ""
        # Plot 1: Perbandingan Metrik
        try:
            fig_metric = self._create_metric_plot_for_html()
            buf = io.BytesIO()
            fig_metric.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plots_html += f'<h4>Perbandingan Kinerja Model</h4><div class="plot-container"><img src="data:image/png;base64,{img_base64}" alt="Metric Comparison Plot"></div>'
            plt.close(fig_metric)
        except Exception as e:
            plots_html += f"<p>Gagal membuat plot perbandingan metrik: {e}</p>"
        
        # Plot 2: SHAP Summary
        try:
            model_to_explain = self.best_model_info.get('model_object')
            model_name = self.best_model_info.get('model_name', 'Model').upper()
            
            X_test_df = pd.DataFrame(self.X_test_final.toarray() if hasattr(self.X_test_final, 'toarray') else self.X_test_final)
            if self.preprocessor:
                try:
                    feature_names = self.preprocessor.get_feature_names_out()
                    X_test_df.columns = feature_names
                except Exception:
                    pass

            explainer = shap.Explainer(model_to_explain, self.X_train_final)
            shap_values = explainer(X_test_df)
            
            shap.summary_plot(shap_values, X_test_df, show=False)
            fig_shap = plt.gcf()
            fig_shap.suptitle(f"Pentingnya Fitur (SHAP) untuk {model_name}", fontsize=16)
            
            buf_shap = io.BytesIO()
            fig_shap.savefig(buf_shap, format='png', bbox_inches='tight')
            buf_shap.seek(0)
            img_base64_shap = base64.b64encode(buf_shap.read()).decode('utf-8')
            plots_html += f'<br><h4>Pentingnya Fitur (SHAP)</h4><div class="plot-container"><img src="data:image/png;base64,{img_base64_shap}" alt="SHAP Summary Plot"></div>'
            plt.close(fig_shap)
        except Exception as e:
            plots_html += f"<p>Gagal membuat plot SHAP: {e}</p>"
            
        return plots_html

    def generate_html_report(self, filepath: Optional[str] = None) -> str:
        if not self.best_model_info:
            msg = "Laporan tidak dapat dibuat. Jalankan .run_pipeline() terlebih dahulu."
            logging.error(msg)
            return f"<p>{msg}</p>"

        if filepath:
            logging.info(f"Membuat laporan HTML di: {filepath}...")
        
        summary_html = self._get_summary_html()
        comparison_table_html = self._get_comparison_table_html()
        plots_html = self._get_plots_html()
        
        html_template = f"""
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Laporan Manual Predictor</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Exo+2:wght@700&display=swap');
        :root {{
            --bg-dark-1: #0D1117; --bg-dark-2: #161B22; --border-color: #30363D;
            --primary-blue: #58A6FF; --primary-orange: #F78166;
            --text-light: #C9D1D9; --text-medium: #8B949E;
        }}
        body {{
            font-family: 'Roboto', sans-serif; background-color: var(--bg-dark-1);
            color: var(--text-light); margin: 0; padding: 20px;
        }}
        .container {{
            max-width: 1200px; margin: auto; background-color: var(--bg-dark-2);
            border: 1px solid var(--border-color); border-radius: 8px; overflow: hidden;
        }}
        header {{
            padding: 30px; background-color: var(--bg-dark-1); text-align: center;
            border-bottom: 1px solid var(--border-color);
        }}
        header h1 {{ font-family: 'Exo 2', sans-serif; color: var(--primary-blue); margin: 0; }}
        .navbar {{ display: flex; background-color: var(--bg-dark-2); border-bottom: 1px solid var(--border-color); }}
        .nav-btn {{
            background: none; border: none; color: var(--text-medium);
            padding: 15px 25px; cursor: pointer; font-size: 16px;
            border-bottom: 3px solid transparent;
        }}
        .nav-btn:hover {{ color: var(--text-light); }}
        .nav-btn.active {{ color: var(--primary-orange); border-bottom-color: var(--primary-orange); }}
        .content-section {{ padding: 30px; display: none; }}
        .grid-container {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px; margin-bottom: 30px;
        }}
        .grid-item {{
            background-color: var(--bg-dark-1); padding: 20px;
            border-radius: 6px; border: 1px solid var(--border-color);
        }}
        .score-card {{
            text-align: center; background: linear-gradient(145deg, #1A2D40, #101820);
        }}
        .score-card .model-name {{
            font-family: 'Exo 2'; font-size: 2.5em;
            color: var(--primary-orange); margin: 10px 0;
        }}
        .score-card .metric-score {{ font-size: 1.5em; color: var(--text-light); margin: 0; }}
        .params-box {{
            background-color: #010409; padding: 10px; border-radius: 4px;
            font-family: monospace; font-size: 0.9em;
            white-space: pre-wrap; word-wrap: break-word;
        }}
        .styled-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        .styled-table th, .styled-table td {{
            padding: 12px 15px; text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        .styled-table thead th {{ background-color: #1A2D40; color: var(--primary-blue); }}
        .styled-table tbody tr:hover {{ background-color: #222b38; }}
        .plot-container {{
            background-color: var(--bg-dark-1); padding: 15px; border-radius: 6px;
            border: 1px solid var(--border-color); margin-top: 20px;
        }}
        .plot-container img {{ max-width: 100%; height: auto; display: block; margin: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Laporan Analisis Manual Predictor</h1>
            <p>Hasil detail dari pipeline pelatihan dan evaluasi model.</p>
        </header>
        <nav class="navbar">
            <button class="nav-btn active" onclick="showTab(event, 'summary')">Ringkasan</button>
            <button class="nav-btn" onclick="showTab(event, 'comparison')">Perbandingan Model</button>
            <button class="nav-btn" onclick="showTab(event, 'plots')">Visualisasi</button>
        </nav>
        <main>
            <section id="summary" class="content-section" style="display: block;">
                <h2>Ringkasan Eksekusi</h2>
                {summary_html}
            </section>
            <section id="comparison" class="content-section">
                <h2>Detail Perbandingan Metrik</h2>
                {comparison_table_html}
            </section>
            <section id="plots" class="content-section">
                <h2>Visualisasi Hasil</h2>
                {plots_html}
            </section>
        </main>
    </div>
    <script>
        function showTab(event, tabName) {{
            let i, sections, navbuttons;
            sections = document.getElementsByClassName("content-section");
            for (i = 0; i < sections.length; i++) {{
                sections[i].style.display = "none";
            }}
            navbuttons = document.getElementsByClassName("nav-btn");
            for (i = 0; i < navbuttons.length; i++) {{
                navbuttons[i].className = navbuttons[i].className.replace(" active", "");
            }}
            document.getElementById(tabName).style.display = "block";
            event.currentTarget.className += " active";
        }}
        document.addEventListener("DOMContentLoaded", function() {{
            if (document.querySelector('.nav-btn')) {{
               document.querySelector('.nav-btn').click();
            }}
        }});
    </script>
</body>
</html>
        """
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(html_template)
                logging.info(f"✅ Laporan HTML berhasil disimpan.")
            except Exception as e:
                logging.error(f"Gagal menyimpan laporan HTML: {e}")

        return html_template

    def display_report(self):
        """
        Menampilkan laporan HTML langsung di output cell Jupyter/Colab.
        """
        logging.info("Mempersiapkan laporan untuk ditampilkan di output...")
        try:
            html_content = self.generate_html_report() 
            display(HTML(html_content))
            logging.info("✅ Laporan berhasil ditampilkan.")
        except NameError:
             logging.warning("Tidak dapat menampilkan laporan. Pastikan Anda menjalankan ini di lingkungan Jupyter/Colab dan modul 'display', 'HTML' sudah diimpor.")
        except Exception as e:
            logging.error(f"Gagal menampilkan laporan: {e}")