import pandas as pd
import numpy as np
import pickle
import time
import logging
import warnings
import yaml
import os
from typing import Dict, Any, List, Union, Optional, Tuple

import io
import base64
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns

import optuna
import shap
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_curve
)
from sklearn.ensemble import StackingClassifier, StackingRegressor

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    ImbPipeline = Pipeline

from joblib import Memory
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.ERROR)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def get_rf_params(trial):
    return {
        'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 1000),
        'model__max_depth': trial.suggest_int('model__max_depth', 5, 50),
        'model__min_samples_split': trial.suggest_int('model__min_samples_split', 2, 20),
        'model__min_samples_leaf': trial.suggest_int('model__min_samples_leaf', 1, 10),
        'model__class_weight': trial.suggest_categorical('model__class_weight', ['balanced', None]),
    }

def get_xgb_params(trial):
    return {
        'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 2000),
        'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.3),
        'model__max_depth': trial.suggest_int('model__max_depth', 3, 12),
        'model__subsample': trial.suggest_float('model__subsample', 0.6, 1.0),
        'model__colsample_bytree': trial.suggest_float('model__colsample_bytree', 0.6, 1.0),
        'model__gamma': trial.suggest_float('model__gamma', 0, 5),
    }

def get_lgbm_params(trial):
    return {
        'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 2000),
        'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.3),
        'model__num_leaves': trial.suggest_int('model__num_leaves', 20, 150),
        'model__max_depth': trial.suggest_int('model__max_depth', 3, 15),
        'model__reg_alpha': trial.suggest_float('model__reg_alpha', 0.0, 1.0),
        'model__reg_lambda': trial.suggest_float('model__reg_lambda', 0.0, 1.0),
        'model__is_unbalance': trial.suggest_categorical('model__is_unbalance', [True, False]),
    }
    
def get_catboost_params(trial):
    return {
        'model__iterations': trial.suggest_int('model__iterations', 100, 1000),
        'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.3),
        'model__depth': trial.suggest_int('model__depth', 4, 10),
        'model__l2_leaf_reg': trial.suggest_float('model__l2_leaf_reg', 1.0, 10.0),
    }


def get_rf_reg_params(trial):
    return {
        'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 1000),
        'model__max_depth': trial.suggest_int('model__max_depth', 5, 50),
        'model__min_samples_split': trial.suggest_int('model__min_samples_split', 2, 20),
        'model__min_samples_leaf': trial.suggest_int('model__min_samples_leaf', 1, 10),
    }

def get_xgb_reg_params(trial):
    return {
        'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 2000),
        'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.3),
        'model__max_depth': trial.suggest_int('model__max_depth', 3, 12),
        'model__subsample': trial.suggest_float('model__subsample', 0.6, 1.0),
        'model__colsample_bytree': trial.suggest_float('model__colsample_bytree', 0.6, 1.0),
    }

def get_lgbm_reg_params(trial):
    return {
        'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 2000),
        'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.3),
        'model__num_leaves': trial.suggest_int('model__num_leaves', 20, 150),
        'model__max_depth': trial.suggest_int('model__max_depth', 3, 15),
    }

MODEL_CONFIG = {
    'classification': {
        'logistic_regression': {'model': LogisticRegression, 'params': None},
        'random_forest': {'model': RandomForestClassifier, 'params': get_rf_params},
        'xgboost': {'model': xgb.XGBClassifier, 'params': get_xgb_params},
        'lightgbm': {'model': lgb.LGBMClassifier, 'params': get_lgbm_params},
        'catboost': {'model': cb.CatBoostClassifier, 'params': get_catboost_params},

    },
    'regression': {
        'linear_regression': {'model': LinearRegression, 'params': None},
        'random_forest': {'model': RandomForestRegressor, 'params': get_rf_reg_params},
        'xgboost': {'model': xgb.XGBRegressor, 'params': get_xgb_reg_params},
        'lightgbm': {'model': lgb.LGBMRegressor, 'params': get_lgbm_reg_params},
        'catboost': {'model': cb.CatBoostRegressor, 'params': get_catboost_params},

    }
}
class ManualPredictor:
    def __init__(
        self,
        model_name: Union[str, List[str]],
        task: str,
        random_state: int = 42,
        tune_hyperparameters: bool = False,
        n_trials: int = 50,
        cv_folds: int = 3,
        imbalance_handler: Optional[str] = None,
        feature_engineering: Optional[List[str]] = None,
        show_tuning_plots: bool = False,
        enable_caching: bool = False,
        cache_dir: str = './.noventis_cache',
        use_mlflow: bool = False,
        mlflow_experiment_name: str = 'Noventis_Manual_Predictor'
    ):
        # --- Konfigurasi Pipeline ---
        self.model_name = model_name
        self.task = task.lower()
        self.random_state = random_state

        # --- Konfigurasi Proses ---
        self.tune_hyperparameters = tune_hyperparameters
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.show_tuning_plots = show_tuning_plots
        
        # --- Fitur Lanjutan ---
        self.imbalance_handler = imbalance_handler
        self.feature_engineering = feature_engineering or []
        self.enable_caching = enable_caching
        self.cache_dir = cache_dir
        
        # --- MLOps (MLflow) ---
        self.use_mlflow = use_mlflow
        self.mlflow_experiment_name = mlflow_experiment_name

        # --- Atribut Internal untuk Menyimpan Hasil ---
        self.best_model_info_ = {}
        self.all_results_ = []
        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = None, None, None, None
        self.recommendations_ = []
        self.is_fitted_ = False
        
        # --- Validasi & Setup ---
        self._validate_params()
        self._setup_cache()
        if self.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.set_experiment(self.mlflow_experiment_name)

    def _validate_params(self):
        if self.task not in ['classification', 'regression']:
            raise ValueError("Task harus 'classification' atau 'regression'.")
        if self.imbalance_handler and not IMBLEARN_AVAILABLE:
            raise ImportError("Paket 'imbalanced-learn' dibutuhkan untuk imbalance handling. Silakan install dengan 'pip install imbalanced-learn'.")
        if self.imbalance_handler and self.task == 'regression':
            logging.warning("Imbalance handler tidak relevan untuk task regresi dan akan diabaikan.")
            self.imbalance_handler = None
        if self.use_mlflow and not MLFLOW_AVAILABLE:
            logging.warning("Paket 'mlflow' tidak ditemukan. Logging dinonaktifkan. Install dengan 'pip install mlflow'.")
            self.use_mlflow = False

    def _setup_cache(self):
        if self.enable_caching:
            logging.info(f"Caching diaktifkan. Direktori cache: {self.cache_dir}")
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            self.memory = Memory(self.cache_dir, verbose=0)
        else:
            self.memory = None

    def _get_base_model(self, name: str) -> Any:
        name = name.lower()
        config = MODEL_CONFIG[self.task].get(name)
        if config is None:
            raise ValueError(f"Model '{name}' tidak dikenali untuk task '{self.task}'.")
        
        model_class = config['model']
        params = {'random_state': self.random_state} if 'random_state' in model_class().get_params() else {}
        
        # Nonaktifkan verbosity untuk model-model tertentu
        if name in ['catboost', 'xgboost', 'lightgbm']:
            params.update({'verbose': 0, 'verbosity': 0} if name != 'lightgbm' else {'verbose': -1})
        if name == 'xgboost':
            params.update({
                'use_label_encoder': False,
                'eval_metric': 'logloss' if self.task == 'classification' else 'rmse'
            })
            
        return model_class(**params)

    def _build_pipeline(self, model: Any, numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
        # Step 1: Preprocessor untuk imputasi, scaling, dan encoding
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )

        # Buat daftar langkah-langkah pipeline
        steps = [('preprocessor', preprocessor)]

        # Step 2 (Opsional): Imbalance Handling
        if self.imbalance_handler and self.task == 'classification':
            if self.imbalance_handler == 'smote':
                handler = SMOTE(random_state=self.random_state)
            elif self.imbalance_handler == 'random_oversampler':
                handler = RandomOverSampler(random_state=self.random_state)
            else:
                raise ValueError(f"Imbalance handler '{self.imbalance_handler}' tidak didukung.")
            steps.append(('resampler', handler))

        # Step 3 (Opsional): Feature Engineering
        if 'polynomial' in self.feature_engineering:
            # Note: PolynomialFeatures harus dijalankan setelah imputasi & scaling
            # Ini adalah contoh sederhana; implementasi yang lebih canggih mungkin diperlukan
            # Untuk sekarang, kita akan membuat pipeline terpisah untuk ini
            # TODO: Integrasikan FE lebih baik ke dalam preprocessor utama
            pass # Implementasi lebih lanjut diperlukan untuk integrasi yang mulus

        # Step 4: Model
        steps.append(('model', model))

        # Pilih tipe pipeline (ImbPipeline menangani resampler dengan benar)
        pipeline_class = ImbPipeline if self.imbalance_handler else Pipeline
        return pipeline_class(steps, memory=self.memory)

    def _calculate_all_metrics(self, y_true, y_pred, y_proba=None) -> Dict:
        if self.task == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            if y_proba is not None and len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            return metrics
        else: # Regression
            mse = mean_squared_error(y_true, y_pred)
            return {
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2_score': r2_score(y_true, y_pred)
            }
            
    def _check_imbalance(self, y: pd.Series, threshold: float = 0.4):
        """Mendeteksi jika dataset tidak seimbang."""
        if self.task != 'classification':
            return
        
        counts = y.value_counts(normalize=True)
        if len(counts) > 1 and counts.min() < threshold:
            minority_class = counts.idxmin()
            minority_percentage = counts.min() * 100
            logging.warning(f"Potensi imbalance terdeteksi. Kelas minoritas '{minority_class}' hanya {minority_percentage:.2f}% dari data.")
            if not self.imbalance_handler:
                self.recommendations_.append(
                    "Dataset Anda terlihat tidak seimbang. Pertimbangkan untuk menggunakan parameter `imbalance_handler='smote'` "
                    "untuk meningkatkan performa model pada kelas minoritas."
                )

    def _tune_with_optuna(self, pipeline: Pipeline, model_name: str, X_train, y_train) -> Dict:
        logging.info(f"🔬 Memulai hyperparameter tuning untuk {model_name.upper()}...")
        
        param_func = MODEL_CONFIG[self.task][model_name.lower()].get('params')
        if not param_func:
            logging.warning(f"Tidak ada search space untuk '{model_name}'. Menggunakan parameter default.")
            return {}

        def objective(trial):
            params = param_func(trial)
            pipeline.set_params(**params)
            
            # Tambahkan early stopping untuk model yang mendukung
            fit_params = {}
            if 'lgbm' in model_name and 'model__n_estimators' in params:
                fit_params['model__callbacks'] = [lgb.early_stopping(15, verbose=False)]
            elif 'xgb' in model_name and 'model__n_estimators' in params:
                 fit_params['model__early_stopping_rounds'] = 15
            
            metric = 'f1_weighted' if self.task == 'classification' else 'r2'
            
            # Lakukan cross-validation pada seluruh pipeline
            score = cross_val_score(pipeline, X_train, y_train, n_jobs=-1, cv=self.cv_folds, scoring=metric, fit_params=fit_params).mean()
            return score

        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=self.n_trials, timeout=600)
        
        if self.show_tuning_plots:
            optuna.visualization.plot_optimization_history(study).show()
            optuna.visualization.plot_param_importances(study).show()
        
        logging.info(f"✅ Tuning selesai. Parameter terbaik: {study.best_params}")
        return study.best_params
    
    def run_pipeline(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        display_report: bool = False
    ) -> Dict:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        stratify = y if self.task == 'classification' else None
        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify
        )
        
        self._check_imbalance(self.y_train_)

        numeric_features = self.X_train_.select_dtypes(include=np.number).columns.tolist()
        categorical_features = self.X_train_.select_dtypes(exclude=np.number).columns.tolist()
        
        model_list = self.model_name if isinstance(self.model_name, list) else [self.model_name]
        self.all_results_ = []
        
        parent_run_id = None
        if self.use_mlflow:
             # Mulai parent run jika menjalankan lebih dari satu model
            if len(model_list) > 1:
                parent_run = mlflow.start_run(run_name=f"Pipeline Run - {int(time.time())}", nested=False)
                parent_run_id = parent_run.info.run_id
                mlflow.log_param("model_list", model_list)
                mlflow.log_param("task", self.task)

        for name in model_list:
            run_name = f"{name} - {int(time.time())}"
            try:
                with mlflow.start_run(run_name=run_name, nested=True) if self.use_mlflow else self._null_context_manager():
                    if self.use_mlflow:
                        mlflow.log_param("model_name", name)
                        mlflow.log_param("imbalance_handler", self.imbalance_handler)

                    logging.info(f"--- Memproses model: {name.upper()} ---")
                    base_model = self._get_base_model(name)
                    pipeline = self._build_pipeline(base_model, numeric_features, categorical_features)

                    best_params = {}
                    if self.tune_hyperparameters:
                        best_params = self._tune_with_optuna(pipeline, name, self.X_train_, self.y_train_)
                        pipeline.set_params(**best_params)
                        if self.use_mlflow:
                            mlflow.log_params({k: v for k, v in best_params.items()})

                    start_time = time.time()
                    pipeline.fit(self.X_train_, self.y_train_)
                    training_time = time.time() - start_time
                    
                    predictions = pipeline.predict(self.X_test_)
                    y_proba = pipeline.predict_proba(self.X_test_) if hasattr(pipeline, 'predict_proba') else None
                    
                    metrics = self._calculate_all_metrics(self.y_test_, predictions, y_proba)
                    if self.use_mlflow:
                        mlflow.log_metrics(metrics)
                        mlflow.sklearn.log_model(pipeline, "model")

                    self.all_results_.append({
                        'model_name': name,
                        'pipeline_object': pipeline,
                        'predictions': predictions,
                        'prediction_proba': y_proba,
                        'actual': self.y_test_,
                        'metrics': metrics,
                        'training_time_seconds': training_time,
                        'best_params': best_params
                    })
                    logging.info(f"✅ Selesai: {name.upper()} | Metrik utama: {list(metrics.values())[3 if self.task=='classification' else -1]:.4f}")

            except Exception as e:
                logging.error(f"Gagal memproses model {name}: {e}")
                self.all_results_.append({'model_name': name, 'metrics': {}, 'error': str(e)})

        if parent_run_id and self.use_mlflow:
            mlflow.end_run()

        successful_results = [res for res in self.all_results_ if 'error' not in res]
        if not successful_results:
            raise RuntimeError("Tidak ada model yang berhasil dilatih.")

        primary_metric = 'f1_score' if self.task == 'classification' else 'r2_score'
        self.best_model_info_ = max(successful_results, key=lambda x: x['metrics'].get(primary_metric, -1))
        self.is_fitted_ = True
        
        logging.info(f"\n--- Proses Selesai ---")
        logging.info(f"🏆 Model Terbaik: {self.best_model_info_['model_name'].upper()} dengan {primary_metric} = {self.best_model_info_['metrics'][primary_metric]:.4f}")

        if display_report:
            self.display_report()

        return {
            'best_model_details': self.best_model_info_,
            'all_model_results': self.all_results_
        }
    
    # --- Metode Publik Baru & Tambahan ---
    
    def optimize_threshold(self, metric: str = 'f1_score', steps: int = 100) -> Tuple[float, float]:
        if not self.is_fitted_ or self.task != 'classification':
            raise RuntimeError("Metode ini hanya untuk task klasifikasi setelah .run_pipeline() dijalankan.")
        
        pipeline = self.best_model_info_['pipeline_object']
        y_proba = pipeline.predict_proba(self.X_test_)[:, 1]
        
        best_threshold, best_score = 0, -1
        
        if metric == 'f1_score':
            precisions, recalls, thresholds = precision_recall_curve(self.y_test_, y_proba)
            f1_scores = (2 * precisions * recalls) / (precisions + recalls)
            f1_scores = f1_scores[~np.isnan(f1_scores)]
            best_score = np.max(f1_scores)
            best_threshold = thresholds[np.argmax(f1_scores)]
        else:
            # Implementasi untuk metrik lain jika perlu
            raise ValueError(f"Metrik '{metric}' belum didukung untuk optimisasi threshold.")
            
        logging.info(f"Threshold optimal ditemukan: {best_threshold:.4f} (Meningkatkan {metric} menjadi {best_score:.4f})")
        return best_threshold, best_score

    def stack_models(self, model_names: Optional[List[str]] = None, top_n: int = 3, final_estimator_name: str = 'logistic_regression'):
        if not self.is_fitted_:
            raise RuntimeError("Jalankan .run_pipeline() terlebih dahulu.")
        
        if model_names:
            estimators_info = [res for res in self.all_results_ if res['model_name'] in model_names]
        else:
            estimators_info = sorted(self.all_results_, key=lambda x: x['metrics'].get('f1_score' if self.task=='classification' else 'r2_score', -1), reverse=True)[:top_n]
        
        estimators = [(info['model_name'], info['pipeline_object']) for info in estimators_info]
        final_estimator = self._get_base_model(final_estimator_name)

        logging.info(f"Membangun Stacking model dengan: {[e[0] for e in estimators]}")
        
        if self.task == 'classification':
            stacking_model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=self.cv_folds)
        else:
            stacking_model = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=self.cv_folds)
            
        logging.info("Melatih Stacking model...")
        stacking_model.fit(self.X_train_, self.y_train_)
        
        # Evaluasi
        preds = stacking_model.predict(self.X_test_)
        proba = stacking_model.predict_proba(self.X_test_) if hasattr(stacking_model, 'predict_proba') else None
        metrics = self.calculate_all_metrics(self.y_test_, preds, proba)
        
        logging.info(f"✅ Stacking Selesai. Metrik: {metrics}")
        return stacking_model, metrics

    # --- Metode Penyimpanan, Pemuatan & Helper ---

    def save_model(self, filepath: str):
        if not self.is_fitted_:
            raise ValueError("Tidak ada model terbaik untuk disimpan. Jalankan .run_pipeline() dahulu.")
        
        pipeline_to_save = self.best_model_info_.get('pipeline_object')
        logging.info(f"Menyimpan pipeline lengkap '{self.best_model_info_['model_name']}' ke {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_to_save, f)
        logging.info("✅ Pipeline berhasil disimpan.")

    @staticmethod
    def load_model(filepath: str) -> Pipeline:
        logging.info(f"Memuat pipeline dari {filepath}...")
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
        logging.info("✅ Pipeline berhasil dimuat.")
        return pipeline

    def explain_model(self, pipeline_object=None, plot_type='summary'):
        if not self.is_fitted_:
            raise RuntimeError("Jalankan .run_pipeline() terlebih dahulu.")
        
        pipeline_to_explain = pipeline_object or self.best_model_info_.get('pipeline_object')
        model = pipeline_to_explain.named_steps['model']
        preprocessor = pipeline_to_explain.named_steps['preprocessor']

        logging.info(f"Membuat SHAP Explainer untuk model di dalam pipeline...")
        
        # Transformasi data test untuk explainer
        X_test_transformed = preprocessor.transform(self.X_test_)
        
        # SHAP memerlukan DataFrame untuk nama fitur yang bagus
        try:
            feature_names = preprocessor.get_feature_names_out()
            X_test_transformed_df = pd.DataFrame(X_test_transformed.toarray(), columns=feature_names)
        except:
            X_test_transformed_df = pd.DataFrame(X_test_transformed)
        
        explainer = shap.Explainer(model, X_test_transformed_df)
        shap_values = explainer(X_test_transformed_df)
        
        plt.figure()
        if plot_type == 'summary':
            shap.summary_plot(shap_values, X_test_transformed_df, show=False)
        elif plot_type == 'beeswarm':
             shap.plots.beeswarm(shap_values, show=False)
        else:
             logging.warning(f"Plot tipe '{plot_type}' belum didukung.")

        plt.tight_layout()
        plt.show()

    # --- Metode Laporan HTML (Sudah Dioptimalkan) ---

    def generate_html_report(self, filepath: Optional[str] = None) -> str:
        # ... (Kode HTML template dan CSS tetap sama) ...
        # Perubahan utama ada pada fungsi-fungsi yang dipanggil di bawah ini
        
        summary_html = self._get_summary_html()
        recommendations_html = self._get_recommendations_html()
        comparison_table_html = self._get_comparison_table_html()
        plots_html = self._get_plots_html()
        diagnostics_html = self._get_diagnostics_html()
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="id">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Laporan Manual Predictor</title>
            <style>
                /* ... CSS LENGKAP ANDA DI SINI ... */
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
                    <button class="nav-btn" onclick="showTab(event, 'recommendations')">Rekomendasi</button>
                    <button class="nav-btn" onclick="showTab(event, 'comparison')">Perbandingan</button>
                    <button class="nav-btn" onclick="showTab(event, 'plots')">Visualisasi</button>
                    <button class="nav-btn" onclick="showTab(event, 'diagnostics')">Diagnostik</button>
                </nav>
                <main>
                    <section id="summary" class="content-section" style="display: block;">
                        <h2>Ringkasan Eksekusi</h2>
                        {summary_html}
                    </section>
                    <section id="recommendations" class="content-section">
                        <h2>💡 Rekomendasi Otomatis</h2>
                        {recommendations_html}
                    </section>
                    <section id="comparison" class="content-section">
                        <h2>Detail Perbandingan Metrik</h2>
                        {comparison_table_html}
                    </section>
                    <section id="plots" class="content-section">
                        <h2>Visualisasi Hasil</h2>
                        {plots_html}
                    </section>
                    <section id="diagnostics" class="content-section">
                        <h2>Diagnostik Model Terbaik</h2>
                        {diagnostics_html}
                    </section>
                </main>
            </div>
            <script>
                // ... (JavaScript Anda dari kode sebelumnya diletakkan di sini) ...
            </script>
        </body>
        </html>
        """
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f: f.write(html_template)
            logging.info(f"✅ Laporan HTML berhasil disimpan di {filepath}")

        return html_template
    
    def _get_summary_html(self) -> str:
        # Sedikit dimodifikasi untuk menampilkan parameter baru
        best_model_info = self.best_model_info_
        primary_metric = 'f1_score' if self.task == 'classification' else 'r2_score'
        
        return f"""
        <div class="grid-container">
            <div class="grid-item">
                <h4>Konfigurasi Pipeline</h4>
                <p><strong>Tipe Task:</strong> {self.task.title()}</p>
                <p><strong>Model Diuji:</strong> {str([res['model_name'] for res in self.all_results_ if 'error' not in res])}</p>
                <p><strong>Imbalance Handler:</strong> {self.imbalance_handler or 'Tidak Aktif'}</p>
                <p><strong>Feature Engineering:</strong> {str(self.feature_engineering) if self.feature_engineering else 'Tidak Aktif'}</p>
            </div>
            <div class="grid-item score-card">
                <h4>🏆 Model Terbaik</h4>
                <p class="model-name">{best_model_info['model_name'].upper()}</p>
                <p class="metric-score">{primary_metric.replace('_', ' ').title()}: {best_model_info['metrics'].get(primary_metric, 0):.4f}</p>
            </div>
            <div class="grid-item">
                <h4>Detail Eksekusi</h4>
                <p><strong>Tuning Hyperparameters:</strong> {'Aktif' if self.tune_hyperparameters else 'Nonaktif'}</p>
                <p><strong>Waktu Pelatihan Terbaik:</strong> {best_model_info.get('training_time_seconds', 0):.2f} detik</p>
                <p><strong>MLflow Logging:</strong> {'Aktif' if self.use_mlflow else 'Nonaktif'}</p>
            </div>
        </div>
        """
        
    def _get_recommendations_html(self) -> str:
        if not self.recommendations_:
            return "<p>Tidak ada rekomendasi spesifik untuk proses ini. Hasil terlihat bagus!</p>"
        
        html = "<ul>"
        for rec in self.recommendations_:
            html += f"<li class='recommendation-item'>{rec}</li>"
        html += "</ul>"
        return html
        
    def _get_diagnostics_html(self) -> str:
        """Membuat plot Learning Curve untuk model terbaik."""
        try:
            pipeline = self.best_model_info_['pipeline_object']
            model_name = self.best_model_info_['model_name'].upper()
            
            train_sizes, train_scores, test_scores = learning_curve(
                pipeline, self.X_train_, self.y_train_, cv=self.cv_folds, n_jobs=-1, 
                train_sizes=np.linspace(.1, 1.0, 5)
            )

            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.grid()
            ax.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                            np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1, color="r")
            ax.fill_between(train_sizes, np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                            np.mean(test_scores, axis=1) + np.std(test_scores, axis=1), alpha=0.1, color="g")
            ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
            ax.set_title(f'Learning Curve untuk {model_name}')
            ax.set_xlabel("Training examples")
            ax.set_ylabel("Score")
            ax.legend(loc="best")
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return f'<div class="plot-container"><img src="data:image/png;base64,{img_base64}" alt="Learning Curve Plot"></div>'

        except Exception as e:
            return f"<p>Gagal membuat plot diagnostik: {e}</p>"
            
    # Metode lainnya (_get_comparison_table_html, _get_plots_html, display_report, etc.)
    # sebagian besar tetap sama, hanya perlu memastikan mereka menggunakan `self.all_results_`
    # dan `self.best_model_info_`
    def get_results_dataframe(self) -> pd.DataFrame:
        if not self.all_results_:
            logging.warning("Tidak ada hasil. Jalankan .run_pipeline() dahulu.")
            return pd.DataFrame()
        
        records = [
            {'model': res['model_name'], **res['metrics']}
            for res in self.all_results_ if 'error' not in res
        ]
        df = pd.DataFrame(records).set_index('model')
        primary_metric = 'f1_score' if self.task == 'classification' else 'r2_score'
        is_higher_better = primary_metric in ['f1_score', 'r2_score', 'accuracy', 'precision', 'recall', 'roc_auc']
        return df.sort_values(by=primary_metric, ascending=not is_higher_better)

    def _get_comparison_table_html(self) -> str:
        df_results = self.get_results_dataframe()
        if df_results.empty:
            return "<p>Tidak ada data perbandingan model.</p>"
        return df_results.to_html(classes='styled-table', border=0, float_format='{:.4f}'.format)

    def _get_plots_html(self) -> str:
        # ... (Kode ini hampir tidak berubah, hanya memastikan variabelnya benar) ...
        # Contohnya:
        try:
            # ... (kode plot perbandingan metrik dari kode sebelumnya) ...
            # ... (kode plot SHAP dari kode sebelumnya) ...
            return "..."
        except Exception as e:
            return f"<p>Gagal membuat plot: {e}</p>"

    def display_report(self):
        logging.info("Mempersiapkan laporan untuk ditampilkan di output...")
        html_content = self.generate_html_report() 
        display(HTML(html_content))
        logging.info("✅ Laporan berhasil ditampilkan.")

    def _null_context_manager(self):
        # Helper untuk blok 'with' saat mlflow dinonaktifkan
        from contextlib import contextmanager
        @contextmanager
        def null_manager():
            yield None
        return null_manager()