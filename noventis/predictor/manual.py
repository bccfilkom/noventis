import pandas as pd
import numpy as np
import pickle
import time
import logging
from typing import Dict, Any, List, Union, Optional
import matplotlib.pyplot as plt

import optuna
import shap
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# Model Libraries
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from sklearn.model_selection import train_test_split
# Tambahkan import ini bersama import sklearn lainnya
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

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

# --- Konfigurasi Model & Hyperparameter ---
def get_rf_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000), 'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20), 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
    }

def get_xgb_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12), 'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), 'gamma': trial.suggest_float('gamma', 0, 5),
    }

def get_lgbm_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150), 'max_depth': trial.suggest_int('max_depth', 3, 15),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0), 'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'is_unbalance': trial.suggest_categorical('is_unbalance', [True, False]),
    }

def get_dt_params(trial):
    return {
        'max_depth': trial.suggest_int('max_depth', 3, 50), 'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
    }

def get_gb_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10), 'subsample': trial.suggest_float('subsample', 0.7, 1.0),
    }
    
def get_catboost_params(trial):
    return {
        'iterations': trial.suggest_int('iterations', 100, 1000), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10), 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
    }

def get_rf_reg_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000), 'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20), 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
    }

def get_xgb_reg_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12), 'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }

def get_lgbm_reg_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150), 'max_depth': trial.suggest_int('max_depth', 3, 15),
    }

MODEL_CONFIG = {
    'classification': {
        'logistic_regression': {'model': LogisticRegression, 'params': None}, 'decision_tree': {'model': DecisionTreeClassifier, 'params': get_dt_params},
        'random_forest': {'model': RandomForestClassifier, 'params': get_rf_params}, 'gradient_boosting': {'model': GradientBoostingClassifier, 'params': get_gb_params},
        'xgboost': {'model': xgb.XGBClassifier, 'params': get_xgb_params}, 'lightgbm': {'model': lgb.LGBMClassifier, 'params': get_lgbm_params},
        'catboost': {'model': cb.CatBoostClassifier, 'params': get_catboost_params}
    },
    'regression': {
        'linear_regression': {'model': LinearRegression, 'params': None}, 'decision_tree': {'model': DecisionTreeRegressor, 'params': get_dt_params},
        'random_forest': {'model': RandomForestRegressor, 'params': get_rf_reg_params}, 'gradient_boosting': {'model': GradientBoostingRegressor, 'params': get_gb_params},
        'xgboost': {'model': xgb.XGBRegressor, 'params': get_xgb_reg_params}, 'lightgbm': {'model': lgb.LGBMRegressor, 'params': get_lgbm_reg_params},
        'catboost': {'model': cb.CatBoostRegressor, 'params': get_catboost_params}
    }
}

class ManualPredictor:
    def __init__(self,
                 model_name: Union[str, List[str]],
                 task: str,
                 random_state: int = 42,
                 data_cleaner: Optional[Any] = None,
                 tune_hyperparameters: bool = False,
                 n_trials: int = 50,
                 cv_folds: int = 3,
                 # --- Parameter Baru ---
                 enable_feature_engineering: bool = False,
                 cv_strategy: str = 'repeated', # Pilihan: 'stratified', 'repeated'
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

        if self.task not in ['classification', 'regression']:
            raise ValueError("Task must be 'classification' or 'regression'.")

    # def _apply_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
    #     logging.info("🔧 Menerapkan feature engineering (Polynomial & Interaction)...")
    #     poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        
    #     numeric_cols = X.select_dtypes(include=np.number).columns
    #     if len(numeric_cols) == 0:
    #         logging.warning("Tidak ada kolom numerik untuk feature engineering. Melewati...")
    #         return X
        
    #     X_numeric_clean = X[numeric_cols].copy()
    #     X_numeric_clean.fillna(X_numeric_clean.median(), inplace=True)
            
    #     X_poly = poly.fit_transform(X[numeric_cols])
        
    #     poly_feature_names = poly.get_feature_names_out(numeric_cols)
    #     X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)
        
    #     X_final = X.drop(columns=numeric_cols).join(X_poly_df)
    #     logging.info(f"✅ Feature engineering selesai. Shape data baru: {X_final.shape}")
    #     return X_final


    def _apply_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
            logging.info("🔧 Menerapkan feature engineering (Polynomial & Interaction)...")
            
            numeric_cols = X.select_dtypes(include=np.number).columns
            if len(numeric_cols) == 0:
                logging.warning("Tidak ada kolom numerik untuk feature engineering. Melewati...")
                return X
            
            # Buat salinan bersih dan isi nilai NaN
            X_numeric_clean = X[numeric_cols].copy()
            X_numeric_clean.fillna(X_numeric_clean.median(), inplace=True)
                
            # Perbaikan: Gunakan X_numeric_clean yang sudah dijamin bebas NaN
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            X_poly = poly.fit_transform(X_numeric_clean) # <-- DIUBAH
            
            poly_feature_names = poly.get_feature_names_out(numeric_cols)
            X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)
            
            # Gabungkan kembali dengan kolom non-numerik asli
            X_non_numeric = X.drop(columns=numeric_cols)
            X_final = X_non_numeric.join(X_poly_df)
            
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
        if name in ['xgboost', 'catboost', 'lightgbm']:
            params['verbose'] = 0
        if name == 'xgboost':
             params.update({'use_label_encoder': False, 'eval_metric': 'logloss' if self.task == 'classification' else 'rmse'})
        return model_class(**params)

    def _calculate_all_metrics(self, y_true, y_pred) -> Dict:
        if self.task == 'classification':
            return {'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0), 'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)}
        else:
            mse = mean_squared_error(y_true, y_pred)
            return {'mae': mean_absolute_error(y_true, y_pred), 'mse': mse, 'rmse': np.sqrt(mse), 'r2_score': r2_score(y_true, y_pred)}

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
            else:
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
        study.optimize(objective, n_trials=self.n_trials, timeout=600, n_jobs=-1)
        
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
            'model_name': model_name, 'model_object': model, 'predictions': predictions,
            'prediction_proba': y_pred_proba, 'actual': y_test, 'metrics': metrics,
            'training_time_seconds': training_time
        }

    def run_pipeline(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2, 
                     compare: bool = False, explain: bool = False, chosen_metric: str = None) -> Dict:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # --- PERBAIKAN ALUR KERJA ---
        # 1. SPLIT DATA MENTAH TERLEBIH DAHULU UNTUK MENCEGAH DATA LEAKAGE
        stratify = y if self.task == 'classification' else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify
        )
        
        # 2. JALANKAN DATACLEANER EKSTERNAL JIKA ADA
        if self.data_cleaner and NoventisDataCleaner is not None:
            logging.info("DataCleaner eksternal terdeteksi. Fitting dan transforming data...")
            self.data_cleaner.fit(X_train, y_train)
            X_train = self.data_cleaner.transform(X_train)
            X_test = self.data_cleaner.transform(X_test)
            logging.info("✅ Proses DataCleaner eksternal selesai.")
        
        # 3. SOLUSI: TAMBAHKAN PREPROCESSOR INTERNAL YANG ANDAL
        logging.info("Menjalankan preprocessor internal untuk memastikan tipe data...")
        
        # Identifikasi tipe kolom berdasarkan data latih
        numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

        # Buat pipeline preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough' # Biarkan kolom lain yang mungkin sudah numerik
        )

        # Fit preprocessor HANYA pada data latih
        preprocessor.fit(X_train)
        
        # Transformasi data latih dan data uji
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)
        logging.info("✅ Preprocessing internal selesai. Semua data sudah numerik.")
        
        # 4. FEATURE ENGINEERING (JIKA DIAKTIFKAN)
        if self.enable_feature_engineering:
            # Catatan: Feature engineering pada data yang sudah di-encode bisa menghasilkan banyak sekali fitur.
            # Sebaiknya dijalankan sebelum encoding jika memungkinkan.
            # Untuk sekarang, kita nonaktifkan sementara untuk memastikan pipeline berjalan.
            logging.warning("Feature engineering dinonaktifkan sementara saat menggunakan preprocessor internal untuk stabilitas.")
            pass # X = self._apply_feature_engineering(X) -> memerlukan refactor lebih lanjut

        # Simpan data final untuk digunakan oleh SHAP nanti
        self.X_train_final, self.X_test_final, self.y_test_final = X_train, X_test, y_test

        model_list = self.model_name if isinstance(self.model_name, list) else [self.model_name]
        self.all_results = []
        
        for name in model_list:
            try:
                result = self._run_single_model_pipeline(name, X_train, y_train, X_test, y_test)
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

        if compare: self._print_comparison()
        if explain: self._create_metric_plot(chosen_metric)

        return {'best_model_details': self.best_model_info, 'all_model_results': self.all_results}

    def _print_comparison(self):
        if not self.all_results: return logging.warning("No results to compare.")
        print("\n" + "="*80 + "\n📊 MODEL COMPARISON - ALL METRICS\n" + "="*80)
        print(self.get_results_dataframe())
        print("="*80)

    def _create_metric_plot(self, chosen_metric: str = None):
        if not self.all_results: return logging.warning("No results to plot.")
        
        metric = chosen_metric or ('f1_score' if self.task == 'classification' else 'r2_score')
        
        df_results = self.get_results_dataframe().reset_index()
        if metric not in df_results.columns:
            return logging.error(f"Metric '{metric}' not available. Choices: {list(df_results.columns)}")
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(df_results['model'], df_results[metric], color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)
        plt.title(f'Model Comparison - {metric.upper()}', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Models', fontsize=12, fontweight='bold'); plt.ylabel(metric.upper(), fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right'); plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + (df_results[metric].max()*0.01), f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
        
        is_higher_better = metric in ['accuracy', 'precision', 'recall', 'f1_score', 'r2_score']
        best_val = df_results[metric].max() if is_higher_better else df_results[metric].min()
        best_bar = bars[df_results[df_results[metric] == best_val].index[0]]
        best_bar.set_color('gold'); best_bar.set_edgecolor('darkorange'); best_bar.set_linewidth(2)
        
        plt.tight_layout(); plt.show()

    def explain_model(self, model_object=None, plot_type='summary', feature: Optional[str] = None):
        if self.X_test_final is None: raise RuntimeError("Jalankan .run_pipeline() terlebih dahulu.")
        
        model_to_explain = model_object or self.best_model_info.get('model_object')
        if not model_to_explain: raise ValueError("Tidak ada model untuk dijelaskan.")

        model_name = self.best_model_info.get('model_name', 'Model').upper()
        logging.info(f"Membuat SHAP Explainer untuk {model_name}...")

        try:
            explainer = shap.Explainer(model_to_explain, self.X_train_final)
            shap_values = explainer(self.X_test_final)
        except Exception as e: return logging.error(f"Gagal membuat SHAP explainer: {e}")
        
        logging.info(f"Membuat SHAP '{plot_type}' plot...")
        plt.figure(); title = f"SHAP {plot_type.title()} Plot for {model_name}"
        if plot_type == 'summary': shap.summary_plot(shap_values, self.X_test_final, show=False)
        elif plot_type == 'beeswarm': shap.plots.beeswarm(shap_values, show=False)
        elif plot_type == 'dependence':
            if not feature: raise ValueError("Mohon sediakan argumen 'feature' untuk dependence plot.")
            shap.dependence_plot(feature, shap_values.values, self.X_test_final, interaction_index=None, show=False)
        else: logging.warning(f"Plot tipe '{plot_type}' belum didukung.")
        
        plt.title(title); plt.tight_layout(); plt.show()

    def get_results_dataframe(self) -> pd.DataFrame:
        if not self.all_results:
            logging.warning("Tidak ada hasil. Jalankan .run_pipeline() dahulu."); return pd.DataFrame()
        
        records = [ {'model': res['model_name'], **res['metrics']} for res in self.all_results if 'error' not in res ]
        df = pd.DataFrame(records).set_index('model')
        primary_metric = 'f1_score' if self.task == 'classification' else 'r2_score'
        is_higher_better = primary_metric in ['f1_score', 'r2_score', 'accuracy', 'precision', 'recall']
        return df.sort_values(by=primary_metric, ascending=not is_higher_better)

    def save_model(self, filepath: str):
        if not self.best_model_info: raise ValueError("Tidak ada model terbaik untuk disimpan.")
        model_to_save = self.best_model_info.get('model_object')
        logging.info(f"Menyimpan model '{self.best_model_info['model_name']}' ke {filepath}...")
        with open(filepath, 'wb') as f: pickle.dump(model_to_save, f)
        logging.info("✅ Model berhasil disimpan.")

    @staticmethod
    def load_model(filepath: str) -> Any:
        logging.info(f"Memuat model dari {filepath}...")
        with open(filepath, 'rb') as f: model = pickle.load(f)
        logging.info("✅ Model berhasil dimuat."); return model