import pandas as pd
import numpy as np
import pickle
import time
import logging
from typing import Dict, Any, List, Union

# Model Selection, Ensembling, and Metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

# Model Libraries
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ManualPredictor:
    """
    Kelas untuk menjalankan pipeline machine learning manual secara robust.
    Mendukung model tunggal, perbandingan multi-model, dan ensembling.
    """
    def __init__(self, model_name: Union[str, List[str]], task: str, random_state: int = 42):
        self.model_name = model_name
        self.task = task.lower()
        self.random_state = random_state
        self.best_model_info = {}

        if self.task not in ['classification', 'regression']:
            raise ValueError("Tugas harus 'classification' atau 'regression'.")

    def _load_single_model(self, name: str) -> Any:
        """Memuat satu instance model berdasarkan nama."""
        name = name.lower()
        classification_models = {
            'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'decision_tree': DecisionTreeClassifier(random_state=self.random_state),
            'random_forest': RandomForestClassifier(random_state=self.random_state),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state),
            'xgboost': xgb.XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='logloss'),
            'lightgbm': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1),
            'catboost': cb.CatBoostClassifier(random_state=self.random_state, verbose=0)
        }
        regression_models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(random_state=self.random_state),
            'gradient_boosting': GradientBoostingRegressor(random_state=self.random_state),
            'xgboost': xgb.XGBRegressor(random_state=self.random_state),
            'lightgbm': lgb.LGBMRegressor(random_state=self.random_state, verbose=-1),
            'catboost': cb.CatBoostRegressor(random_state=self.random_state, verbose=0)
        }
        models = classification_models if self.task == 'classification' else regression_models
        model = models.get(name)
        if model is None:
            raise ValueError(f"Model '{name}' tidak dikenal untuk tugas {self.task}.")
        return model

    def _get_metrics(self, y_true, y_pred, task):
        if task=='classification':
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'f1_score_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'f1_score_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
                'f1_score_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
                'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0)
            }
        else:
            return {
                'r2_score': r2_score(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            }

    def _run_single_model_pipeline(self, model_name: str, X_train, y_train, X_test, y_test) -> Dict:
        """Menjalankan pipeline untuk satu model."""
        logging.info(f"--- Memproses model: {model_name.upper()} ---")
        model = self._load_single_model(model_name)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        logging.info(f"Training selesai dalam {training_time:.2f} detik.")
        
        predictions = model.predict(X_test)
        
        metrics = self._get_metrics(y_test, predictions, self.task)

        return {
            'model_name': model_name,
            'model_object': model,
            'metrics': metrics,
            'training_time_seconds': training_time
        }

    def run_pipeline(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2) -> Dict:
        """
        Menjalankan pipeline lengkap. Jika model_name adalah list, akan menjalankan
        perbandingan dan ensembling.
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        stratify = y if self.task == 'classification' else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify
        )

        if isinstance(self.model_name, str):
            logging.info(f"Menjalankan pipeline untuk model tunggal: {self.model_name}")
            self.best_model_info = self._run_single_model_pipeline(self.model_name, X_train, y_train, X_test, y_test)
            return self.best_model_info

        elif isinstance(self.model_name, list):
            logging.info("Memulai mode perbandingan dan ensembling untuk beberapa model.")
            all_results = []
            
            for name in self.model_name:
                try:
                    result = self._run_single_model_pipeline(name, X_train, y_train, X_test, y_test)
                    all_results.append(result)
                except Exception as e:
                    logging.error(f"Gagal memproses model {name}: {e}")

            if not all_results:
                raise RuntimeError("Tidak ada model individual yang berhasil dilatih.")

            logging.info("--- Memproses model: ENSEMBLE ---")
            estimators = [(res['model_name'], res['model_object']) for res in all_results]
            
            ensemble_model = VotingClassifier(estimators=estimators, voting='soft') if self.task == 'classification' else VotingRegressor(estimators=estimators)

            try:
                start_time = time.time()
                ensemble_model.fit(X_train, y_train)
                training_time = time.time() - start_time
                logging.info(f"Training Ensemble selesai dalam {training_time:.2f} detik.")
                
                predictions = ensemble_model.predict(X_test)
                metrics = self._get_metrics(y_test, predictions, self.task)

                all_results.append({
                    'model_name': 'ensemble', 'model_object': ensemble_model,
                    'metrics': metrics, 'training_time_seconds': training_time
                })
            except Exception as e:
                logging.error(f"Gagal melatih model ensemble: {e}")

            primary_metric = 'f1_weighted_score' if self.task == 'classification' else 'r2_score'
            self.best_model_info = max(all_results, key=lambda x: x['metrics'][primary_metric])
            
            logging.info(f"\n--- Perbandingan Selesai ---")
            logging.info(f"🏆 Model Terbaik: {self.best_model_info['model_name'].upper()} dengan {primary_metric} = {self.best_model_info['metrics'][primary_metric]:.4f}")

            return {'best_model_details': self.best_model_info, 'all_model_results': all_results}
        else:
            raise TypeError("model_name harus berupa string atau list of strings.")

    def save_model(self, filepath: str):
        """Menyimpan model TERBAIK yang telah dilatih ke file."""
        if not self.best_model_info:
            raise ValueError("Tidak ada model terbaik untuk disimpan. Jalankan .run_pipeline() terlebih dahulu.")
        
        model_to_save = self.best_model_info.get('model_object')
        logging.info(f"Menyimpan model terbaik '{self.best_model_info['model_name']}' ke {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(model_to_save, f)
        logging.info("✅ Model berhasil disimpan.")

    @staticmethod
    def load_model(filepath: str) -> Any:
        """Memuat model dari file."""
        logging.info(f"Memuat model dari {filepath}...")
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logging.info("✅ Model berhasil dimuat.")
        return model
