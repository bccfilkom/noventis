import pandas as pd
import numpy as np
import os
import pickle
from typing import Union, Optional, Dict, Any, List
import warnings

# Import library FLAML untuk AutoML, scikit-learn untuk evaluasi & split data, serta visualisasi
from flaml import AutoML as FLAMLAutoML
from flaml.automl.data import get_output_from_log
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, r2_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


from .manual import ManualPredictor

warnings.filterwarnings('ignore')  # Menonaktifkan warning agar output lebih bersih

class NoventisAutoML:
    """
    Kelas utama untuk menjalankan pipeline AutoML menggunakan FLAML.
    Mendukung deteksi otomatis tipe tugas (klasifikasi/regresi), pelatihan model, evaluasi, 
    penyimpanan model, visualisasi, dan perbandingan dengan model manual.
    """

    def __init__(self, data: Union[str, pd.DataFrame], target: str, task: Optional[str] = None, test_size: float = 0.2, random_state: int = 42):
        self.target_column = target
        self.task_type = task.lower() if task else None
        self.test_size = test_size
        self.random_state = random_state
        self.flaml_model = None
        self.results = {}
        self._load_data(data)
        self._setup_data()

    def _load_data(self, data: Union[str, pd.DataFrame]):
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise TypeError("Input 'data' harus berupa path CSV atau pandas DataFrame.")

    def _detect_task_type(self) -> str:
        y = self.df[self.target_column]
        if pd.api.types.is_numeric_dtype(y) and (len(y.unique()) > 25 and (len(y.unique()) / len(y)) >= 0.05):
            return "regression"
        return "classification"

    def _setup_data(self):
        if self.target_column not in self.df.columns:
            raise ValueError(f"Kolom target '{self.target_column}' tidak ditemukan.")
        X, y = self.df.drop(columns=[self.target_column]), self.df[self.target_column]
        if self.task_type is None:
            self.task_type = self._detect_task_type()
            print(f"✅ Tipe tugas terdeteksi: {self.task_type}")
        stratify = y if self.task_type == "classification" else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify
        )
        print(f"✅ Data berhasil dibagi: Train={len(self.X_train)}, Test={len(self.X_test)}")

    def fit(self, time_budget: int = 60, metric: Optional[str] = None, explain: bool = True, output_dir: str = "noventis_output", **kwargs) -> Dict:
        print("🚀 Memulai proses AutoML dengan FLAML...")
        flaml_metric = self._convert_metric_to_flaml(metric)
        self.flaml_model = FLAMLAutoML(
            task=self.task_type, metric=flaml_metric, time_budget=time_budget,
            seed=self.random_state, log_file_name=f'{output_dir}/flaml.log', **kwargs
        )
        # Membuat direktori output jika belum ada
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"⏳ Melatih model (Metrik: {flaml_metric}, Waktu: {time_budget}s)...")
        self.flaml_model.fit(X_train=self.X_train, y_train=self.y_train)
        y_pred = self.flaml_model.predict(self.X_test)
        
        metrics, y_pred_proba = (
            self._eval_classification(self.y_test, y_pred),
            self.flaml_model.predict_proba(self.X_test) if hasattr(self.flaml_model, 'predict_proba') else None
        ) if self.task_type == "classification" else (
            self._eval_regression(self.y_test, y_pred), None
        )
        
        self.results = {
            'model': self.flaml_model,
            'predictions': y_pred,
            'prediction_proba': y_pred_proba,
            'actual': self.y_test,
            'metrics': metrics,
            'task_type': self.task_type,
            'feature_importance': self._get_feature_importance(),
            'best_estimator': self.flaml_model.best_estimator,
            'best_config': self.flaml_model.best_config,
            'training_history': self._get_training_history(f'{output_dir}/flaml.log')
        }
        
        model_path = os.path.join(output_dir, 'best_automl_model.pkl')
        self._save_model(self.flaml_model, model_path)
        self.results['model_path'] = model_path
        print(f"💾 Model berhasil disimpan di: {model_path}")

        if explain:
            self.results['visualization_paths'] = self._generate_visualizations(self.results, output_dir)
            print(f"📊 Visualisasi berhasil dibuat dan disimpan di direktori '{output_dir}'!")
            
        print(f"\n🎉 Proses AutoML Selesai! Estimator terbaik: {self.flaml_model.best_estimator}")
        return self.results

    def compare_models(self, models_to_compare: Optional[List[str]] = None, output_dir: str = "noventis_output", **kwargs) -> Dict:
        if models_to_compare is None:
            models_to_compare = ['logistic_regression', 'random_forest', 'xgboost'] if self.task_type == 'classification' else ['linear_regression', 'random_forest', 'xgboost']
        
        os.makedirs(output_dir, exist_ok=True)
        all_results = {}
        for model_name in models_to_compare:
            print(f"\n{'='*20} Melatih {model_name.replace('_', ' ').title()} {'='*20}")
            try:
                predictor = ManualPredictor(model_name=model_name, task=self.task_type, random_state=self.random_state)
                result = predictor.run_pipeline(self.df, target_column=self.target_column, test_size=self.test_size)
                all_results[model_name] = result
            except Exception as e:
                print(f"❌ Error saat melatih {model_name}: {e}")
                all_results[model_name] = {'error': str(e)}
        
        ranked_results = self._rank_models(all_results)
        self._visualize_model_comparison(ranked_results, output_dir)
        print(f"📊 Visualisasi perbandingan model disimpan di direktori '{output_dir}'.")
        return ranked_results

    def _eval_classification(self, y_true, y_pred):
        return {'f1_score_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)}

    def _eval_regression(self, y_true, y_pred):
        return {'r2_score': r2_score(y_true, y_pred)}

    def _convert_metric_to_flaml(self, metric: Optional[str]) -> str:
        if metric is None:
            return "roc_auc" if self.task_type == "classification" else "r2"
        return metric

    def _get_feature_importance(self) -> Optional[pd.DataFrame]:
        try:
            return pd.DataFrame({
                'feature': self.X_train.columns, 
                'importance': self.flaml_model.feature_importances_
            }).sort_values('importance', ascending=False)
        except:
            return None

    def _get_training_history(self, log_file) -> Optional[pd.DataFrame]:
        try:
            if os.path.exists(log_file):
                time_h, loss_h, _, _, _ = get_output_from_log(filename=log_file, time_budget=float('inf'))
                return pd.DataFrame({'time_seconds': time_h, 'best_validation_loss': loss_h})
        except:
            return None

    def _save_model(self, model, path):
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    def _generate_visualizations(self, results, output_dir: str) -> List[str]:
        """Membuat dan menyimpan visualisasi hasil pelatihan AutoML."""
        paths = []
        plt.style.use('seaborn-v0_8-whitegrid')

        # 1. Feature Importance
        if results['feature_importance'] is not None and not results['feature_importance'].empty:
            plt.figure(figsize=(10, max(6, len(results['feature_importance']) * 0.4)))
            sns.barplot(x='importance', y='feature', data=results['feature_importance'], palette='viridis')
            plt.title('Feature Importance', fontsize=16)
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.tight_layout()
            path = os.path.join(output_dir, 'feature_importance.png')
            plt.savefig(path)
            paths.append(path)
            plt.close()

        # 2. Training History (Learning Curve)
        if results['training_history'] is not None and not results['training_history'].empty:
            plt.figure(figsize=(10, 6))
            history = results['training_history']
            plt.plot(history['time_seconds'], history['best_validation_loss'], marker='o', linestyle='-', color='b')
            plt.title('AutoML Training History', fontsize=16)
            plt.xlabel('Time (seconds)', fontsize=12)
            plt.ylabel('Best Validation Loss', fontsize=12)
            plt.tight_layout()
            path = os.path.join(output_dir, 'training_history.png')
            plt.savefig(path)
            paths.append(path)
            plt.close()

        # 3. Visualisasi spesifik per tugas
        if results['task_type'] == 'classification':
            # Confusion Matrix
            plt.figure(figsize=(8, 8))
            cm = confusion_matrix(results['actual'], results['predictions'], labels=self.flaml_model.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.flaml_model.classes_)
            disp.plot(cmap='Blues', values_format='d')
            plt.title('Confusion Matrix', fontsize=16)
            plt.xticks(rotation=45)
            path = os.path.join(output_dir, 'confusion_matrix.png')
            plt.savefig(path, bbox_inches='tight')
            paths.append(path)
            plt.close()
        
        elif results['task_type'] == 'regression':
            # Prediction vs Actual Plot
            plt.figure(figsize=(8, 8))
            plt.scatter(results['actual'], results['predictions'], alpha=0.6, edgecolors='k')
            # Garis diagonal y=x sebagai referensi
            perfect_line = np.linspace(min(results['actual']), max(results['actual']), 100)
            plt.plot(perfect_line, perfect_line, color='red', linestyle='--', lw=2, label='Perfect Prediction')
            plt.title('Predictions vs. Actual Values', fontsize=16)
            plt.xlabel('Actual Values', fontsize=12)
            plt.ylabel('Predicted Values', fontsize=12)
            plt.legend()
            plt.grid(True)
            path = os.path.join(output_dir, 'predictions_vs_actual.png')
            plt.savefig(path)
            paths.append(path)
            plt.close()

        return paths

    def _rank_models(self, results) -> Dict:
        rankings = []
        metric = 'f1_score' if self.task_type == "classification" else 'r2_score'
        
        for name, res in results.items():
            if 'error' in res or 'metrics' not in res:
                continue
            score = res['metrics'].get(metric, -1)
            rankings.append({'model': name.replace('_', ' ').title(), 'score': score})
        
        rankings.sort(key=lambda x: x['score'], reverse=True)
        return {'rankings': rankings, 'best_model': rankings[0]['model'] if rankings else None}

    def _visualize_model_comparison(self, ranked_results, output_dir: str):
        """Membuat dan menyimpan visualisasi perbandingan performa model."""
        if not ranked_results['rankings']:
            print("Tidak ada model untuk dibandingkan.")
            return

        df_ranks = pd.DataFrame(ranked_results['rankings'])
        metric = 'F1 Score (Macro)' if self.task_type == "classification" else 'R2 Score'
        
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(x='score', y='model', data=df_ranks, palette='magma')
        plt.title('Model Performance Comparison', fontsize=16)
        plt.xlabel(metric, fontsize=12)
        plt.ylabel('Model', fontsize=12)
        
        # Menambahkan label skor pada setiap bar
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f')
            
        plt.tight_layout()
        path = os.path.join(output_dir, 'model_comparison.png')
        plt.savefig(path)
        plt.close()

