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
from sklearn.metrics import f1_score, r2_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

from .manual import ManualPredictor

warnings.filterwarnings('ignore')

class NoventisAutoML:
    """
    Kelas utama untuk menjalankan pipeline AutoML menggunakan FLAML.
    Mendukung deteksi otomatis tipe tugas (klasifikasi/regresi), pelatihan model, evaluasi, 
    penyimpanan model, visualisasi, dan perbandingan dengan model manual.
    """

    def __init__(
        self, 
        data: Union[str, pd.DataFrame], 
        target: str, 
        task: Optional[str] = None, 
        models: List[str]=None,
        explain: bool=True,
        compare: bool=True,
        metrics: str=None,
        time_budget: int=60,
        output_dir: str='Noventis_Results',
        test_size: float = 0.2, 
        random_state: int = 42,
    ):
        self.target_column = target
        self.task_type = task.lower() if task else None
        self.test_size = test_size
        self.random_state = random_state
        self.explain=explain
        self.compare=compare
        self.metrics=metrics
        self.time_budget=time_budget
        self.output_dir=output_dir
        self.model_list=models
        self.use_automl = True if (compare or self.model_list is not None) else False 

        self.flaml_model = None
        self.manual_model = None
        self.results = {}
        self._load_data(data)
        self._setup_data()

    def _load_data(self, data: Union[str, pd.DataFrame]):
        """Load data dari CSV atau DataFrame"""
        if isinstance(data, str):
            self.df = pd.read_csv(data)
            print(f"✅ Data berhasil dimuat dari file: {data}")
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
            print("✅ Data berhasil dimuat dari DataFrame")
        else:
            raise TypeError("Input 'data' harus berupa path CSV atau pandas DataFrame.")
        
        print(f"📊 Shape data: {self.df.shape}")
        print(f"📋 Kolom: {list(self.df.columns)}")

    def _detect_task_type(self) -> str: #bnr
        """Deteksi otomatis tipe tugas berdasarkan target variable"""
        y = self.df[self.target_column]
        unique_values = len(y.unique())
        unique_ratio = unique_values / len(y)
        
        if pd.api.types.is_numeric_dtype(y):
            if unique_values > 25 and unique_ratio >= 0.05:
                return "regression"
            else:
                return "classification"
        else:
            return "classification"

    def _setup_data(self):  # bnr
        """Setup dan split data untuk training dan testing"""
        if self.target_column not in self.df.columns:
            raise ValueError(f"Kolom target '{self.target_column}' tidak ditemukan.")
        
        # Pisahkan features dan target
        X, y = self.df.drop(columns=[self.target_column]), self.df[self.target_column]
        
        # Auto-detect task type jika tidak dispesifikasi
        if self.task_type is None:
            self.task_type = self._detect_task_type()
            print(f"✅ Tipe tugas terdeteksi: {self.task_type}")
        
        # Split data dengan stratification untuk classification
        stratify = y if self.task_type == "classification" else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify
        )
        
        print(f"✅ Data berhasil dibagi: Train={len(self.X_train)}, Test={len(self.X_test)}")
        print(f"📈 Target distribution: {dict(y.value_counts()) if self.task_type == 'classification' else f'Range: {y.min():.2f} - {y.max():.2f}'}")

    def fit(self, time_budget: int = 60, metric: Optional[str] = None) -> Dict:
        """
        Melatih model AutoML dengan FLAML
        
        Parameters:
        - time_budget: Waktu maksimum untuk training (detik)
        - metric: Metrik evaluasi yang digunakan
        - explain: Jika True, generate visualisasi dan penjelasan
        - compare: Jika True, bandingkan dengan model manual lainnya
        - output_dir: Direktori untuk menyimpan hasil
        """
        print("🚀 Memulai proses AutoML dengan FLAML...")
        
        # Convert metric untuk FLAML
        flaml_metric = self._convert_metric_to_flaml(self.metrics)  #bnr

        # Buat direktori output
        os.makedirs(self.output_dir, exist_ok=True)
        if self.use_automl:      
            # Initialize FLAML AutoML
            self.flaml_model = FLAMLAutoML( #bnr
                task=self.task_type, 
                metric=flaml_metric, 
                seed=self.random_state, 
                verbose=2,
            )
            
            print(f"⏳ Melatih model (Metrik: {flaml_metric}, Waktu: {time_budget}s)...")
            
            # Training model
            self.flaml_model.fit(
                X_train=self.X_train, 
                y_train=self.y_train,
                log_file_name=f'{self.output_dir}/flaml.log', 
                time_budget=self.time_budget,
            )
            
            # Prediksi
            y_pred = self.flaml_model.predict(self.X_test)
            
            # Evaluasi berdasarkan task type
            if self.task_type == "classification":
                metrics = self._eval_classification(self.y_test, y_pred)
                y_pred_proba = self.flaml_model.predict_proba(self.X_test) if hasattr(self.flaml_model, 'predict_proba') else None
            else:
                metrics = self._eval_regression(self.y_test, y_pred)
                y_pred_proba = None
            
            # Compile results
            self.results['AutoML'] = {    # fix : ga perlu config, training history?
                'model': self.flaml_model,
                'predictions': y_pred,
                'prediction_proba': y_pred_proba,
                'actual': self.y_test,
                'metrics': metrics,
                'task_type': self.task_type,
                'feature_importance': self._get_feature_importance(),
                'best_estimator': self.flaml_model.best_estimator,
                'best_config': self.flaml_model.best_config,
                'training_history': self._get_training_history(f'{self.output_dir}/flaml.log')
            }
            
            # Save model
            model_path = os.path.join(self.output_dir, 'best_automl_model.pkl')
            self._save_model(self.flaml_model, model_path)
            self.results['model_path'] = model_path
            print(f"💾 Model berhasil disimpan di: {model_path}")
        
        else:
            predictor = ManualPredictor(
                model_name=self.model_list, 
                task=self.task_type, 
                random_state=self.random_state
            )
            result = predictor.run_pipeline(
                self.df, 
                target_column=self.target_column, 
                test_size=self.test_size
            )
            self.manual_model = predictor

            all_model_results = result['all_model_results']
            for model_result in all_model_results:
                model_name = model_result['model_name']
                metrics = model_result['metrics']
                y_pred = model_result['predictions']
                y_pred_proba = model_result['prediction_proba']
                
                self.results[model_name] = {    # masih harus di tambahin untuk yang None
                    'model': model_name,
                    'predictions': y_pred,
                    'prediction_proba': y_pred_proba,
                    'actual': self.y_test,
                    'metrics': metrics,
                    'task_type': self.task_type,
                    'feature_importance': None,
                    'best_estimator': None,
                    'best_config': None,
                    'training_history': None
                }

        # Compare dengan model lain jika compare=True
        if self.compare:
            print("\n🔍 Memulai perbandingan dengan model lain...")
            comparison_results = self.compare_models(output_dir=self.output_dir, models_to_compare=self.model_list)
            self.results['model_comparison'] = comparison_results
            print(f"\n🎉 Proses AutoML Selesai!")
        else:
            print(f"\n🎉 Proses AutoML Selesai!")
            print(f"🏆 Estimator terbaik: {self.flaml_model.best_estimator}")
            print(f"📊 Metrics: {metrics}")

        # Generate visualizations jika explain=True
        if self.explain:
            print("📊 Membuat visualisasi...")
            self.results['visualization_paths'] = self._generate_visualizations(self.results, self.output_dir)
            self._generate_model_summary(self.results, self.output_dir)
            print(f"📊 Visualisasi berhasil dibuat dan disimpan di direktori '{self.output_dir}'!")
        
        return self.results


    def compare_models(self, models_to_compare: Optional[List[str]] = None, 
                      output_dir: str = "Noventis_results") -> Dict:
        """
        Bandingkan performa AutoML dengan model manual lainnya
        """
        # if models_to_compare is None:
        #     if self.task_type == 'classification':
        #         models_to_compare = ['logistic_regression', 'random_forest', 'xgboost', 'decision_tree', 'lightgbm', 'catboost', 'gradient_boosting']
        #     else:
        #         models_to_compare = ['linear_regression', 'random_forest', 'xgboost', 'gradient_boosting', 'lightgbm', 'catboost']
        
        os.makedirs(output_dir, exist_ok=True)
        all_results = {}
        
        if models_to_compare is None:       #testing
            # buat model sendiri untuk compare dan dibandingkan dengan flaml
            if self.task_type == 'classification':
                models_to_compare = ['logistic_regression', 'random_forest', 'xgboost', 'decision_tree', 'lightgbm', 'catboost', 'gradient_boosting']
            else:
                models_to_compare = ['linear_regression', 'random_forest', 'xgboost', 'gradient_boosting', 'lightgbm', 'catboost']

            # Tambahkan hasil AutoML sebagai baseline
            automl_metrics = self.results['AutoML']['metrics'] if hasattr(self, 'results') else {}
            all_results['AutoML'] = {
                'metrics': automl_metrics,
                'model_name': 'AutoML',
                'best_estimator': getattr(self.flaml_model, 'best_estimator', 'Unknown') if self.flaml_model else 'Unknown'
            }
        
        if self.manual_model is None:
            predictor = ManualPredictor(
                model_name=self.model_list, 
                task=self.task_type, 
                random_state=self.random_state
            )
            result = predictor.run_pipeline(
                self.df, 
                target_column=self.target_column, 
                test_size=self.test_size
            )
            self.manual_model = predictor

            all_model_results = result['all_model_results']
            for model_result in all_model_results:
                model_name = model_result['model_name']
                metrics = model_result['metrics']
                y_pred = model_result['predictions']
                y_pred_proba = model_result['prediction_proba']
                
                self.results[model_name] = {    # masih harus di tambahin untuk yang None
                    'model': model_name,
                    'predictions': y_pred,
                    'prediction_proba': y_pred_proba,
                    'actual': self.y_test,
                    'metrics': metrics,
                    'task_type': self.task_type,
                    'feature_importance': None,
                    'best_estimator': None,
                    'best_config': None,
                    'training_history': None
                }

        # Test model manual lainnya
        for model in self.results:
            if model == 'AutoML':
                continue
            all_results[model] = self.results[model]
        
        # Ranking dan visualisasi
        ranked_results = self._rank_models(all_results)
        self._visualize_model_comparison(ranked_results, all_results, output_dir)
        self._generate_comparison_report(ranked_results, all_results, output_dir)
        predictor.save_model(f'{self.output_dir}/best_model_without_automl.pkl')
        
        print(f"📊 Hasil perbandingan model disimpan di direktori '{output_dir}'.")
        return ranked_results

    def _eval_classification(self, y_true, y_pred) -> Dict:
        """Evaluasi lengkap untuk classification"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    def _eval_regression(self, y_true, y_pred) -> Dict:
        """Evaluasi lengkap untuk regression"""
        mse = mean_squared_error(y_true, y_pred)
        return {
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2_score': r2_score(y_true, y_pred)
            }

    def _convert_metric_to_flaml(self, metric: Optional[str]) -> str:
        """Convert metric name untuk FLAML"""
        if metric is None:
            return "macro_f1" if self.task_type == "classification" else "r2"
            # return 'auto'
        elif metric == 'f1_score':
            return 'f1'
        elif metric == 'r2_score':
            return 'r2'
        return metric

    def _get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Ekstrak feature importance jika tersedia"""
        if self.use_automl:
            try:
                if hasattr(self.flaml_model, 'feature_importances_'):
                    return pd.DataFrame({
                        'feature': self.X_train.columns, 
                        'importance': self.flaml_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                else:
                    return None
            except Exception as e:
                print(f"⚠️ Tidak dapat mengekstrak feature importance: {e}")
                return None
            

    def _get_training_history(self, log_file) -> Optional[pd.DataFrame]:
        """Ekstrak training history dari log file FLAML"""
        if self.use_automl:
            try:
                if os.path.exists(log_file):
                    time_h, loss_h, _, _, _ = get_output_from_log(filename=log_file, time_budget=float('inf'))
                    return pd.DataFrame({
                        'time_seconds': time_h, 
                        'best_validation_loss': loss_h
                    })
                return None
            except Exception as e:
                print(f"⚠️ Tidak dapat membaca training history: {e}")
                return None

    def _save_model(self, model, path):
        """Save model ke file pickle"""
        try:
            with open(path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            print(f"⚠️ Error saat menyimpan model: {e}")

    def _generate_visualizations(self, results, output_dir: str) -> List[str]:
        """Generate comprehensive visualizations"""
        paths = []
        plt.style.use('default')  # Use default style for better compatibility
        
        try:
            if self.use_automl:
                # 1. Feature Importance
                if results['feature_importance'] is not None and not results['feature_importance'].empty:
                    plt.figure(figsize=(12, max(6, len(results['feature_importance']) * 0.4)))
                    top_features = results['feature_importance'].head(20)  # Top 20 features
                    
                    sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
                    plt.title('Top 20 Feature Importance', fontsize=16, fontweight='bold')
                    plt.xlabel('Importance Score', fontsize=12)
                    plt.ylabel('Features', fontsize=12)
                    plt.tight_layout()
                    
                    path = os.path.join(output_dir, 'feature_importance.png')
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    paths.append(path)
                    plt.close()

                # 2. Training History
                if results['training_history'] is not None and not results['training_history'].empty:
                    plt.figure(figsize=(12, 6))
                    history = results['training_history']
                    
                    plt.plot(history['time_seconds'], history['best_validation_loss'], 
                            marker='o', linestyle='-', color='b', linewidth=2, markersize=4)
                    plt.title('AutoML Training Progress', fontsize=16, fontweight='bold')
                    plt.xlabel('Time (seconds)', fontsize=12)
                    plt.ylabel('Best Validation Loss', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    path = os.path.join(output_dir, 'training_history.png')
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    paths.append(path)
                    plt.close()

            # 3. Task-specific visualizations
            if results['task_type'] == 'classification':
                paths.extend(self._generate_classification_plots(results, output_dir))
            else:
                paths.extend(self._generate_regression_plots(results, output_dir))
                
        except Exception as e:
            print(f"⚠️ Error saat membuat visualisasi: {e}")
            
        return paths

    def _generate_classification_plots(self, results, output_dir: str) -> List[str]:
        """Generate classification-specific plots"""
        paths = []
        
        try:
            # Confusion Matrix
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(results['actual'], results['predictions'])
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                       xticklabels=True, yticklabels=True)
            plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            
            path = os.path.join(output_dir, 'confusion_matrix.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            paths.append(path)
            plt.close()
            
            # Classification Metrics Bar Plot
            metrics = results['metrics']
            plt.figure(figsize=(10, 6))
            
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = plt.bar(metric_names, metric_values, color='skyblue', alpha=0.8)
            plt.title('Classification Metrics', fontsize=16, fontweight='bold')
            plt.ylabel('Score', fontsize=12)
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            path = os.path.join(output_dir, 'classification_metrics.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            paths.append(path)
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Error saat membuat plot klasifikasi: {e}")
            
        return paths

    def _generate_regression_plots(self, results, output_dir: str) -> List[str]:
        """Generate regression-specific plots"""
        paths = []
        
        try:
            # Prediction vs Actual Plot
            plt.figure(figsize=(10, 8))
            
            plt.scatter(results['actual'], results['predictions'], 
                       alpha=0.6, edgecolors='k', s=50)
            
            # Perfect prediction line
            min_val = min(min(results['actual']), min(results['predictions']))
            max_val = max(max(results['actual']), max(results['predictions']))
            perfect_line = np.linspace(min_val, max_val, 100)
            
            plt.plot(perfect_line, perfect_line, color='red', linestyle='--', 
                    linewidth=2, label='Perfect Prediction')
            
            plt.title('Predictions vs. Actual Values', fontsize=16, fontweight='bold')
            plt.xlabel('Actual Values', fontsize=12)
            plt.ylabel('Predicted Values', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add R² score as text
            r2 = results['metrics'].get('r2_score', 0)
            plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=12)
            
            plt.tight_layout()
            path = os.path.join(output_dir, 'predictions_vs_actual.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            paths.append(path)
            plt.close()
            
            # Residuals Plot
            residuals = np.array(results['actual']) - np.array(results['predictions'])
            
            plt.figure(figsize=(10, 6))
            plt.scatter(results['predictions'], residuals, alpha=0.6, edgecolors='k')
            plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
            plt.title('Residuals Plot', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Values', fontsize=12)
            plt.ylabel('Residuals', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            path = os.path.join(output_dir, 'residuals_plot.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            paths.append(path)
            plt.close()
            
            # Regression Metrics Bar Plot
            metrics = results['metrics']
            plt.figure(figsize=(12, 6))
            
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = plt.bar(metric_names, metric_values, color='lightcoral', alpha=0.8)
            plt.title('Regression Metrics', fontsize=16, fontweight='bold')
            plt.ylabel('Score', fontsize=12)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                plt.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + (max(metric_values) * 0.01),
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            path = os.path.join(output_dir, 'regression_metrics.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            paths.append(path)
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Error saat membuat plot regresi: {e}")
            
        return paths

    def _generate_model_summary(self, results, output_dir: str):
        """Generate summary report"""
        try:
            summary_path = os.path.join(output_dir, 'model_summary.txt')
            
            with open(summary_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("         NOVENTIS AutoML - MODEL SUMMARY\n")
                f.write("="*60 + "\n\n")
                
                f.write(f"Task Type: {self.task_type}\n")
                # f.write(f"Best Estimator: {results['best_estimator']}\n")
                # f.write(f"Best Configuration: {results['best_config']}\n\n")
                
                f.write("PERFORMANCE METRICS:\n")
                f.write("-" * 30 + "\n")
                for model_name in results:
                    for metric, value in results[model_name]['metrics'].items():
                        f.write(f"{metric.replace('_', ' ').title()}: {value:.4f}\n")
                    
                    if results[model_name]['feature_importance'] is not None:
                        f.write(f"\nTOP 10 IMPORTANT FEATURES:\n")
                        f.write("-" * 30 + "\n")
                        for idx, row in results[model_name]['feature_importance'].head(10).iterrows():
                            f.write(f"{row['feature']}: {row['importance']:.4f}\n")
                
                    f.write(f"\nModel saved at: {results['model_path']}\n")
                
            print(f"📄 Summary report disimpan di: {summary_path}")
            
        except Exception as e:
            print(f"⚠️ Error saat membuat summary: {e}")

    def _rank_models(self, results) -> Dict:
        """Rank models berdasarkan performa"""
        rankings = []
        
        # Tentukan primary metric untuk ranking
        if self.task_type == "classification":
            primary_metric = 'f1_score' if self.metrics is None else self.metrics   
        else:
            primary_metric = 'r2_score' if self.metrics is None else self.metrics
        
        for name, res in results.items():
            if 'error' in res or 'metrics' not in res:
                continue
                
            score = res['metrics'].get(primary_metric, -1)      
            model_display_name = name.replace('_', ' ').title()
            
            rankings.append({
                'model': model_display_name,
                'score': score,
                'metrics': res['metrics']
            })
        
        # Sort by primary metric (descending)
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'rankings': rankings,
            'best_model': rankings[0]['model'] if rankings else None,
            'primary_metric': primary_metric
        }

    def _visualize_model_comparison(self, ranked_results, all_results, output_dir: str):
        """Create comprehensive model comparison visualizations"""
        if not ranked_results['rankings']:
            print("Tidak ada model untuk dibandingkan.")
            return

        try:
            df_ranks = pd.DataFrame(ranked_results['rankings'])
            primary_metric = ranked_results['primary_metric']
            metric_display = primary_metric.replace('_', ' ').title()
            
            # 1. Performance Comparison Bar Chart
            plt.figure(figsize=(14, 8))
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(df_ranks)))
            bars = plt.barh(df_ranks['model'], df_ranks['score'], color=colors)
            
            plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
            plt.xlabel(metric_display, fontsize=12)
            plt.ylabel('Model', fontsize=12)
            
            # Add score labels
            for i, (bar, score) in enumerate(zip(bars, df_ranks['score'])):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{score:.4f}', va='center', ha='left', fontweight='bold')
            
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            path = os.path.join(output_dir, 'model_comparison.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Detailed Metrics Heatmap (if multiple metrics available)
            self._create_metrics_heatmap(df_ranks, output_dir)
            
        except Exception as e:
            print(f"⚠️ Error saat membuat visualisasi perbandingan: {e}")

    def _create_metrics_heatmap(self, df_ranks, output_dir: str):
        """Create heatmap of all metrics across models"""
        try:
            # Extract all metrics from rankings
            all_metrics = {}
            for ranking in df_ranks.to_dict('records'):
                model_name = ranking['model']
                metrics = ranking['metrics']
                all_metrics[model_name] = metrics
            
            if not all_metrics:
                return
                
            # Create DataFrame for heatmap
            metrics_df = pd.DataFrame(all_metrics).T
            
            if len(metrics_df.columns) > 1:  # Only create if multiple metrics
                plt.figure(figsize=(12, 8))
                
                sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn',
                          cbar_kws={'label': 'Score'})
                
                plt.title('Model Performance Heatmap - All Metrics', 
                         fontsize=16, fontweight='bold')
                plt.xlabel('Metrics', fontsize=12)
                plt.ylabel('Models', fontsize=12)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                path = os.path.join(output_dir, 'metrics_heatmap.png')
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"⚠️ Error saat membuat heatmap metrics: {e}")

    def _generate_comparison_report(self, ranked_results, all_results, output_dir: str):
        """Generate detailed comparison report"""
        try:
            report_path = os.path.join(output_dir, 'model_comparison_report.txt')
            
            with open(report_path, 'w') as f:
                f.write("="*70 + "\n")
                f.write("         NOVENTIS AutoML - MODEL COMPARISON REPORT\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"Task Type: {self.task_type.title()}\n")
                f.write(f"Primary Metric: {ranked_results['primary_metric']}\n")
                f.write(f"Best Model: {ranked_results['best_model']}\n\n")
                
                f.write("MODEL RANKINGS:\n")
                f.write("-" * 50 + "\n")
                
                for i, ranking in enumerate(ranked_results['rankings'], 1):
                    f.write(f"{i}. {ranking['model']}\n")
                    f.write(f"   Primary Score: {ranking['score']:.4f}\n")
                    f.write("   All Metrics:\n")
                    
                    for metric, value in ranking['metrics'].items():
                        f.write(f"     {metric.replace('_', ' ').title()}: {value:.4f}\n")
                    f.write("\n")
                
                # Add failed models info
                failed_models = [name for name, res in all_results.items() if 'error' in res]
                if failed_models:
                    f.write("FAILED MODELS:\n")
                    f.write("-" * 30 + "\n")
                    for model_name in failed_models:
                        error_msg = all_results[model_name]['error']
                        f.write(f"• {model_name.replace('_', ' ').title()}: {error_msg}\n")
                
            print(f"📄 Comparison report disimpan di: {report_path}")
            
        except Exception as e:
            print(f"⚠️ Error saat membuat comparison report: {e}")

    def load_model(self, model_path: str):
        """Load saved model dari file pickle"""
        try:
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            print(f"✅ Model berhasil dimuat dari: {model_path}")
            return loaded_model
        except Exception as e:
            print(f"❌ Error saat memuat model: {e}")
            return None

    def predict(self, X_new: Union[pd.DataFrame, np.ndarray], model_path: Optional[str] = None):
        """
        Prediksi dengan model yang sudah dilatih
        
        Parameters:
        - X_new: Data baru untuk diprediksi
        - model_path: Path ke saved model (optional, akan menggunakan current model jika tidak dispesifikasi)
        """
        if model_path:
            model = self.load_model(model_path)
        elif self.flaml_model:
            model = self.flaml_model
        else:
            raise ValueError("Tidak ada model yang tersedia. Latih model terlebih dahulu atau spesifikasi model_path.")
        
        try:
            predictions = model.predict(X_new)
            print(f"✅ Prediksi berhasil untuk {len(X_new)} samples")
            
            # Jika classification dan model mendukung predict_proba
            if self.task_type == "classification" and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_new)
                return {
                    'predictions': predictions,
                    'probabilities': probabilities
                }
            else:
                return {'predictions': predictions}
                
        except Exception as e:
            print(f"❌ Error saat prediksi: {e}")
            return None

    def get_model_info(self) -> Dict:
        """Get informasi detail tentang model yang sudah dilatih"""
        if not self.flaml_model:
            return {"error": "Model belum dilatih. Jalankan fit() terlebih dahulu."}
        
        info = {
            'best_estimator': self.flaml_model.best_estimator,
            'best_config': self.flaml_model.best_config,
            'task_type': self.task_type,
            'training_duration': getattr(self.flaml_model, 'training_duration', 'Unknown'),
            'classes_': getattr(self.flaml_model, 'classes_', None),
            'feature_names': list(self.X_train.columns) if hasattr(self, 'X_train') else None
        }
        
        return info

    def export_results_to_csv(self, output_dir: str = "noventis_output"):
        """Export hasil prediksi dan metrics ke CSV"""
        if not hasattr(self, 'results') or not self.results:
            print("❌ Tidak ada hasil untuk diekspor. Jalankan fit() terlebih dahulu.")
            return
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Export predictions
            predictions_df = pd.DataFrame({
                'actual': self.results['actual'],
                'predicted': self.results['predictions']
            })
            
            if self.results['prediction_proba'] is not None:
                proba_cols = [f'prob_class_{i}' for i in range(self.results['prediction_proba'].shape[1])]
                proba_df = pd.DataFrame(self.results['prediction_proba'], columns=proba_cols)
                predictions_df = pd.concat([predictions_df, proba_df], axis=1)
            
            pred_path = os.path.join(output_dir, 'predictions.csv')
            predictions_df.to_csv(pred_path, index=False)
            
            # Export metrics
            metrics_df = pd.DataFrame([self.results['metrics']])
            metrics_path = os.path.join(output_dir, 'metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            
            # Export feature importance jika ada
            if self.results['feature_importance'] is not None:
                fi_path = os.path.join(output_dir, 'feature_importance.csv')
                self.results['feature_importance'].to_csv(fi_path, index=False)
            
            print(f"✅ Hasil berhasil diekspor ke direktori: {output_dir}")
            print(f"   • Prediksi: {pred_path}")
            print(f"   • Metrics: {metrics_path}")
            if self.results['feature_importance'] is not None:
                print(f"   • Feature Importance: {fi_path}")
                
        except Exception as e:
            print(f"❌ Error saat export ke CSV: {e}")

    def get_hyperparameter_suggestions(self, estimator_name: str = None) -> Dict:
        """Get saran hyperparameter untuk model tertentu"""
        if not estimator_name and self.flaml_model:
            estimator_name = self.flaml_model.best_estimator
            
        # Hyperparameter suggestions berdasarkan estimator dan task type
        suggestions = {
            'lgbm': {
                'classification': {
                    'n_estimators': [100, 200, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7, -1],
                    'num_leaves': [15, 31, 63]
                },
                'regression': {
                    'n_estimators': [100, 200, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7, -1],
                    'num_leaves': [15, 31, 63]
                }
            },
            'xgboost': {
                'classification': {
                    'n_estimators': [100, 200, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'regression': {
                    'n_estimators': [100, 200, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'rf': {
                'classification': {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'regression': {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }
        }
        
        if estimator_name in suggestions:
            return suggestions[estimator_name].get(self.task_type, {})
        else:
            return {"message": f"Tidak ada saran hyperparameter untuk {estimator_name}"}

    def __repr__(self):
        """String representation of NoventisAutoML object"""
        status = "Trained" if self.flaml_model else "Not Trained"
        best_model = getattr(self.flaml_model, 'best_estimator', 'None') if self.flaml_model else 'None'
        
        return f"""
NoventisAutoML(
    task_type='{self.task_type}',
    target_column='{self.target_column}',
    status='{status}',
    best_estimator='{best_model}',
    data_shape={getattr(self, 'df', pd.DataFrame()).shape}
"""
    