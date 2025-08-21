import pandas as pd
import numpy as np

# Model Selection
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import xgboost as xgb

import lightgbm as lgb
import catboost as cb

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR

# Evaluation Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_model_classification(model_name: str, random_state: int = 42):
    """
    Memuat model klasifikasi berdasarkan nama yang diberikan.

    Parameters:
    - model_name (str): Nama model. Pilihan: 'logistic_regression', 'decision_tree', 
                        'random_forest', 'extra_trees', 'gradient_boosting', 
                        'adaboost', 'svm', 'xgboost', 'lightgbm', 'catboost'.
    - random_state (int): Random state untuk reproduktivitas.

    Returns:
    - Objek model klasifikasi yang telah di-inisialisasi.
    """
    models = {
        'logistic_regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'decision_tree': DecisionTreeClassifier(random_state=random_state),
        'random_forest': RandomForestClassifier(random_state=random_state),
        'extra_trees': ExtraTreesClassifier(random_state=random_state),
        'gradient_boosting': GradientBoostingClassifier(random_state=random_state),
        'adaboost': AdaBoostClassifier(random_state=random_state),
        'svm': SVC(probability=True, random_state=random_state), # probability=True untuk predict_proba
        'xgboost': xgb.XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss'),
        'lightgbm': lgb.LGBMClassifier(random_state=random_state, verbose=-1),
        'catboost': cb.CatBoostClassifier(random_state=random_state, verbose=0)
    }
    
    model = models.get(model_name.lower())
    if model is None:
        raise ValueError(f"Model '{model_name}' tidak dikenal. Pilihan yang tersedia: {list(models.keys())}")
    return model

def load_model_regression(model_name: str, random_state: int = 42):
    """
    Memuat model regresi berdasarkan nama yang diberikan.

    Parameters:
    - model_name (str): Nama model. Pilihan: 'linear_regression', 'ridge', 'lasso',
                        'decision_tree', 'random_forest', 'extra_trees', 
                        'gradient_boosting', 'adaboost', 'svr', 'xgboost', 
                        'lightgbm', 'catboost'.
    - random_state (int): Random state untuk reproduktivitas.

    Returns:
    - Objek model regresi yang telah di-inisialisasi.
    """
    models = {
        'linear_regression': LinearRegression(),
        'ridge': Ridge(random_state=random_state),
        'lasso': Lasso(random_state=random_state),
        'decision_tree': DecisionTreeRegressor(random_state=random_state),
        'random_forest': RandomForestRegressor(random_state=random_state),
        'extra_trees': ExtraTreesRegressor(random_state=random_state),
        'gradient_boosting': GradientBoostingRegressor(random_state=random_state),
        'adaboost': AdaBoostRegressor(random_state=random_state),
        'svr': SVR(),
        'xgboost': xgb.XGBRegressor(random_state=random_state),
        'lightgbm': lgb.LGBMRegressor(random_state=random_state, verbose=-1),
        'catboost': cb.CatBoostRegressor(random_state=random_state, verbose=0)
    }
    
    model = models.get(model_name.lower())
    if model is None:
        raise ValueError(f"Model '{model_name}' tidak dikenal. Pilihan yang tersedia: {list(models.keys())}")
    return model

def run_classification_pipeline(df: pd.DataFrame, model_name: str, target_column: str = 'target', test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    """
    Menjalankan pipeline klasifikasi lengkap.
    
    Parameters:
    - df (pd.DataFrame): Dataset
    - model_name (str): Nama model yang akan digunakan
    - target_column (str): Nama kolom target
    - test_size (float): Proporsi data test (default: 0.2)
    - random_state (int): Random state untuk reproduktivitas
    
    Returns:
    - pd.DataFrame: DataFrame dengan hasil prediksi
    """
    if target_column not in df.columns:
        raise ValueError(f"Kolom target '{target_column}' tidak ditemukan dalam dataset")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    model = load_model_classification(model_name, random_state=random_state)

    print(f"Melatih model {model_name}...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    result_df = X_test.copy()
    result_df['actual_target'] = y_test
    result_df['predicted_target'] = predictions
    
    return result_df

def run_regression_pipeline(df: pd.DataFrame, model_name: str, target_column: str = 'target', test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    """
    Menjalankan pipeline regresi lengkap.
    
    Parameters:
    - df (pd.DataFrame): Dataset
    - model_name (str): Nama model yang akan digunakan
    - target_column (str): Nama kolom target
    - test_size (float): Proporsi data test (default: 0.2)
    - random_state (int): Random state untuk reproduktivitas
    
    Returns:
    - pd.DataFrame: DataFrame dengan hasil prediksi
    """
    if target_column not in df.columns:
        raise ValueError(f"Kolom target '{target_column}' tidak ditemukan dalam dataset")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    model = load_model_regression(model_name, random_state=random_state)
    
    print(f"Melatih model {model_name}...")
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)

    result_df = X_test.copy()
    result_df['actual_target'] = y_test
    result_df['predicted_target'] = predictions
    
    return result_df

def eval_regression(y_true, y_pred):
    """
    Mengevaluasi hasil prediksi regresi.
    
    Parameters:
    - y_true: Nilai aktual
    - y_pred: Nilai prediksi
    
    Returns:
    - dict: Dictionary berisi metrik evaluasi
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2_score': r2_score(y_true, y_pred)
    }
    
    print("=== Hasil Evaluasi Regresi ===")
    print(f"MSE (Mean Squared Error): {metrics['mse']:.4f}")
    print(f"RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}")
    print(f"MAE (Mean Absolute Error): {metrics['mae']:.4f}")
    print(f"R² Score: {metrics['r2_score']:.4f}")
    
    return metrics

def eval_classification(y_true, y_pred, average='weighted'):
    """
    Mengevaluasi hasil prediksi klasifikasi.
    
    Parameters:
    - y_true: Nilai aktual
    - y_pred: Nilai prediksi
    - average: Metode averaging untuk multi-class (default: 'weighted')
    
    Returns:
    - dict: Dictionary berisi metrik evaluasi
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average),
        'recall': recall_score(y_true, y_pred, average=average),
        'f1_score': f1_score(y_true, y_pred, average=average)
    }
    
    print("=== Hasil Evaluasi Klasifikasi ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred))
    
    return metrics

