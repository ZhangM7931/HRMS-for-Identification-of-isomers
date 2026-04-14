
import optuna
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import zhplot
logger = logging.getLogger(__name__)

def objective(trial, model_name, X_train, y_train, X_test, y_test):
    if model_name == 'RF':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
        }
        model = RandomForestClassifier(**params, random_state=42)
        
    elif model_name == 'SVM':
        params = {
            'C': trial.suggest_float('C', 0.1, 100, log=True),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear'])
        }
        model = SVC(**params, probability=True, random_state=42)
        
    elif model_name == 'KNN':
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'p': trial.suggest_int('p', 1, 2)
        }
        model = KNeighborsClassifier(**params)
        
    elif model_name == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0)
        }
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', random_state=42)
        
    else:
        return 0

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

def optimize_model(model_name, X_train, y_train, X_test, y_test, n_trials=20):
    """
    对给定模型运行 Optuna 优化。
    """
    logger.info(f"正在使用 Optuna 优化 {model_name}...")
    # 设置 optuna 的日志级别，避免过多输出
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, model_name, X_train, y_train, X_test, y_test), n_trials=n_trials)
    
    logger.info(f"最佳参数: {study.best_params}")
    logger.info(f"最佳准确率: {study.best_value}")
    
    # 重新训练最佳模型
    if model_name == 'RF':
        best_model = RandomForestClassifier(**study.best_params, random_state=42)
    elif model_name == 'SVM':
        best_model = SVC(**study.best_params, probability=True, random_state=42)
    elif model_name == 'KNN':
        best_model = KNeighborsClassifier(**study.best_params)
    elif model_name == 'XGBoost':
        best_model = XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    best_model.fit(X_train, y_train)
    return best_model

def explain_model_shap(model, X_train, X_test, feature_names, save_dir):
    """
    生成 SHAP 图。
    """
    logger.info("正在生成 SHAP 解释...")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 选择解释器
    # 为了速度，如果 X_train 很大，使用背景数据的子集
    background = shap.sample(X_train, 100) if len(X_train) > 100 else X_train
    
    try:
        explainer = shap.Explainer(model, background)
        shap_values = explainer(X_test)
    except Exception as e:
        logger.warning(f"SHAP Explainer 失败，尝试 KernelExplainer: {e}")
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test)
        # KernelExplainer 返回分类列表，如果是二分类则取类 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # 假设是二分类且对正类感兴趣
            
    # 摘要图
    plt.figure()
    try:
        logger.info(f"shap_values type: {type(shap_values)}")
        if isinstance(shap_values, shap.Explanation):
             logger.info(f"shap_values shape: {shap_values.shape}")
             
             # 处理多类输出 (n_samples, n_features, n_classes)
             if len(shap_values.shape) == 3:
                 logger.info("检测到多类 SHAP 值，选择类 1")
                 shap_values = shap_values[:, :, 1]
             
             if len(shap_values.shape) > 1 and shap_values.shape[1] == 1:
                 logger.info("只有一个特征，绘制散点图代替 beeswarm")
                 # 对于单特征，直接在当前画布绘制散点图作为 summary
                 plt.scatter(X_test.iloc[:, 0], shap_values.values[:, 0])
                 plt.xlabel(feature_names[0])
                 plt.ylabel('SHAP value')
                 plt.title('SHAP Summary (Single Feature)')
             else:
                 shap.plots.beeswarm(shap_values, show=False)
        else:
             # Legacy array
             logger.info(f"shap_values shape: {np.array(shap_values).shape}")
             if X_test.shape[1] == 1:
                 logger.info("只有一个特征，跳过 summary_plot 的排序或使用默认绘图")
                 # 同样为 legacy array 绘制散点图
                 plt.figure()
                 # 假设 shap_values 是 (n_samples, 1) 或 (n_samples,)
                 vals = np.array(shap_values).flatten()
                 plt.scatter(X_test.iloc[:, 0], vals)
                 plt.xlabel(feature_names[0])
                 plt.ylabel('SHAP value')
                 plt.title('SHAP Dependence Plot (Single Feature)')
                 plt.savefig(save_dir / "shap_dependence.png", bbox_inches='tight')
                 plt.close()
                 
                 shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, sort=False)
             else:
                 shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    except Exception as e:
        logger.error(f"绘制 summary_plot 失败: {e}")
    
    plt.savefig(save_dir / "shap_summary.png", bbox_inches='tight')
    plt.close()
    
    # 条形图
    plt.figure()
    try:
        # 检查 shap_values 是 Explanation 对象还是数组
        if isinstance(shap_values, shap.Explanation):
            shap.plots.bar(shap_values, show=False)
        else:
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    except Exception as e:
        logger.error(f"绘制 bar plot 失败: {e}")

    plt.savefig(save_dir / "shap_bar.png", bbox_inches='tight')
    plt.close()
    logger.info("SHAP 解释完成")
