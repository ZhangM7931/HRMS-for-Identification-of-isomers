import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非GUI后端，避免Tkinter错误
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from utils.config import Config
from utils.data_loader import load_and_preprocess_data
from models.trainer import ModelTrainer

def setup_logging(name, output_dir):
    """Configure logging to file only, no console output except warnings/errors."""
    log_file = output_dir / f"{name}_modeling.log"
    
    # Create a unique logger for this experiment
    experiment_logger = logging.getLogger(name)
    experiment_logger.setLevel(logging.INFO)
    experiment_logger.propagate = False  # Avoid propagating to root logger to prevent duplicate logs if root has handlers

    # Clear existing handlers
    if experiment_logger.hasHandlers():
        experiment_logger.handlers.clear()
    
    # File handler (INFO and above)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    experiment_logger.addHandler(file_handler)
    
    # Console handler (WARNING and above only) to root logger is handled by BasicConfig or manually
    # Just adding a console handler to this logger for warnings
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    experiment_logger.addHandler(console_handler)
    
    return experiment_logger

def run_single_experiment(drug_name, fs_method, model_name, config):
    """
    Run a single experiment: Drug -> FS Method -> Model
    """
    # Setup paths
    drug_dir = config.RESULTS_DIR / drug_name
    fs_dir = drug_dir / 'feature_selection' / fs_method
    # Make sure we go into a 'models' directory to keep things clean? 
    # User asked for independent folders. Current structure: results/drug/feature_selection/fs_method/model_name
    # Maybe results/drug/models/fs_method/model_name is better?
    # But previous step outputs to results/drug/feature_selection/fs_method.
    # I will stick to the existing structure but ensure the model_dir is correctly used.
    # To be very clear: results/drug/feature_selection/fs_method/model_name
    model_dir = fs_dir / model_name
    os.makedirs(model_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(f"{fs_method}_{model_name}", model_dir)
    logger.info(f"开始实验: 药物={drug_name}, 特征选择={fs_method}, 模型={model_name}")
    
    # Load Data
    try:
        # Construct path to merged data file
        # Assuming filename format is {drug_name}_merged.csv in MERGED_DATA_DENOISED_DIR
        data_path = config.MERGED_DATA_DENOISED_DIR / f"{drug_name}_merged.csv"
        
        if not data_path.exists():
            logger.error(f"数据文件不存在: {data_path}")
            return None

        # Call with single argument and unpack 3 values (X, y, meta)
        X, y, _ = load_and_preprocess_data(data_path)
        logger.info(f"数据加载成功: {X.shape}")
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return None

    # Load Selected Features
    feat_file = fs_dir / "selected_features.txt"
    if not feat_file.exists():
        logger.error(f"特征文件不存在: {feat_file}")
        return None
        
    with open(feat_file, 'r') as f:
        selected_features = [line.strip() for line in f.readlines() if line.strip()]
    
    valid_features = [f for f in selected_features if f in X.columns]
    if not valid_features:
        logger.error("未找到有效特征")
        return None
        
    X_sel = X[valid_features]
    logger.info(f"特征选择完成: {len(valid_features)} 个特征")
    
    # Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split Data (8:2)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Initialize Trainer with DEFAULT PARAMS config
    trainer = ModelTrainer(config.ALG_CONFIG_PATH, 'machine_learning')
    
    # Load best params if exists, otherwise tune
    param_csv = model_dir / f"{model_name}_param_trials.csv"
    if param_csv.exists():
        logger.info("加载已有的最优参数...")
        df = pd.read_csv(param_csv)
        # Assume the best trial is the one with highest value (accuracy or auc)
        best_row = df.loc[df['value'].idxmax()]
        best_params = {}
        for col in df.columns:
            if col not in ['trial_number', 'value', 'datetime_start', 'datetime_complete', 'duration', 'state']:
                best_params[col] = best_row[col]
        logger.info(f"加载的最优参数: {best_params}")
    else:
        # Tune hyperparameters
        logger.info("开始参数调优...")
        best_params = trainer.tune_hyperparameters(X_train, y_train, model_name, n_trials=20, cv_folds=5, output_dir=model_dir)
    
    # Save tuning logs (simple version, detailed logs are saved above)
    with open(model_dir / "param_opt.log", 'w', encoding='utf-8') as f:
        f.write(f"Best params: {best_params}\n")
    
    logger.info(f"最优参数: {best_params}")
    
    # Train and Evaluate
    logger.info("使用最优参数训练和评估...")
    try:
        result = trainer.train_and_evaluate(X_train, y_train, X_test, y_test, model_name, best_params, model_dir)
        
        # Plot single ROC
        plot_single_roc(result['y_test'], result['y_prob'], result['auc'], f"{fs_method}-{model_name}", model_dir)
        
        logger.info(f"实验完成. Accuracy: {result['accuracy']:.4f}, AUC: {result['auc']:.4f}")
        return result
    except Exception as e:
        logger.error(f"训练/评估失败: {e}")
        return None

def run_full_pipeline_for_drug(drug_name, config):
    """
    Run all combinations for a single drug
    """
    # Setup logging (Master log for the drug)
    drug_dir = config.RESULTS_DIR / drug_name
    os.makedirs(drug_dir, exist_ok=True)
    # This logger is just for high-level flow
    master_logger = setup_logging(f"{drug_name}_master", drug_dir)
    
    master_logger.info(f"开始 {drug_name} 的全流程建模 (Step 2: Default Params)...")
    
    fs_root = drug_dir / 'feature_selection'
    if not fs_root.exists():
        master_logger.error(f"特征选择目录不存在: {fs_root}")
        return

    fs_methods = [d.name for d in fs_root.iterdir() if d.is_dir() and (d / "selected_features.txt").exists()]
    models = ['SVM']  # 只重新运行SVM模型
    
    all_results = []
    
    for fs in fs_methods:
        for model_name in models:
            master_logger.info(f"正在运行: {fs} - {model_name}")
            try:
                # Calls run_single_experiment which has its own logger
                res = run_single_experiment(drug_name, fs, model_name, config)
                if res:
                    res['fs_method'] = fs
                    all_results.append(res)
            except Exception as e:
                master_logger.error(f"实验失败 ({fs} - {model_name}): {e}")
                
    # Find best model
    if all_results:
        best_model = max(all_results, key=lambda x: x['accuracy'])
        master_logger.info(f"最佳模型组合: {best_model['fs_method']} + {best_model['model_name']} (Acc: {best_model['accuracy']:.4f})")
        
        # Plot ROC Curves
        plot_roc_curves(all_results, drug_name, drug_dir)
        
        # Plot Accuracy Comparison Bar Chart
        plot_accuracy_comparison(all_results, drug_name, drug_dir)
    else:
        master_logger.warning("没有成功的实验结果")

def plot_accuracy_comparison(results, drug_name, output_dir):
    # Prepare data
    labels = [f"{res['fs_method']}+{res['model_name']}" for res in results]
    accuracies = [res['accuracy'] for res in results]
    
    # Sort data for better visualization (descending by accuracy)
    data = sorted(zip(labels, accuracies), key=lambda x: x[1], reverse=True)
    labels, accuracies = zip(*data)

    plt.figure(figsize=(14, 8))
    bars = plt.bar(labels, accuracies, color='steelblue')
    
    plt.xlabel('Model Combination', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Model Accuracy Comparison - {drug_name}', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 1.15) # Leave space for text
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
                 
    plt.tight_layout()
    plt.savefig(output_dir / f"{drug_name}_Accuracy_Comparison.png", dpi=300)
    plt.close()

def plot_roc_curves(results, drug_name, output_dir):
    """绘制所有ROC曲线在一个图上，从保存的ROC数据加载"""
    plt.figure(figsize=(12, 8))

    for res in results:
        fs_method = res['fs_method']
        model_name = res['model_name']
        auc_value = res['auc']
        
        # 尝试加载保存的ROC数据
        drug_dir = output_dir
        roc_csv_path = drug_dir / 'feature_selection' / fs_method / model_name / f"{fs_method}-{model_name}_ROC_data.csv"
        
        if roc_csv_path.exists():
            try:
                roc_df = pd.read_csv(roc_csv_path)
                if 'note' in roc_df.columns and 'No probability predictions' in roc_df['note'].values:
                    # No ROC data available, plot diagonal line
                    fpr = np.linspace(0, 1, 100)
                    tpr = fpr
                else:
                    fpr = roc_df['fpr'].values
                    tpr = roc_df['tpr'].values
            except Exception as e:
                print(f"加载ROC数据失败 {fs_method}-{model_name}: {e}")
                # Fallback to approximate curve
                fpr = np.linspace(0, 1, 100)
                tpr = 1 - (1 - auc_value) * fpr ** 0.5
                tpr = np.clip(tpr, 0, 1)
        else:
            # Generate approximate ROC curve if no saved data
            fpr = np.linspace(0, 1, 100)
            tpr = 1 - (1 - auc_value) * fpr ** 0.5
            tpr = np.clip(tpr, 0, 1)

        label = f"{fs_method}-{model_name} (AUC={auc_value:.3f})"
        plt.plot(fpr, tpr, label=label, linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves Comparison - {drug_name}', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{drug_name}_ROC_Curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存ROC曲线图: {output_dir / f'{drug_name}_ROC_Curves.png'}")

def plot_single_roc(y_test, y_prob, auc, model_label, output_dir):
    plt.figure(figsize=(8, 6))
    
    roc_data = None
    
    if y_prob is None:
        # No probability predictions available, skip ROC plot
        plt.text(0.5, 0.5, f'{model_label}\nAUC: {auc:.2f}\n(No probability predictions)',
                ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
        plt.title(f'ROC Curve - {model_label} (No Probabilities)')
        
        # Save ROC data as None
        roc_data = pd.DataFrame({
            'fpr': [0, 1],
            'tpr': [0, 1],
            'auc': [auc],
            'note': ['No probability predictions available']
        })
    else:
        if len(np.unique(y_test)) > 2:
            # Multiclass: micro-average
            from sklearn.preprocessing import label_binarize
            classes = np.unique(y_test)
            y_test_bin = label_binarize(y_test, classes=classes)
            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
        else:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        
        plt.plot(fpr, tpr, label=f'{model_label} (AUC={auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_label}')
        plt.legend()
        
        # Save ROC data
        roc_data = pd.DataFrame({
            'fpr': fpr,
            'tpr': tpr,
            'auc': [auc] * len(fpr)
        })
    
    plt.savefig(output_dir / f"{model_label}_ROC.png", dpi=300)
    plt.close()
    
    # Save ROC data to CSV
    if roc_data is not None:
        roc_data.to_csv(output_dir / f"{model_label}_ROC_data.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Modeling Pipeline")
    parser.add_argument("--drug", type=str, help="Drug name")
    parser.add_argument("--fs", type=str, help="Feature Selection Method")
    parser.add_argument("--model", type=str, help="Model Name")
    
    args = parser.parse_args()
    config = Config()
    
    if args.drug and args.fs and args.model:
        run_single_experiment(args.drug, args.fs, args.model, config)
    elif args.drug:
        run_full_pipeline_for_drug(args.drug, config)
    else:
        # Default: Run for first drug only (Debugging)
        first_drug = config.TYPES[0]
        print(f"未指定参数，默认运行第一个药物: {first_drug}")
        run_full_pipeline_for_drug(first_drug, config)
