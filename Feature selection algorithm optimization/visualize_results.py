import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Add project root to sys.path to ensure imports work
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from utils.config import Config
from utils.data_loader import load_and_preprocess_data
from features.visualizer import DataVisualizer
from features.evaluator import FeatureEvaluator
from features.selector import FeatureSelector

# 设置日志
logging.basicConfig(
    level=logging.WARNING, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
warnings.filterwarnings('ignore')

def main():
    config = Config()
    
    # 获取药物数据文件列表
    data_files = list(config.MERGED_DATA_DENOISED_DIR.glob("*.csv"))
    
    if not data_files:
        logger.error(f"在 {config.MERGED_DATA_DENOISED_DIR} 中未找到数据文件")
        return

    evaluator = FeatureEvaluator()

    for data_file in data_files:
        drug_name = data_file.stem.replace("_merged", "")
        
        drug_results_dir = config.RESULTS_DIR / drug_name
        drug_plots_dir = config.PLOTS_DIR / drug_name
        
        if not drug_results_dir.exists():
            # logger.warning(f"未找到 {drug_name} 的结果目录。跳过。")
            continue

        # 设置药物特定日志
        log_file = drug_results_dir / "only_visualization.log"
        # 移除现有的文件处理程序以避免重复
        for h in logger.handlers[:]:
            if isinstance(h, logging.FileHandler):
                logger.removeHandler(h)
                
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        logger.info(f"================ 处理药物: {drug_name} ================")
            
        # 1. 加载数据
        try:
            X, y, meta = load_and_preprocess_data(data_file)
            logger.info(f"数据已加载: {X.shape[0]} 样本, {X.shape[1]} 特征")
        except Exception as e:
            logger.error(f"加载 {drug_name} 数据失败: {e}")
            continue
            
        # 编码标签
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        visualizer = DataVisualizer(drug_plots_dir)
        
        # 2. 可视化原始数据 (所有样本)
        logger.info("正在绘制原始特征的 PCA...")
        visualizer = DataVisualizer(drug_plots_dir)
        # 使用原始标签 y 而不是编码后的 y_encoded
        visualizer.plot_pca(X, y, title="PCA - 原始特征 (所有样本)")
        
        # 3. 加载最优参数并复现特征选择
        # 在 drug_results_dir 中查找子目录
        feature_selection_dir = drug_results_dir / 'feature_selection'
        if not feature_selection_dir.exists():
            logger.warning(f"未找到 {feature_selection_dir} 目录。跳过特征选择可视化。")
            continue

        fs_methods = [d for d in feature_selection_dir.iterdir() if d.is_dir()]
        
        metrics_list = []
        
        # 初始化特征选择器
        feature_selector = FeatureSelector(config.ALG_CONFIG_PATH)
        
        for method_dir in fs_methods:
            method_name = method_dir.name
            optimal_params_file = method_dir / "optimal_params.csv"
            
            if not optimal_params_file.exists():
                logger.warning(f"未找到 {optimal_params_file}，跳过 {method_name}。")
                continue
                
            logger.info(f"正在处理来自以下方法的特征: {method_name}")
            
            try:
                # 读取最优参数
                optimal_params_df = pd.read_csv(optimal_params_file)
                optimal_params = optimal_params_df.iloc[0].to_dict()
                # 移除 'best_score' 如果存在
                optimal_params.pop('best_score', None)
                
                # 使用最优参数选择特征
                # 注意：select_features 需要 DataFrame 格式的 X
                X_df = pd.DataFrame(X, columns=X.columns)
                selected_features = feature_selector.select_features(X_df, y_encoded, method_name, **optimal_params)
                
                if not selected_features:
                    logger.warning(f"{method_name} 未选择任何特征")
                    continue
                    
                # 保存特征到txt（如果不存在）
                feature_file = method_dir / "selected_features.txt"
                if not feature_file.exists():
                    with open(feature_file, "w", encoding='utf-8') as f:
                        f.write("\n".join(selected_features))
                    logger.info(f"特征列表已保存至 {feature_file}")
                
                # 数据子集
                X_sel = X[selected_features]
                
                # 可视化 (所有样本) - 保存到方法目录
                visualizer_method = DataVisualizer(method_dir)
                # 使用原始标签 y
                visualizer_method.plot_pca(X_sel, y, title=f"PCA - {method_name} (复现)")
                
                # 计算指标 (所有样本)
                metrics = evaluator.evaluate_feature_subset(X_sel, y_encoded)
                metrics['fs_method'] = method_name
                metrics_list.append(metrics)
                
            except Exception as e:
                logger.error(f"处理 {method_name} 时出错: {e}")
                
        # 保存比较指标
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            # 重新排序列
            cols = ['fs_method', 'n_features', 'redundancy', 'silhouette', 'calinski_harabasz', 'adjusted_rand_index', 'intra_inter_ratio', 'cv_acc_baseline']
            cols = [c for c in cols if c in metrics_df.columns]
            metrics_df = metrics_df[cols]
            
            output_path = feature_selection_dir / "feature_selection_comparison_metrics.csv"
            metrics_df.to_csv(output_path, index=False)
            logger.info(f"已保存比较指标至 {output_path}")
            
        logger.removeHandler(file_handler)

if __name__ == "__main__":
    main()
