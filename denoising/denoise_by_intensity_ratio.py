import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
import re
import logging
import argparse

# 添加项目根目录到路径以导入 utils
# 假设此脚本从项目根目录运行或我们设置了路径
current_dir = Path(__file__).resolve().parent
all_days_merged_dir = current_dir.parent
sys.path.insert(0, str(all_days_merged_dir))

from utils.config import Config

START_INTENSITY = 1
END_INTENSITY = 50

def setup_logger(log_dir):
    log_file = log_dir / "denoising_process.log"
    # 如果有现有的处理器，移除它们以避免重复日志
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers = []
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

def calculate_intensity_ratios(config, logger):
    logger.info(f"正在查找合并数据: {config.MERGED_DATA_DIR}")
    csv_files = list(config.MERGED_DATA_DIR.glob("*_merged.csv"))
    
    if not csv_files:
        logger.warning("未找到合并数据文件。")
        return

    for csv_file in csv_files:
        logger.info(f"正在处理 {csv_file.name}...")
        try:
            df = pd.read_csv(csv_file)
            
            # 识别元数据列
            metadata_cols = ['ID', 'names', 'labels', 'methods']
            # 特征列是其余的列
            feature_cols = [c for c in df.columns if c not in metadata_cols]
            
            # 准备结果列表
            results = []
            
            for index, row in df.iterrows():
                # 获取该物质的强度
                intensities = row[feature_cols].values.astype(float)
                
                # 过滤非零强度作为分母
                non_zero_intensities = intensities[intensities > 0]
                total_non_zero = len(non_zero_intensities)
                
                row_result = {
                    'ID': row.get('ID', ''),
                    'names': row.get('names', ''),
                    'labels': row.get('labels', ''),
                    'methods': row.get('methods', '')
                }
                
                if total_non_zero == 0:
                    # 避免除以零
                    for t in range(START_INTENSITY, END_INTENSITY + 1):
                        row_result[f'intensity={t}'] = 0.0
                else:
                    for t in range(START_INTENSITY, END_INTENSITY + 1):
                        # 统计强度 <= t 的特征数量
                        count = np.sum(non_zero_intensities <= t)
                        ratio = count / total_non_zero
                        row_result[f'intensity={t}'] = round(ratio, 4)  # 保留4位小数
                
                results.append(row_result)
            
            # 创建 DataFrame
            result_df = pd.DataFrame(results)
            
            # 保存
            type_name = csv_file.stem.replace('_merged', '')
            save_path = INTEN_MASS_RATIO_DIR / f"{type_name}_ratio.csv"
            result_df.to_csv(save_path, index=False)
            logger.info(f"已保存占比数据至 {save_path}")
            
        except Exception as e:
            logger.error(f"处理 {csv_file.name} 时出错: {e}")

def calculate_max_slope(target_dir, logger):
    logger.info(f"正在计算最大斜率，目标目录: {target_dir}...")
    if not target_dir.exists():
        logger.error(f"目录 {target_dir} 不存在。")
        return

    csv_files = list(target_dir.glob("*_ratio.csv"))
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # 查找 intensity 列
            intensity_cols = [c for c in df.columns if c.startswith('intensity=')]
            
            if not intensity_cols:
                logger.warning(f"在 {csv_file.name} 中未找到 intensity 列")
                continue

            # 按强度值排序
            # 从 'intensity=X' 中提取数字
            try:
                intensity_cols.sort(key=lambda x: int(x.split('=')[1]))
            except ValueError:
                logger.error(f"无法解析 {csv_file.name} 中的 intensity 列")
                continue
            
            max_slope_infos = []
            
            for index, row in df.iterrows():
                max_k = -float('inf')
                max_pair = None
                
                # 遍历相邻对
                for i in range(len(intensity_cols) - 1):
                    col1 = intensity_cols[i]
                    col2 = intensity_cols[i+1]
                    
                    val1 = float(row[col1])
                    val2 = float(row[col2])
                    
                    # 斜率 k = (y2 - y1) / (x2 - x1)
                    x1 = int(col1.split('=')[1])
                    x2 = int(col2.split('=')[1])
                    dx = x2 - x1
                    
                    if dx == 0: continue
                    
                    k = (val2 - val1) / dx
                    
                    if k > max_k:
                        max_k = k
                        max_pair = (col1, col2)
                
                if max_pair:
                    # 格式: （intensity=9，intensity=10，k=？）
                    info = f"（{max_pair[0]}，{max_pair[1]}，k={max_k:.4f}）"
                    max_slope_infos.append(info)
                else:
                    max_slope_infos.append("")
            
            df['max_slope_info'] = max_slope_infos
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            logger.info(f"已更新 {csv_file.name} 的最大斜率信息。")
            
        except Exception as e:
            logger.error(f"处理 {csv_file.name} 时出错: {e}")

def apply_denoising(inten_mass_ratio_dir, config, logger):
    logger.info(f"正在根据最大斜率信息应用去噪...")
    
    # 映射类型名称到比率文件
    ratio_files = list(inten_mass_ratio_dir.glob("*_ratio.csv"))
    
    for ratio_file in ratio_files:
        type_name = ratio_file.stem.replace('_ratio', '')
        merged_file = config.MERGED_DATA_DIR / f"{type_name}_merged.csv"
        
        if not merged_file.exists():
            logger.warning(f"未找到 {type_name} 的合并数据文件")
            continue
            
        logger.info(f"正在处理 {type_name}...")
        
        try:
            # 读取比率文件以获取阈值
            ratio_df = pd.read_csv(ratio_file)
            
            # 创建一个将 ID 映射到阈值的字典
            id_to_threshold = {}
            
            for index, row in ratio_df.iterrows():
                slope_info = row.get('max_slope_info', '')
                if pd.isna(slope_info) or not isinstance(slope_info, str):
                    continue
                
                # 解析斜率信息: （intensity=9，intensity=10，k=0.6980）
                # 我们想要第二个强度值 (10)，[,，]用于匹配任意的中英文逗号
                match = re.search(r'intensity=(\d+)[,，]\s*intensity=(\d+)', slope_info)
                if match:
                    threshold = int(match.group(2))
                    row_id = row.get('ID') # 假设 ID 列存在且唯一
                    if row_id:
                        id_to_threshold[row_id] = threshold
            
            # 读取合并数据
            merged_df = pd.read_csv(merged_file)
            
            # 识别特征列
            metadata_cols = ['ID', 'names', 'labels', 'methods']
            feature_cols = [c for c in merged_df.columns if c not in metadata_cols]
            num_features_before = len(feature_cols)
            
            # 应用去噪
            denoised_data = []
            
            for index, row in merged_df.iterrows():
                row_id = row.get('ID')
                threshold = id_to_threshold.get(row_id, 0) # 如果未找到阈值，默认为 0
                
                logger.info(f"物质: {row_id}, 去噪阈值: {threshold}")

                # 复制行数据
                new_row = row.copy()
                
                if threshold > 0:
                    # 获取特征值
                    vals = row[feature_cols].values.astype(float)
                    # 应用阈值: 将 <= threshold 的值设置为 0
                    vals[vals <= threshold] = 0
                    # 更新行
                    new_row[feature_cols] = vals
                
                denoised_data.append(new_row)
            
            denoised_df = pd.DataFrame(denoised_data)
            
            # 移除全为 0 的特征列
            cols_to_drop = [col for col in feature_cols if (denoised_df[col] == 0).all()]
            if cols_to_drop:
                logger.info(f"丢弃 {len(cols_to_drop)} 个全为0的特征列。")
                denoised_df.drop(columns=cols_to_drop, inplace=True)
            
            # 计算剩余特征
            # 注意: denoised_df 列包含元数据列
            current_feature_cols = [c for c in denoised_df.columns if c not in metadata_cols]
            num_features_after = len(current_feature_cols)

            # 保存到新目录
            save_path = config.MERGED_DATA_DENOISED_DIR / f"{type_name}_merged.csv"
            denoised_df.to_csv(save_path, index=False)
            logger.info(f"已保存去噪数据至 {save_path}")
            logger.info(f"文件 {type_name}: 去噪前特征列数: {num_features_before}, 去噪后保留特征列数: {num_features_after}")
            
        except Exception as e:
            logger.error(f"处理 {type_name} 时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Denoise data by intensity ratio')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    
    args = parser.parse_args()
    
    config = Config(output_dir=args.output_dir)
    
    INTEN_MASS_RATIO_DIR = config.OUTPUT_DIR / "inten_mass_ratio"
    MERGED_DATA_DENOISED_DIR = config.MERGED_DATA_DENOISED_DIR
    os.makedirs(INTEN_MASS_RATIO_DIR, exist_ok=True)
    
    logger = setup_logger(MERGED_DATA_DENOISED_DIR)
    
    # 1. 计算比率
    calculate_intensity_ratios(config, logger)
    
    # 2. 计算最大斜率
    calculate_max_slope(INTEN_MASS_RATIO_DIR, logger)
    
    # 3. 应用去噪
    apply_denoising(INTEN_MASS_RATIO_DIR, config, logger)
