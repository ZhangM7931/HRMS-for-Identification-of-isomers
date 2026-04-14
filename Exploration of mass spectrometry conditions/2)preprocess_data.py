from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.DBService import DBService, RawData, DataType
import pandas as pd
import numpy as np
from utils.config import processed_path

def merge_close_peaks(series: pd.Series, precision: float = 0.0001, method: str = 'sum') -> pd.Series:
    """
    整合相近的质谱峰位置
    
    Args:
        series: 质谱数据Series，index为mass/charge，value为intensity
        precision: 精度阈值，默认0.0001
        method: 合并方法，'sum'表示求和，'mean'表示取平均，默认'sum'
    
    Returns:
        整合后的质谱数据Series
    """
    if series.empty:
        return series
    
    # 获取峰位置和强度
    mz_values = series.index.values
    intensity_values = series.values
    
    # 按m/z值排序
    sorted_indices = np.argsort(mz_values)
    sorted_mz = mz_values[sorted_indices]
    sorted_intensity = intensity_values[sorted_indices]
    
    # 合并相近的峰
    merged_mz = []
    merged_intensity = []
    
    i = 0
    while i < len(sorted_mz):
        current_mz = sorted_mz[i]
        current_intensity = sorted_intensity[i]
        count = 1
        
        # 查找所有相近的峰
        j = i + 1
        while j < len(sorted_mz) and abs(sorted_mz[j] - current_mz) <= precision:
            current_intensity += sorted_intensity[j]
            count += 1
            j += 1
        
        # 计算加权平均m/z值
        if count > 1:
            # 使用强度作为权重计算加权平均
            weighted_mz = current_mz
            total_intensity = current_intensity
            for k in range(i + 1, j):
                weighted_mz = (weighted_mz * total_intensity + sorted_mz[k] * sorted_intensity[k]) / (total_intensity + sorted_intensity[k])
                total_intensity += sorted_intensity[k]
            merged_mz.append(weighted_mz)
        else:
            merged_mz.append(current_mz)
        
        # 根据method参数选择合并方式
        if method == 'mean' and count > 1:
            merged_intensity.append(current_intensity / count)  # 取平均
        else:
            merged_intensity.append(current_intensity)  # 求和（默认）
        i = j
    
    # 创建新的Series
    merged_series = pd.Series(merged_intensity, index=merged_mz, name=series.name)
    
    # 按m/z值排序
    merged_series = merged_series.sort_index()
    
    # 处理可能的重复index（由于精度问题）
    merged_series = merged_series.groupby(merged_series.index).sum()
    
    return merged_series

types = ["咪酯类", "尼秦类", "芬太尼类", "苯丙胺类", "美托咪酯"]
METHODS = ["CE20", "CE40+-20", "CE60", "KE10", "KE15", "KE20"]

# record =  DBService.get_raw_data_by_type_and_method(type=types[0], method=METHODS[0])[0]

# print(1)

for type_ in types:
    for method in METHODS:
        records: list[RawData] = DBService.get_raw_data_by_type_and_method(type=type_, method=method)
        if len(records) == 0:
            continue
        names: list[str] = []
        labels: list[str] = []
        methods: list[str] = []
        series_data: list[pd.DataFrame] = []
        for record in records:
            names.append(record.name)
            labels.append(record.label)
            methods.append(record.method)


            tmp_series : pd.Series = pd.Series(record.data, name=f"{record.name}-{record.method}")

            tmp_series.index = tmp_series.index.astype(float)
            len1 = len(tmp_series)
            # 清除噪声
            tmp_series = tmp_series[tmp_series / tmp_series.max() > 0.01]
            len2 = len(tmp_series)
            # 整合相近峰位置，精度为0.01
            # method='sum' 表示求和，method='mean' 表示取平均
            tmp_series = merge_close_peaks(tmp_series, precision=0.1, method='sum')
            len3 = len(tmp_series)

            # 四舍五入到2位小数
            tmp_series.index = tmp_series.index.round(1)
            
            # 处理重复的index：将相同index的值合并
            # 这里使用sum()，如果您想用平均可以改为mean()
            tmp_series = tmp_series.groupby(tmp_series.index).sum()

            series_data.append(tmp_series.to_frame())
        series_data = pd.concat(series_data, axis=1).T
        data = pd.DataFrame([], index=series_data.index)
        data["names"] = names
        data["labels"] = labels
        data["methods"] = methods
        data = pd.concat([data, series_data], axis=1)
        data.to_excel(processed_path / f"{type_}-{method}.xlsx")

