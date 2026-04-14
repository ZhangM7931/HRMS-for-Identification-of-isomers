from __future__ import annotations

from pathlib import Path
import sys

# 获取项目根目录的绝对路径
project_root = Path(__file__).resolve().parent.parent.parent
project_root_str = str(project_root)

# 确保项目根目录在 sys.path 中（使用绝对路径）
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

import pandas as pd
from utils.config import raw_path, sql_path
from utils.DBService import DBService, RawData, DataType
from utils.FileService import FileService

excel_file_list = list((raw_path / "实验条件探索").glob("*/*.xlsx"))

if __name__ == "__main__":
    for file_path in excel_file_list:
        type_ = file_path.stem.split("-")[0]
        method = file_path.parent.stem
        with pd.ExcelFile(file_path) as reader:
            for name in reader.sheet_names:
                data = reader.parse(name, index_col=1).iloc[:, 1]
                label = "-".join(name.split("-")[:-1])
                queryed_data: RawData | None = DBService.get_raw_data_by_name_and_method(name, method)
                if queryed_data is None:
                    raw_data: RawData = DBService.create_raw_data(
                        name=name, type=type_, label=label, data=data.to_dict(), method=method
                    )
                    DBService.add_raw_data(raw_data)
                else:
                    print(f"数据已存在: {name} - {method}")
                # break
        # break
