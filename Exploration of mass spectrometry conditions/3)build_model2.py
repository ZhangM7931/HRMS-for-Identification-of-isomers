from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import shap
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
from utils.config import processed_path, result_path
from utils.DBService import DataType
from utils.validation import val_loocv, val_test
from utils.metrics import mcls
from utils.shap import shap_importance

from xgboost import XGBClassifier

algo = XGBClassifier
results = []
if not result_path.exists():
    result_path.mkdir()

types = ["咪酯类", "尼秦类", "芬太尼类", "苯丙胺类", "美托咪酯"]
METHODS = ["CE20", "CE40+-20", "CE60", "KE10", "KE15", "KE20"]

for type_ in types:
    for method in METHODS:
        if not (processed_path / f"{type_}-{method}.xlsx").exists():
            continue
        data = pd.read_excel(processed_path / f"{type_}-{method}.xlsx", index_col=0).iloc[:, 1:]
        X = data.iloc[:, 2:]
        label = data.iloc[:, 0]
        label = label.apply(lambda x: x.lower())
        if type_ == "尼秦类":
            label[label=="butonitazne"] = "butonitazene"
            label[label=="sec-butonitazne"] = "sec-butonitazene"
        elif type_ == "芬太尼类":
            label[label=="丁酰芬太尼"] = "butyrylfentanyl"
            label[label=="异丁酰芬太尼"] = "isobutyrylfentanyl"
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(label)
        X = X.fillna(0)
        X.columns = X.columns.astype(str)
        X = X.astype(float)
        X[:] = StandardScaler().fit_transform(X.astype(float))
        
        importance = shap_importance(algo, X, y)
        
        X_train, y_train = X.loc[:, importance.index[:10]].copy(), y
        
        loocv_scatter = val_loocv(algo, X_train, y_train)
        loocv_mcls = mcls(*loocv_scatter)
        
        results.append([
            method,
            type_,
            loocv_mcls["accuracy"]
            ])
results = pd.DataFrame(results, columns=["method", "type", "accuracy"])

# 绘制柱状图
fig, ax = plt.subplots(figsize=(12, 6))
results_pivot = results.pivot(columns="type", index="method", values="accuracy")
results_pivot.plot(kind="bar", ax=ax, width=0.8)

# 在柱状图上添加数值标签
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', rotation=0, fontsize=10, padding=3)

ax.set_xlabel("化合物类型", fontsize=12)
ax.set_ylabel("准确率", fontsize=12)
ax.set_title("不同方法和化合物类型的准确率对比", fontsize=14)
ax.legend(title="实验方法", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(result_path / "accuracy_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# 保存结果
results.to_excel(result_path / "accuracy_results.xlsx", index=False)
print("结果已保存")