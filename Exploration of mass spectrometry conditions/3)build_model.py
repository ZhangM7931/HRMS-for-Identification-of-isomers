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

for data_name in [
        "咪酯类", 
        "尼秦类", 
        "芬太尼类", 
        "苯丙胺类", 
        "美托咪酯"
        ]:
    data_path = result_path / data_name
    if not data_path.exists():
        data_path.mkdir(parents=True)
    
    data = pd.read_excel(processed_path / f"{data_name}.xlsx", index_col=0).iloc[:, 1:]
    X = data.iloc[:, 1:]
    label = data.iloc[:, 0]
    label = label.apply(lambda x: x.lower())
    if data_name == "尼秦类":
        label[label=="butonitazne"] = "butonitazene"
        label[label=="sec-butonitazne"] = "sec-butonitazene"
    elif data_name == "芬太尼类":
        label[label=="丁酰芬太尼"] = "butyrylfentanyl"
        label[label=="异丁酰芬太尼"] = "isobutyrylfentanyl"
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(label)
    X["methods"] = LabelEncoder().fit_transform(X["methods"])
    X = X.fillna(0)
    X.columns = X.columns.astype(str)
    X = X.astype(float)
    X[:] = StandardScaler().fit_transform(X.astype(float))
    
    importance = shap_importance(algo, X, y)
    
    X_train, X_test, y_train,  y_test = train_test_split(X.loc[:, importance.index[:10]], y, test_size=0.3, random_state=42)
    
    loocv_scatter = val_loocv(algo, X_train, y_train)
    loocv_mcls = mcls(*loocv_scatter)
    
    test_scatter = val_test(algo, X_train, y_train, X_test, y_test)
    test_mcls = mcls(*test_scatter)
    
    with open(data_path / "建模结果.txt", "a") as f:
        f.write(f"数据名称: {data_name}\n")
        f.write(f"标签编码: {label_encoder.transform(label_encoder.classes_)}, 对应标签: {label_encoder.classes_}\n")
        f.write(f"建模算法: {algo.__name__}\n")
        f.write(f"LOOCV结果: \n")
        f.write(f"    准确率: {round(loocv_mcls['accuracy'], 4)}\n")
        f.write(f"    精确率: {round(loocv_mcls['precision'], 4)}\n")
        f.write(f"    F1值: {round(loocv_mcls['f1'], 4)}\n")
        f.write(f"    召回率: {round(loocv_mcls['recall'], 4)}\n")
        f.write(f"    0类准确率: {round(loocv_mcls['accuracy_0'], 4)}\n")
        f.write(f"    1类准确率: {round(loocv_mcls['accuracy_1'], 4)}\n")
        f.write(f"    混淆矩阵: \n")
        f.write(f"        {loocv_mcls['cm']}\n")
        f.write(f"测试结果: \n")
        f.write(f"    准确率: {round(test_mcls['accuracy'], 4)}\n")
        f.write(f"    精确率: {round(test_mcls['precision'], 4)}\n")
        f.write(f"    F1值: {round(test_mcls['f1'], 4)}\n")
        f.write(f"    召回率: {round(test_mcls['recall'], 4)}\n")
        f.write(f"    0类准确率: {round(test_mcls['accuracy_0'], 4)}\n")
        f.write(f"    1类准确率: {round(test_mcls['accuracy_1'], 4)}\n")
        f.write(f"    混淆矩阵: \n")
        f.write(f"        {test_mcls['cm']}\n")
    
    explainer = shap.TreeExplainer(algo().fit(X, y))
    svalues = explainer(X)
    fig, ax = plt.subplots()
    shap.plots.bar(svalues, ax=ax, show=False)
    ax.set_title(data_name)
    plt.savefig(data_path / "重要性.png", bbox_inches="tight")
    plt.cla()
    shap.plots.scatter(svalues[:, 0], show=False)
    plt.gca().set_title(f"实验条件对{data_name}的影响")
    plt.savefig(data_path / f"实验条件对{data_name}的影响", bbox_inches="tight")
    plt.cla()
    shap.plots.beeswarm(svalues, show=False)
    plt.gca().set_title(data_name)
    plt.savefig(data_path / "蜂群图.png", bbox_inches="tight")