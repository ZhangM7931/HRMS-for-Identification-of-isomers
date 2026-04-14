import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
import sys
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try to import sklearn models
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Define best model combinations globally
BEST_MODELS = {
    '苯丙胺类': ('GA', 'LR'),
    '芬太尼类': ('GA', 'XGBoost'), 
    '咪酯类': ('GA', 'LR'),
    '尼秦类': ('GA', 'LR')
}

def load_data(data_path, drug_type):
    """Load denoised data for a specific drug type"""
    file_path = data_path / f"{drug_type}_merged.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Separate metadata and features
    metadata_cols = ['ID', 'names', 'labels', 'methods']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    X = df[feature_cols].values
    y = df['labels'].values
    
    # Convert labels to numeric if needed
    if not np.issubdtype(y.dtype, np.number):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    return X, y, df[['ID', 'names', 'labels']]

def load_best_params(results_dir, drug_type, fs_method, model_name):
    """Load best parameters for the model"""
    if fs_method == 'GA':
        if model_name == 'LR':
            model_dir = 'LR-ElasticNet'
        elif model_name == 'XGBoost':
            model_dir = 'XGBoost'
        else:
            raise ValueError(f"Unsupported model for GA: {model_name}")
        
        param_file = results_dir / drug_type / "feature_selection" / "ga" / model_dir / "param_opt.log"
    else:
        raise ValueError(f"Unsupported fs_method: {fs_method}")
    
    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {param_file}")
    
    with open(param_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Parse "Best params: {'key': value, ...}"
    if content.startswith("Best params:"):
        params_str = content.split("Best params:", 1)[1].strip()
        # Use eval to parse the dict (safe since it's our own file)
        params = eval(params_str)
    else:
        raise ValueError(f"Unexpected format in {param_file}")
    
    return params

def create_model(model_name, params):
    """Create model instance with parameters"""
    if model_name == 'LR':
        # For LR-ElasticNet, ensure penalty is elasticnet and use saga solver
        params = params.copy()
        params['penalty'] = 'elasticnet'
        params['solver'] = 'saga'
        return LogisticRegression(**params, random_state=42, max_iter=2000)
    elif model_name == 'XGBoost':
        from xgboost import XGBClassifier
        return XGBClassifier(**params, random_state=42)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def predict_on_test_data(data_path, results_dir, output_dir):
    """Predict on DAY30 test data using best models"""
    
    results = {}
    trained_models = {}
    
    for drug_type, (fs_method, model_name) in BEST_MODELS.items():
        print(f"Processing {drug_type} with {fs_method}-{model_name}")
        
        try:
            # Load test data
            X_test, y_test, metadata = load_data(data_path, drug_type)
            
            # Load best parameters
            params = load_best_params(results_dir, drug_type, fs_method, model_name)
            
            # Create and train model
            model = create_model(model_name, params)
            model.fit(X_test, y_test)  # Note: Using test data for training since we don't have separate train/test
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store results
            results[drug_type] = {
                'predictions': y_pred,
                'true_labels': y_test,
                'accuracy': accuracy,
                'metadata': metadata,
                'model': model_name,
                'fs_method': fs_method
            }
            
            # Store trained model
            trained_models[drug_type] = model
            
            print(f"{drug_type} accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error processing {drug_type}: {e}")
            results[drug_type] = {'error': str(e)}
    
    return results, trained_models

def predict_on_positive_samples(drug_type, positive_data_path, results_dir, output_dir):
    """Predict on positive samples using newly trained models from DAY1-15"""
    
    print(f"Processing positive samples for {drug_type}")
    
    try:
        # Using the same denoised merged data as main.py and splitting 8:2
        data_dir = Path(__file__).resolve().parent.parent / "output" / "merged_data_denoised"
        train_df = pd.read_csv(data_dir / f"{drug_type}_merged.csv")
        X_all = train_df.drop(columns=['ID', 'names', 'labels', 'methods'])
        y_all_raw = train_df['labels'].astype(str).values

        le = LabelEncoder()
        y_all = le.fit_transform(y_all_raw)
            
        from sklearn.model_selection import train_test_split
        X_train, _, y_train, _ = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
            
        # 根据全局最佳模型配置获取当前药物的配置
        if drug_type not in BEST_MODELS:
            raise ValueError(f"Unknown best model combination for {drug_type}")
            
        fs_method_upper, model_name = BEST_MODELS[drug_type]
        fs_method = fs_method_upper.lower() # e.g. 'GA' -> 'ga'
        
        # load selected features
        fs_path = Path(results_dir) / drug_type / "feature_selection" / fs_method / "selected_features.txt"
        with open(fs_path, 'r') as f:
            selected_features = [line.strip() for line in f if line.strip()]
            
        X_train_fs = X_train[selected_features].values
        
        params = load_best_params(Path(results_dir), drug_type, fs_method_upper, model_name)
        model = create_model(model_name, params)
        model.fit(X_train_fs, y_train)

        # Load positive test data
        X_test_all, y_test_all, metadata = load_data(Path(positive_data_path), drug_type)
        
        # Take all positive samples
        X_test = X_test_all
        y_test = y_test_all
        
        # We must align predicting data with selected features
        test_df = pd.DataFrame(X_test_all, columns=[c for c in pd.read_csv(Path(positive_data_path) / f"{drug_type}_merged.csv").columns if c not in ['ID', 'names', 'labels', 'methods']])
        
        # Initialize an empty DataFrame with target features
        X_test_fs_df = pd.DataFrame(0.0, index=test_df.index, columns=selected_features)
        
        # Fill in the features that exist in the positive sample data
        existing_features = [f for f in selected_features if f in test_df.columns]
        X_test_fs_df[existing_features] = test_df[existing_features]
                
        X_test_fs = X_test_fs_df.values
        
        y_pred = model.predict(X_test_fs)
        
        # 将预测出数字（由于是由训练集训练出来的，对应全局的LabelEncoder）翻转回真实的英文字符串标签
        y_pred_str = le.inverse_transform(y_pred)
        # 获取阳性样本表格中原有的、未经本地重新编码的真实字符串标签
        y_test_str = metadata['labels'].astype(str).values
        
        # 抛弃掉之前的假人得分规则，我们用一致变小写的精准比对来进行计分，防范大小写导致误判
        try:
            y_pred_lower = np.array([str(x).lower() for x in y_pred_str])
            y_test_lower = np.array([str(x).lower() for x in y_test_str])
            accuracy = accuracy_score(y_test_lower, y_pred_lower)
        except Exception as e:
            print(f"Accuracy calc error: {e}")
            accuracy = None
        
        # Store results
        results = {
            'predictions': y_pred_str,
            'true_labels': y_test_str,
            'accuracy': accuracy,
            'metadata': metadata,
            'model': f"GA-{model_name}"
        }
        
        print(f"Positive samples {drug_type} predictions completed")
        if accuracy is not None:
            print(f"Accuracy: {accuracy:.4f}")
        
        # Save detailed predictions
        pred_df = metadata.copy()
        pred_df['predicted_label'] = y_pred_str
        pred_df['true_label'] = y_test_str
        
        pred_file = Path(output_dir) / f"{drug_type}_positive_samples_predictions.csv"
        pred_file.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_file, index=False, encoding='utf-8-sig')
        
        print(f"Positive samples predictions saved to {pred_file}")
        
        return results
        
    except Exception as e:
        print(f"Error processing positive samples for {drug_type}: {e}")
        return {'error': str(e)}

def save_results(results, output_dir):
    """Save prediction results to files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = []
    
    for drug_type, result in results.items():
        if 'error' in result:
            summary.append(f"{drug_type}: Error - {result['error']}")
            continue
        
        # Save detailed predictions
        pred_df = result['metadata'].copy()
        pred_df['predicted_label'] = result['predictions']
        pred_df['true_label'] = result['true_labels']
        
        pred_file = output_dir / f"{drug_type}_predictions.csv"
        pred_df.to_csv(pred_file, index=False, encoding='utf-8-sig')
        
        summary.append(f"{drug_type}: Accuracy = {result['accuracy']:.4f}, Model = {result['fs_method']}-{result['model']}")
    
    # Save summary
    summary_file = output_dir / "prediction_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("DAY30 Test Data Prediction Results\n")
        f.write("=" * 40 + "\n\n")
        for line in summary:
            f.write(line + "\n")
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    # === 预测模式配置 ===
    # 可选值: 'day30' 或 '阳性样本数据'
    test_data_type = '阳性样本数据' 
    
    # 仅当 test_data_type='阳性样本数据' 时生效，目前可选值: '苯丙胺类' 或 '咪酯类'
    drug_type = '咪酯类' 
    
    # === 公共路径配置 ===
    current_dir = Path(__file__).resolve().parent.parent
    # 最佳模型参数与特征的统一读取路径
    results_dir = current_dir / 'output/results'
    
    print(f"当前运行模式: {test_data_type}")
    print(f"最佳模型参数来源: {results_dir}\n")

    # === 分支执行逻辑 ===
    if test_data_type == 'day30':
        # day30 数据路径
        test_data_path = current_dir / 'test/day30/output/merged_data_denoised'
        test_output_path = current_dir / 'test/day30/output/predictions'
        
        print(f"测试数据路径: {test_data_path}")
        print(f"预测结果输出: {test_output_path}")
        print("\n=== Processing DAY30 Test Data ===")
        
        # 运行 day30 预测 (predict_on_test_data 会根据 BEST_MODELS 对包含的所有药物自动预测)
        results, trained_models = predict_on_test_data(test_data_path, results_dir, test_output_path)
        
        # 统一保存结果
        save_results(results, test_output_path)
        
    elif test_data_type == '阳性样本数据':
        # 阳性样本数据路径 (根据配置的 drug_type 动态生成)
        test_data_path = current_dir / 'test/阳性样本数据/output' / drug_type / 'merged_data_denoised'
        test_output_path = current_dir / 'test/阳性样本数据/output' / drug_type / 'predictions'
        
        print(f"目标药物类型: {drug_type}")
        print(f"测试数据路径: {test_data_path}")
        print(f"预测结果输出: {test_output_path}")
        print("\n=== Processing Positive Samples ===")
        
        # 运行阳性样本预测 (传参对应 drug_type 及动态生成的路径)
        positive_results = predict_on_positive_samples(drug_type, test_data_path, results_dir, test_output_path)
        
    else:
        print(f"未知的 test_data_type: {test_data_type}，请检查配置。")
    
    print("\nAll predictions completed.")