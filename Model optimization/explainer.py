import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def run_shap_analysis(model, X_train, X_test, feature_names, output_dir, class_names=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure X_train and X_test are DataFrames with correct columns
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)
        
    # Determine explainer type
    model_type = type(model).__name__
    logger.info(f"Running SHAP for model type: {model_type}")
    
    try:
        explainer = None
        shap_values = None
        
        if model_type in ['RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
        elif model_type == 'LogisticRegression':
            # LinearExplainer is faster for LR
            # Masker is needed for recent SHAP versions
            masker = shap.maskers.Independent(data=X_train)
            explainer = shap.LinearExplainer(model, masker=masker)
            shap_values = explainer.shap_values(X_test)
            
        else:
            # KernelExplainer for SVM, ANN, KNN
            # Use a summary of background data (kmeans) to speed up
            # Limit background samples to 50 for speed
            if len(X_train) > 50:
                background = shap.kmeans(X_train, 50)
            else:
                background = X_train
                
            # Predict proba function
            if hasattr(model, 'predict_proba'):
                predict_fn = model.predict_proba
            else:
                predict_fn = model.predict
                
            explainer = shap.KernelExplainer(predict_fn, background)
            # Limit test samples for KernelExplainer as it is slow
            # Take up to 50 samples from X_test
            if len(X_test) > 50:
                X_test_shap = X_test.iloc[:50]
            else:
                X_test_shap = X_test
                
            shap_values = explainer.shap_values(X_test_shap)
            X_test = X_test_shap # Update X_test for plotting to match shap_values

        # Handle SHAP values format
        # shap_values can be list (multiclass) or array
        shap_values_for_plot = shap_values
        
        # 1. Summary Plot
        plt.figure()
        shap.summary_plot(shap_values_for_plot, X_test, feature_names=feature_names, class_names=class_names, show=False)
        plt.savefig(output_dir / "shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Bar Plot (Feature Importance)
        plt.figure()
        shap.summary_plot(shap_values_for_plot, X_test, feature_names=feature_names, class_names=class_names, plot_type="bar", show=False)
        plt.savefig(output_dir / "shap_importance_bar.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save Feature Importance to CSV
        # Calculate mean absolute SHAP value
        if isinstance(shap_values, list):
            # Sum over classes
            vals = np.sum([np.abs(sv) for sv in shap_values], axis=0)
        else:
            vals = np.abs(shap_values)
            
        if vals.ndim > 1:
            mean_shap = np.mean(vals, axis=0)
        else:
            mean_shap = vals
            
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv(output_dir / "shap_feature_importance.csv", index=False)
        
    except Exception as e:
        logger.error(f"SHAP analysis failed: {e}", exc_info=True)
