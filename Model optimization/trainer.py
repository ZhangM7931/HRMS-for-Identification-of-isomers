import pandas as pd
import numpy as np
import optuna
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import yaml
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path, config_key='machine_learning'):
        self.config = self._load_config(config_path, config_key)
        self.model_map = {
            'lr': 'LR-ElasticNet',
            'svm': 'SVM',
            'ann': 'ANN',
            'knn': 'KNN',
            'xgboost': 'XGBoost'
        }

    def _load_config(self, config_path, config_key):
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
        return full_config[config_key]

    def get_model_class(self, model_name):
        # Map input name to config name
        config_name = self.model_map.get(model_name.lower(), model_name)
        
        if config_name == 'LR-ElasticNet':
            return LogisticRegression
        elif config_name == 'SVM':
            return SVC
        elif config_name == 'ANN':
            return MLPClassifier
        elif config_name == 'KNN':
            return KNeighborsClassifier
        elif config_name == 'XGBoost':
            return XGBClassifier
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def objective(self, trial, X, y, model_name, cv_folds):
        # Convert X to numpy if it's a DataFrame to avoid pandas overhead in sklearn checks
        if isinstance(X, pd.DataFrame):
            X = X.values

        config_name = self.model_map.get(model_name.lower(), model_name)
        model_config = self.config[config_name]
        params = {}
        
        # Dynamically suggest params
        for param_name, param_config in model_config['params'].items():
            if 'values' in param_config:
                params[param_name] = trial.suggest_categorical(param_name, param_config['values'])
            elif 'low' in param_config:
                params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=param_config.get('log', False))
        
        # Add fixed params
        if config_name == 'LR-ElasticNet':
            params.update({'penalty': 'elasticnet', 'solver': 'saga', 'max_iter': 5000, 'random_state': 42})
            model = LogisticRegression(**params)
        elif config_name == 'SVM':
            params.update({'probability': True, 'random_state': 42})
            model = SVC(**params)
        elif config_name == 'ANN':
            params.update({'max_iter': 1000, 'random_state': 42, 'early_stopping': True})
            model = MLPClassifier(**params)
        elif config_name == 'KNN':
            model = KNeighborsClassifier(**params)
        elif config_name == 'XGBoost':
            params.update({'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'logloss'})
            model = XGBClassifier(**params)
        else:
            return 0.0

        # Cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        try:
            scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
            return scores.mean()
        except Exception as e:
            logger.error(f"Trial failed for {config_name}: {e}")
            return 0.0

    def tune_hyperparameters(self, X, y, model_name, n_trials=20, cv_folds=5, output_dir=None):
        # Suppress Optuna logging to stdout
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Prepare logging for parameter optimization
        param_log_path = None
        param_csv_path = None
        if output_dir:
            param_log_path = output_dir / "param_opt_detailed.log"
            param_csv_path = output_dir / "param_opt_trials.csv"
        
        study = optuna.create_study(direction='maximize')
        
        # Get parameter ranges for logging
        config_name = self.model_map.get(model_name.lower(), model_name)
        model_config = self.config[config_name]
        param_ranges = {}
        for param_name, param_config in model_config['params'].items():
            if 'values' in param_config:
                param_ranges[param_name] = f"categorical: {param_config['values']}"
            elif 'low' in param_config:
                param_ranges[param_name] = f"float: [{param_config['low']}, {param_config['high']}]" + (f", log={param_config.get('log', False)}" if 'log' in param_config else "")
        
        # Write header to log file
        if param_log_path:
            with open(param_log_path, 'w', encoding='utf-8') as f:
                f.write(f"Parameter Optimization for {config_name}\n")
                f.write("=" * 50 + "\n")
                f.write("Parameter ranges:\n")
                for param_name, param_range in param_ranges.items():
                    f.write(f"  {param_name}: {param_range}\n")
                f.write("\nTrial Results:\n")
                f.write("-" * 80 + "\n")
        
        # Initialize CSV data
        csv_data = []
        
        def objective_with_logging(trial):
            # Convert X to numpy if it's a DataFrame to avoid pandas overhead in sklearn checks
            if isinstance(X, pd.DataFrame):
                X_local = X.values
            else:
                X_local = X
            
            X_local = np.asarray(X_local)
            y_local = np.asarray(y).ravel()
            
            config_name_local = self.model_map.get(model_name.lower(), model_name)
            model_config_local = self.config[config_name_local]
            params = {}
            
            # Dynamically suggest params
            for param_name, param_config in model_config_local['params'].items():
                if 'values' in param_config:
                    params[param_name] = trial.suggest_categorical(param_name, param_config['values'])
                elif 'low' in param_config:
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=param_config.get('log', False))
            
            # Add fixed params
            if config_name_local == 'LR-ElasticNet':
                params.update({'penalty': 'elasticnet', 'solver': 'saga', 'max_iter': 5000, 'random_state': 42})
                model = LogisticRegression(**params)
            elif config_name_local == 'SVM':
                params.update({'probability': True, 'random_state': 42})
                model = SVC(**params)
            elif config_name_local == 'ANN':
                params.update({'max_iter': 1000, 'random_state': 42, 'early_stopping': True})
                model = MLPClassifier(**params)
            elif config_name_local == 'KNN':
                model = KNeighborsClassifier(**params)
            elif config_name_local == 'XGBoost':
                params.update({'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'logloss'})
                model = XGBClassifier(**params)
            else:
                return 0.0

            # Cross-validation
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            try:
                scores = cross_val_score(model, X_local, y_local, cv=skf, scoring='accuracy', n_jobs=1)
                accuracy = scores.mean()
                
                # Log this trial
                if param_log_path:
                    with open(param_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"Trial {trial.number}: Params={params}, Accuracy={accuracy:.4f}\n")
                
                # Add to CSV data
                csv_row = {'trial': trial.number, 'accuracy': accuracy}
                csv_row.update(params)
                csv_data.append(csv_row)
                
                return accuracy
            except Exception as e:
                logger.error(f"Trial {trial.number} failed for {config_name_local}: {e}")
                # Log failed trial
                if param_log_path:
                    with open(param_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"Trial {trial.number}: Params={params}, Failed: {e}\n")
                
                # Add failed trial to CSV
                csv_row = {'trial': trial.number, 'accuracy': 0.0}
                csv_row.update(params)
                csv_data.append(csv_row)
                
                return 0.0
        
        study.optimize(objective_with_logging, n_trials=n_trials)
        
        # Write best result to log
        if param_log_path:
            with open(param_log_path, 'a', encoding='utf-8') as f:
                f.write("\n" + "=" * 50 + "\n")
                f.write("OPTIMIZATION RESULTS:\n")
                f.write(f"Best Trial: {study.best_trial.number}\n")
                f.write(f"Best Accuracy: {study.best_value:.4f}\n")
                f.write(f"Best Parameters: {study.best_params}\n")
        
        # Save CSV
        if param_csv_path and csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(param_csv_path, index=False)
        
        logger.info(f"{model_name} 最佳参数: {study.best_params}")
        return study.best_params

    def get_default_params(self, model_name):
        config_name = self.model_map.get(model_name.lower(), model_name)
        if config_name not in self.config:
            logger.warning(f"No config found for {config_name}, using empty params.")
            return {}
        return self.config[config_name].get('default_params', {}).copy()

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, model_name, best_params, output_dir):
        # Convert to numpy arrays to ensure compatibility with sklearn
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)
        
        # Debug logging
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        if X_train.shape[0] != len(y_train):
            raise ValueError(f"Training data length mismatch: X_train {X_train.shape[0]} != y_train {len(y_train)}")
        if X_test.shape[0] != len(y_test):
            raise ValueError(f"Test data length mismatch: X_test {X_test.shape[0]} != y_test {len(y_test)}")
        
        config_name = self.model_map.get(model_name.lower(), model_name)
        model_class = self.get_model_class(model_name)
        
        # Prepare params
        final_params = best_params.copy()
        
        # Add common fixed params if not present (ensure reproducibility)
        if 'random_state' not in final_params and config_name != 'KNN':
            final_params['random_state'] = 42
            
        # Initialize and train
        try:
            model = model_class(**final_params)
            model.fit(X_train, y_train)
        except TypeError as e:
            logger.error(f"Error initializing/fitting {config_name} with params {final_params}: {e}")
            raise e
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        
        # Handle multiclass AUC
        if y_prob is not None:
            if len(np.unique(y_test)) > 2:
                auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
            else:
                auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = 0.0  # No AUC available
            
        # Save classification report
        report = classification_report(y_test, y_pred)
        report_path = output_dir / "model_metrics.log"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {config_name}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"AUC: {auc:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(report)
            
        logger.info(f"已保存 {config_name} 的评估报告至 {report_path}")
        
        return {
            'model_name': config_name,
            'accuracy': acc,
            'auc': auc,
            'model': model,
            'y_test': y_test,
            'y_prob': y_prob
        }

    def train_final_model(self, X_train, y_train, model_name, best_params):
        config_name = self.model_map.get(model_name.lower(), model_name)
        
        if config_name == 'LR-ElasticNet':
            model = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=5000, random_state=42, **best_params)
        elif config_name == 'SVM':
            model = SVC(probability=True, random_state=42, **best_params)
        elif config_name == 'ANN':
            model = MLPClassifier(max_iter=1000, random_state=42, early_stopping=True, **best_params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test, label_encoder=None):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'precision_macro': precision_score(y_test, y_pred, average='macro')
        }
        
        # AUC (handle multiclass)
        if y_prob is not None:
            try:
                if len(np.unique(y_test)) == 2:
                    metrics['auc'] = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    metrics['auc'] = roc_auc_score(y_test, y_prob, multi_class='ovr')
            except Exception as e:
                logger.warning(f"AUC calculation failed: {e}")
                metrics['auc'] = 0.0
                
        return metrics, y_pred, y_prob
