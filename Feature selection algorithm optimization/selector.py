import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非GUI后端
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
import yaml
import logging
import random
import optuna
from tqdm import tqdm
from features.evaluator import FeatureEvaluator

logger = logging.getLogger(__name__)

class FeatureSelector:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.selector_map = {
            'pls_da': self.select_features_pls_da,
            'mrmr': self.select_features_mrmr,
            'genetic': self.select_features_ga
        }
        self.gpu_available, self.cp = self._detect_gpu()
        self.use_gpu = self.gpu_available
        self.evaluator = FeatureEvaluator(use_gpu=self.use_gpu, cp=self.cp)

    def _detect_gpu(self):
        """Detect CuPy/cuML for optional GPU acceleration."""
        try:
            import cupy as cp
            import cuml  # noqa: F401
            logger.info("检测到 GPU 环境 (CuPy/cuML 可用)，将尽可能使用 GPU 加速。")
            return True, cp
        except Exception:
            logger.info("未检测到可用的 GPU (CuPy/cuML)，回退至 CPU。")
            return False, None

    def _load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
            
    def get_search_space(self, method_name):
        """Retrieve search space for a given method from config."""
        # Mapping logic
        config_key = None
        if 'pls' in method_name.lower(): config_key = 'PLS-DA_VIP'
        elif 'mrmr' in method_name.lower(): config_key = 'mRMR'
        elif 'genetic' in method_name.lower() or 'ga' in method_name.lower(): config_key = 'GA'
        
        if config_key and config_key in self.config['feature_selection']:
            return self.config['feature_selection'][config_key]['params']
        return {}

    def objective(self, trial, X, y, method_name, save_dir):
        search_space = self.get_search_space(method_name)
        params = {}
        
        # Dynamically suggest params based on config
        for param_name, config in search_space.items():
            values = config['values']
            if isinstance(values, list):
                # Check types to decide suggest method
                if all(isinstance(v, int) for v in values):
                     params[param_name] = trial.suggest_categorical(param_name, values)
                elif all(isinstance(v, (int, float)) for v in values):
                     params[param_name] = trial.suggest_categorical(param_name, values) # Use categorical for explicit discrete choices
                else:
                     params[param_name] = trial.suggest_categorical(param_name, values)
        
        # Add necessary fixed or derived params if needed
        # Run feature selection with suggested params
        try:
            selected_features = self.select_features(X, y, method_name, save_dir=save_dir, **params)
            
            # If no features selected, return 0
            if not selected_features:
                return 0.0
                
            # Evaluate using baseline KNN acc (from FeatureEvaluator)
            X_sel = X[selected_features]
            
            # FeatureEvaluator expects Encoded Y? 
            # In main.py evaluator.evaluate_feature_subset(X_sel, y_encoded) is called.
            # Here y is passed from optimize_and_select, need to ensure it's suitable.
            # Assuming y is already encoded or evaluator handles it.
            # evaluate_feature_subset calls cross_val_score with KNN.
            
            metrics = self.evaluator.evaluate_feature_subset(X_sel, y)
            return metrics.get('cv_acc_baseline', 0.0)
            
        except Exception as e:
            logger.warning(f"Optuna trial failed for {method_name} with params {params}: {e}")
            return 0.0

    def optimize_and_select(self, X, y, method, save_dir, n_trials=20, logger=None):
        """
        Run Optuna optimization for the feature selection method, then return best features.
        Also saves intermediate logs and results.
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        
        logger.info(f"开始 {method} 的参数优化...")
        
        # Setup Optuna study
        study_name = f"{method}_optimization"
        # Use TPESampler with a fixed seed for reproducibility
        sampler = optuna.samplers.TPESampler(seed=42)
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', study_name=study_name, sampler=sampler)
        
        # Optimization loop
        with tqdm(total=n_trials, desc=f"Optimizing {method}") as pbar:
            def callback(study, trial):
                logger.info(f"Trial {trial.number + 1}/{n_trials} completed: value={trial.value:.4f}")
                pbar.update(1)
            
            study.optimize(lambda trial: self.objective(trial, X, y, method, save_dir), n_trials=n_trials, callbacks=[callback])
        
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"{method} 最优参数: {best_params}, Best CV Acc: {best_value:.4f}")
        
        # Save optimization history (trials)
        trials_df = study.trials_dataframe()
        trials_file = save_dir / "optimization_trials.csv"
        trials_df.to_csv(trials_file, index=False)
        logger.info(f"优化过程已保存至 {trials_file}")
        
        # Save best params to CSV
        best_params_df = pd.DataFrame([best_params])
        best_params_df['best_score'] = best_value
        best_params_file = save_dir / "optimal_params.csv"
        best_params_df.to_csv(best_params_file, index=False)
        logger.info(f"最优参数已保存至 {best_params_file}")
        
        # Run final selection with best params
        logger.info("使用最优参数进行最终特征选择...")
        final_features = self.select_features(X, y, method, save_dir=save_dir, **best_params)

        # Persist selected features for downstream modeling
        feature_file = Path(save_dir) / "selected_features.txt"
        try:
            with open(feature_file, "w", encoding='utf-8') as f:
                f.write("\n".join(final_features))
            logger.info(f"最终特征列表已保存至 {feature_file}")
        except Exception as e:
            logger.error(f"保存最终特征列表失败: {e}")
        
        return final_features

    def select_features(self, X, y, method, **kwargs):
        method_key = method.lower()
        # Map common names to keys
        if 'pls' in method_key: method_key = 'pls_da'
        if 'mrmr' in method_key: method_key = 'mrmr'
        if 'genetic' in method_key or 'ga' in method_key: method_key = 'genetic'

        if method_key not in self.selector_map:
            raise ValueError(f"未知的特征选择方法: {method}")
        
        logger.info(f"正在使用 {method} 选择特征...")
        return self.selector_map[method_key](X, y, **kwargs)

    # --- PLS-DA VIP ---
    def get_pls_vip(self, X, y, n_components=3):
        try:
            pls = PLSRegression(n_components=n_components, scale=True)
            # PLS requires numeric y. Convert labels to dummies or codes.
            if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
                y_codes = pd.Categorical(y).codes
            else:
                y_codes = y
                
            pls.fit(X, y_codes)
            
            t = pls.x_scores_
            w = pls.x_weights_
            q = pls.y_loadings_
            
            p, h = w.shape
            vips = np.zeros((p,))
            
            s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
            total_s = np.sum(s)
            
            for i in range(p):
                weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
                vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
                
            return vips
        except Exception as e:
            logger.error(f"PLS-DA VIP 计算失败: {e}")
            # Fallback: return all zeros or handle gracefully
            return np.zeros(X.shape[1])

    def select_features_pls_da(self, X, y, **kwargs):
        # Default fallback
        defaults = self.config['feature_selection']['PLS-DA_VIP']['default_params']
        
        # Use kwargs if provided (from optimize), else default
        n_components = kwargs.get('n_components', int(defaults.get('n_components', 3)))
        vip_threshold = kwargs.get('vip_threshold', float(defaults.get('VIP_threshold', 1.0))) # Check casing in config
        
        # In config it is VIP_threshold, but kwargs passed from optuna suggest_categorical usually matches param name from config 'params'
        # The params config keys: n_components, scale, VIP_threshold
        # So kwargs will have 'VIP_threshold' key.
        if 'VIP_threshold' in kwargs:
             vip_threshold = kwargs['VIP_threshold']
        
        vips = self.get_pls_vip(X, y, n_components)
        selected_indices = np.where(vips > vip_threshold)[0]
        selected_features = X.columns[selected_indices].tolist()
        
        # 特征选择完成，不在日志中输出具体特征
        return selected_features

    # --- mRMR ---
    def select_features_mrmr(self, X, y, **kwargs):
        params = self.config['feature_selection']['mRMR']['default_params']
        # Allow kwargs to override config
        n_features_to_select = kwargs.get('n_features', int(params.get('n_features', 20)))
        
        if n_features_to_select > X.shape[1]:
            n_features_to_select = X.shape[1]

        # 1. Calculate Relevance (MI between each feature and target)
        # Ensure y is discrete for classification
        relevance = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
        
        # 2. Iterative Selection
        selected_indices = []
        candidates = list(range(X.shape[1]))
        
        # First feature: max relevance
        first_idx = candidates[np.argmax(relevance[candidates])]
        selected_indices.append(first_idx)
        candidates.remove(first_idx)
        
        # Subsequent features
        while len(selected_indices) < n_features_to_select:
            best_score = -np.inf
            best_idx = -1
            
            # Get the sub-matrix of selected features
            X_selected = X.iloc[:, selected_indices].values
            
            for idx in candidates:
                # Relevance
                rel = relevance[idx]
                
                # Redundancy: Mean absolute correlation with selected features
                current_feat = X.iloc[:, idx].values
                
                corrs = []
                for i in range(len(selected_indices)):
                    feat_sel = X_selected[:, i]
                    # Check for constant features to avoid RuntimeWarning or NaN
                    if np.std(current_feat) == 0 or np.std(feat_sel) == 0:
                        c = 0.0
                    else:
                        c = np.corrcoef(current_feat, feat_sel)[0, 1]
                    corrs.append(c)
                
                corrs = np.abs(corrs)
                # Ensure no NaNs propagate
                corrs = np.nan_to_num(corrs)
                red = np.mean(corrs)
                
                # mRMR score (MID: Rel - Red)
                score = rel - red
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            # Safety: if no improvement, break to avoid infinite loop
            if best_idx == -1:
                logger.warning("mRMR 未找到更多可选特征，提前停止。")
                break

            selected_indices.append(best_idx)
            candidates.remove(best_idx)
            
        selected_features = X.columns[selected_indices].tolist()
        logger.info(f"mRMR 选择了 {len(selected_features)} 个特征。")
        return selected_features

    # --- Genetic Algorithm ---
    def select_features_ga(self, X, y, **kwargs):
        save_dir = kwargs.get('save_dir')
        params = self.config['feature_selection']['GA']['default_params']
        
        # Allow kwargs to override config
        n_population = kwargs.get('population_size', int(params.get('population_size', 50)))
        n_generations = kwargs.get('generations', int(params.get('generations', 20)))
        crossover_prob = kwargs.get('crossover_prob', float(params.get('crossover_prob', 0.8)))
        mutation_prob = kwargs.get('mutation_prob', float(params.get('mutation_prob', 0.1)))
        
        n_features = X.shape[1]
        
        # Initialize population (binary vectors)
        population = [np.random.randint(0, 2, n_features) for _ in range(n_population)]
        
        # Estimator for fitness (fast one, e.g., LR)
        estimator = LogisticRegression(solver='liblinear', penalty='l1', C=0.1, max_iter=1000)
        use_gpu = getattr(self, 'use_gpu', False) and self.cp is not None
        cu_lr = None
        if use_gpu:
            try:
                from cuml.linear_model import LogisticRegression as cuLogReg
                cu_lr = cuLogReg(penalty='l1', C=0.1, max_iter=1000, fit_intercept=True)
            except Exception as e:
                logger.warning(f"GPU LogisticRegression 初始化失败，回退 CPU: {e}")
                use_gpu = False
        
        best_individual = None
        best_fitness = -1.0
        best_fitness_history = []
        
        for gen in range(n_generations):
            fitness_scores = []
            for ind in population:
                cols_idx = np.where(ind == 1)[0]
                if len(cols_idx) == 0:
                    fitness_scores.append(0.0)
                    continue
                
                # Evaluate
                X_subset = X.iloc[:, cols_idx]
                if use_gpu and cu_lr is not None:
                    try:
                        # manual CV on GPU
                        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                        cv_scores = []
                        X_np = X_subset.values
                        for train_idx, test_idx in skf.split(X_np, y):
                            X_tr = self.cp.asarray(X_np[train_idx])
                            X_te = self.cp.asarray(X_np[test_idx])
                            y_tr = self.cp.asarray(y[train_idx])
                            y_te = y[test_idx]
                            cu_lr.fit(X_tr, y_tr)
                            preds = cu_lr.predict(X_te).get()
                            cv_scores.append((preds == y_te).mean())
                        score = float(np.mean(cv_scores))
                    except Exception as e:
                        logger.warning(f"GPU GA 评估失败，回退 CPU: {e}")
                        use_gpu = False
                        score = np.mean(cross_val_score(estimator, X_subset, y, cv=3, scoring='accuracy', n_jobs=-1))
                else:
                    # 3-fold CV for speed (CPU, parallelized)
                    score = np.mean(cross_val_score(estimator, X_subset, y, cv=3, scoring='accuracy', n_jobs=-1))
                fitness_scores.append(score)
                
                if score > best_fitness:
                    best_fitness = score
                    best_individual = ind.copy()
            
            best_fitness_history.append(best_fitness)
            
            # Selection (Tournament)
            new_population = []
            for _ in range(n_population):
                # Select 2 random
                candidates_idx = np.random.choice(len(population), 2, replace=False)
                if fitness_scores[candidates_idx[0]] > fitness_scores[candidates_idx[1]]:
                    winner = population[candidates_idx[0]]
                else:
                    winner = population[candidates_idx[1]]
                new_population.append(winner.copy())
            
            # Crossover
            for i in range(0, n_population, 2):
                if i+1 < n_population and np.random.rand() < crossover_prob:
                    p1 = new_population[i]
                    p2 = new_population[i+1]
                    pt = np.random.randint(1, n_features)
                    # Swap
                    c1 = np.concatenate((p1[:pt], p2[pt:]))
                    c2 = np.concatenate((p2[:pt], p1[pt:]))
                    new_population[i] = c1
                    new_population[i+1] = c2
            
            # Mutation
            for i in range(n_population):
                if np.random.rand() < mutation_prob:
                    # Flip one bit
                    m_pt = np.random.randint(0, n_features)
                    new_population[i][m_pt] = 1 - new_population[i][m_pt]
            
            population = new_population
            logger.info(f"GA 第 {gen+1}/{n_generations} 代 - 最佳适应度: {best_fitness:.4f}")

        # Plot Fitness History
        if save_dir:
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, n_generations + 1), best_fitness_history, marker='o')
                plt.title('遗传算法 - 最佳适应度随代数变化')
                plt.xlabel('代数')
                plt.ylabel('最佳适应度 (CV 准确率)')
                plt.grid(True)
                plt.savefig(Path(save_dir) / "ga_fitness_history.png")
                plt.close()
                
                # Save fitness history to CSV
                history_df = pd.DataFrame({
                    'generation': range(1, n_generations + 1),
                    'best_fitness': best_fitness_history
                })
                history_file = Path(save_dir) / "ga_fitness_history.csv"
                history_df.to_csv(history_file, index=False)
                logger.info(f"GA 适应度历史已保存至 {history_file}")
            except Exception as e:
                logger.error(f"绘制/保存 GA 收敛图失败: {e}")

        if best_individual is None:
            # Fallback
            return X.columns.tolist()

        selected_indices = np.where(best_individual == 1)[0]
        selected_features = X.columns[selected_indices].tolist()
        logger.info(f"GA 选择了 {len(selected_features)} 个特征。")
        return selected_features

