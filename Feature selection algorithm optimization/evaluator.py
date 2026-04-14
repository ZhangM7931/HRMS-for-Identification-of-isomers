import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

class FeatureEvaluator:
    def __init__(self, use_gpu=None, cp=None):
        # If use_gpu is None, auto-detect
        if use_gpu is None:
            try:
                import cupy as _cp  # noqa: F401
                import cuml  # noqa: F401
                use_gpu = True
                cp = _cp
            except Exception:
                use_gpu = False
                cp = None
        self.use_gpu = bool(use_gpu)
        self.cp = cp

    def calculate_redundancy(self, X):
        if X.shape[1] < 2:
            return 0.0
        corr_matrix = X.corr().abs()
        # Upper triangle only, excluding diagonal
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        mean_corr = upper.mean().mean()
        return 0.0 if np.isnan(mean_corr) else mean_corr

    def calculate_separation_metrics(self, X, y):
        """
        Calculate class separation metrics.
        """
        metrics = {}
        
        # 1. Silhouette Score (-1 to 1, higher is better)
        try:
            if len(np.unique(y)) > 1:
                metrics['silhouette'] = silhouette_score(X, y)
            else:
                metrics['silhouette'] = -1
        except:
            metrics['silhouette'] = -1
            
        # 2. Calinski-Harabasz Score (Variance Ratio Criterion) - higher is better
        try:
            if len(np.unique(y)) > 1:
                metrics['calinski_harabasz'] = calinski_harabasz_score(X, y)
            else:
                metrics['calinski_harabasz'] = 0
        except:
            metrics['calinski_harabasz'] = 0

        # 3. Adjusted Rand Index (ARI) - higher is better
        # Measures similarity between ground truth labels and a clustering (here we treat X structure as clustering proxy? 
        # Actually ARI compares two clusterings. If we don't have predicted clusters, we can't strictly use ARI against X directly 
        # unless we cluster X first. 
        # However, the user asked for "Rand Index" as a separation metric. 
        # Usually this implies running a clustering alg (like K-Means) on X and comparing to y.
        # Let's implement K-Means clustering on X then compare to y.
        try:
            from sklearn.cluster import KMeans
            n_classes = len(np.unique(y))
            kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
            y_pred_cluster = kmeans.fit_predict(X)
            metrics['adjusted_rand_index'] = adjusted_rand_score(y, y_pred_cluster)
        except:
            metrics['adjusted_rand_index'] = 0
            
        # 4. Intra/Inter Class Distance Ratio
        # Lower is better
        try:
            # Calculate centroids
            classes = np.unique(y)
            centroids = np.array([X[y == c].mean(axis=0) for c in classes])
            
            if len(classes) > 1:
                # Inter-class distance (mean distance between centroids)
                inter_dist = np.mean(pdist(centroids))
                
                # Intra-class distance (mean distance from samples to their centroid)
                intra_dists = []
                for i, c in enumerate(classes):
                    cluster_samples = X[y == c]
                    if len(cluster_samples) > 0:
                        dists = np.linalg.norm(cluster_samples - centroids[i], axis=1)
                        intra_dists.append(np.mean(dists))
                intra_dist = np.mean(intra_dists)
                
                metrics['intra_inter_ratio'] = intra_dist / inter_dist if inter_dist > 0 else np.inf
            else:
                metrics['intra_inter_ratio'] = np.inf
        except:
            metrics['intra_inter_ratio'] = np.inf
            
        return metrics

    def evaluate_feature_subset(self, X, y, cv=5):
        """
        Evaluate the selected feature subset.
        Returns a dictionary of metrics.
        使用 KNN 模型对选定的特征子集进行交叉验证，返回 cv_acc_baseline（基准准确率）
        """
        metrics = {}
        
        # 1. Dimension
        metrics['n_features'] = X.shape[1]
        
        # 2. Redundancy
        metrics['redundancy'] = self.calculate_redundancy(X)
        
        # 3. Separation Metrics (Standardized X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        sep_metrics = self.calculate_separation_metrics(X_scaled, y)
        metrics.update(sep_metrics)
            
        # 4. Simple Model Performance (KNN as baseline)
        # Prefer GPU KNN if available; fallback to CPU with parallel CV
        try:
            if self.use_gpu and self.cp is not None:
                from cuml.neighbors import KNeighborsClassifier as cuKNN
                skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
                cv_scores = []
                X_np = X_scaled
                for train_idx, test_idx in skf.split(X_np, y):
                    X_tr = self.cp.asarray(X_np[train_idx])
                    X_te = self.cp.asarray(X_np[test_idx])
                    y_tr = self.cp.asarray(y[train_idx])
                    y_te = y[test_idx]
                    knn_gpu = cuKNN(n_neighbors=3)
                    knn_gpu.fit(X_tr, y_tr)
                    preds = knn_gpu.predict(X_te).get()
                    cv_scores.append((preds == y_te).mean())
                metrics['cv_acc_baseline'] = float(np.mean(cv_scores))
            else:
                knn = KNeighborsClassifier(n_neighbors=3)
                scores = cross_val_score(knn, X_scaled, y, cv=cv, scoring='accuracy', n_jobs=-1)
                metrics['cv_acc_baseline'] = scores.mean()
        except Exception:
            metrics['cv_acc_baseline'] = 0.0
        
        return metrics
