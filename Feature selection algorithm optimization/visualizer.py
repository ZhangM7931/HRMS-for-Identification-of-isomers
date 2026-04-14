import matplotlib
matplotlib.use('Agg')  # 设置非GUI后端
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
try:
    import umap
except ImportError:
    umap = None

from pathlib import Path

# Set font for Chinese support if needed, though English labels are safer
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

class DataVisualizer:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_dim_reduction(self, X, y, method='PCA', title='Dim Reduction'):
        if method == 'PCA':
            reducer = PCA(n_components=2)
        elif method == 'LDA':
            # LDA components <= min(n_classes - 1, n_features)
            n_classes = len(np.unique(y))
            n_components = min(2, n_classes - 1)
            if n_components < 1: n_components = 1 # Fallback
            reducer = LinearDiscriminantAnalysis(n_components=n_components)
        elif method == 't-SNE':
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'UMAP':
            if umap is None:
                print("UMAP not installed. Skipping.")
                return
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        if method == 'LDA':
            X_r = reducer.fit_transform(X, y)
        else:
            X_r = reducer.fit_transform(X)
            
        # Handle 1D result from LDA if only 2 classes
        if X_r.shape[1] == 1:
            X_r = np.hstack((X_r, np.zeros_like(X_r)))
            
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=X_r[:, 0], y=X_r[:, 1], hue=y, palette='viridis', s=60, alpha=0.8)
        plt.title(title)
        plt.xlabel(f"{method} 1")
        plt.ylabel(f"{method} 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save
        filename = f"{title.replace(' ', '_').replace('-', '_')}.png"
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_pca(self, X, y, title="PCA"):
        self.plot_dim_reduction(X, y, method='PCA', title=title)

    def plot_lda(self, X, y, title="LDA"):
        self.plot_dim_reduction(X, y, method='LDA', title=title)

    def plot_tsne(self, X, y, title="t-SNE"):
        self.plot_dim_reduction(X, y, method='t-SNE', title=title)

    def plot_umap(self, X, y, title="UMAP"):
        self.plot_dim_reduction(X, y, method='UMAP', title=title)
