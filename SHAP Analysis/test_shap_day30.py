import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings('ignore')

plt.tight_layout = lambda *args, **kwargs: None

plt.rcParams['font.family'] = 'Times New Roman'
pd.options.display.max_columns = None
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['svg.fonttype'] = 'none'  # 保证保存为svg时，字体不转为路径，方便在外部软件中二次编辑

# BEST MODELS
BEST_MODELS = {
    '苯丙胺类': ('GA', 'LR'),
    '芬太尼类': ('GA', 'XGBoost'), 
    '咪酯类': ('GA', 'LR'),
    '尼秦类': ('GA', 'LR')
}

COLOR_SCHEMES = {
    'viridis': plt.cm.viridis,
    'plasma': plt.cm.plasma,
    'coolwarm': plt.cm.coolwarm,
    'RdYlBu': plt.cm.RdYlBu,
    'RdBu_r': plt.cm.RdBu_r
}
selected_color_scheme = 'viridis'
cmap_base = COLOR_SCHEMES[selected_color_scheme]

def create_optimized_cmap(base_cmap, start=0.2, end=0.9):
    colors = base_cmap(np.linspace(start, end, 256))
    return LinearSegmentedColormap.from_list(f'{selected_color_scheme}_optimized', colors)

def load_data(data_path, drug_type):
    file_path = data_path / f"{drug_type}_merged.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    df = pd.read_csv(file_path)
    metadata_cols = ['ID', 'names', 'labels', 'methods']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    X = df[feature_cols].values
    y = df['labels'].astype(str).values
    return X, y, df[['ID', 'names', 'labels']]

def load_best_params(results_dir, drug_type, fs_method, model_name):
    if fs_method == 'GA':
        model_dir = "lr-elasticnet" if model_name == 'LR' else ("xgboost" if model_name == 'XGBoost' else model_name.lower())
        param_file = results_dir / drug_type / "feature_selection" / "ga" / model_dir / "param_opt.log"
    else:
        raise ValueError(f"Unsupported fs_method: {fs_method}")
    
    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {param_file}")
        
    with open(param_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if content.startswith("Best params:"):
        return eval(content.split("Best params:", 1)[1].strip())
    raise ValueError("Unexpected format")

def create_model(model_name, params):
    if model_name == 'LR':
        params = params.copy()
        params['penalty'] = 'elasticnet'
        params['solver'] = 'saga'
        return LogisticRegression(**params, random_state=42, max_iter=2000)
    elif model_name == 'XGBoost':
        return xgb.XGBClassifier(**params, random_state=42)
    raise ValueError(f"Unsupported model: {model_name}")

def main():
    base_dir = Path(r"D:\Projects\SZW\sfjd-fentanyl-ml\src\稳定性考察数据\all_days_merged")
    output_merged_dir = base_dir / "output" / "merged_data_denoised"
    results_dir = base_dir / "output" / "results"
    
    day30_data_path = base_dir / "test" / "day30" / "output" / "merged_data_denoised"
    day30_shap_out = base_dir / "test" / "day30" / "shap"
    day30_shap_out.mkdir(parents=True, exist_ok=True)
    
    for drug_type, (fs_method_upper, model_name) in BEST_MODELS.items():
        print(f"\nProcessing {drug_type}...")
        try:
            # 1. Load baseline continuous training data
            train_df = pd.read_csv(output_merged_dir / f"{drug_type}_merged.csv")
            X_all = train_df.drop(columns=['ID', 'names', 'labels', 'methods'])
            y_all_raw = train_df['labels'].astype(str).values
            
            le = LabelEncoder()
            y_all = le.fit_transform(y_all_raw)
            
            X_train, _, y_train, _ = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
            
            fs_method = fs_method_upper.lower()
            fs_path = results_dir / drug_type / "feature_selection" / fs_method / "selected_features.txt"
            with open(fs_path, 'r') as f:
                selected_features = [line.strip() for line in f if line.strip()]
            
            X_train_fs = X_train[selected_features].values
            
            params = load_best_params(results_dir, drug_type, fs_method_upper, model_name)
            model = create_model(model_name, params)
            model.fit(X_train_fs, y_train)
            
            # 2. Load Day30 test data
            test_df = pd.read_csv(day30_data_path / f"{drug_type}_merged.csv")
            X_day30 = test_df.drop(columns=['ID', 'names', 'labels', 'methods'], errors='ignore')
            
            print("2. Loaded Day30 data")
            
            X_test_fs_df = pd.DataFrame(0.0, index=test_df.index, columns=selected_features)
            existing_features = [f for f in selected_features if f in X_day30.columns]
            X_test_fs_df[existing_features] = X_day30[existing_features]
            
            X_test_fs = X_test_fs_df.values
            feature_names = selected_features
            
            # 3. Handle SHAP
            print("3. Handling SHAP")
            X_test_df_shap = pd.DataFrame(X_test_fs, columns=feature_names)
            X_train_df_shap = pd.DataFrame(X_train_fs, columns=feature_names)
            
            if model_name == 'XGBoost':
                print("Running TreeExplainer...")
                explainer = shap.TreeExplainer(model)
                shap_values1 = explainer(X_test_df_shap)
                shap_values_raw = explainer.shap_values(X_test_df_shap)
            else:
                print("Running LinearExplainer...")
                explainer = shap.LinearExplainer(model, X_train_df_shap)
                shap_values1 = explainer(X_test_df_shap)
                shap_values_raw = explainer.shap_values(X_test_df_shap)
            print("SHAP values computed")
                
            # Formatting shap values 
            if isinstance(shap_values_raw, list):
                if len(shap_values_raw) == 2:
                    shap_values_to_plot = shap_values_raw[1]
                    shap_values1_obj = shap_values1[..., 1]
                else:
                    shap_values_to_plot = np.mean([np.abs(sv) for sv in shap_values_raw], axis=0)
                    shap_values1_obj = shap_values1[..., 0] 
            else:
                if len(shap_values_raw.shape) == 3:
                    shap_values_to_plot = shap_values_raw[:,:,1] if shap_values_raw.shape[2] == 2 else np.mean(np.abs(shap_values_raw), axis=2)
                    shap_values1_obj = shap_values1[..., 1] if shap_values1.shape[2] == 2 else shap_values1[..., 0]
                else:
                    shap_values_to_plot = shap_values_raw
                    shap_values1_obj = shap_values1
            
            print("4. Saving raw data")
            drug_out = day30_shap_out / drug_type
            drug_out.mkdir(parents=True, exist_ok=True)
            
            # --- Remove all-zero columns from X_test_df_shap ---
            non_zero_cols = X_test_df_shap.columns[(X_test_df_shap != 0).any(axis=0)]
            print(f"  Filtering all-zero columns: {len(X_test_df_shap.columns)} -> {len(non_zero_cols)}")
            X_test_df_shap = X_test_df_shap[non_zero_cols]
            
            shap_values_df = pd.DataFrame(shap_values_to_plot, columns=feature_names)[non_zero_cols]
            shap_values_to_plot_filtered = shap_values_df.values
            feature_names_filtered = non_zero_cols.tolist()
            
            mean_abs_shap = np.abs(shap_values_to_plot_filtered).mean(axis=0)
            shap_values3 = pd.Series(mean_abs_shap, index=feature_names_filtered)
            shap_values3.sort_values(ascending=False, inplace=True)
            
            # limiting to top 10 features
            MAX_DISPLAY = 10
            sorted_features = shap_values3.index.tolist()[:MAX_DISPLAY]
            sorted_shap_values = shap_values3.values[:MAX_DISPLAY]
            
            if len(sorted_features) == 0:
                print(f"No features found for {drug_type}.")
                continue
            
            # 1. Bar Chart Data
            bar_df = pd.DataFrame({
                'Feature': sorted_features,
                'Mean_Abs_SHAP': sorted_shap_values
            })
            try:
                bar_df.to_csv(drug_out / "bar_chart_data.csv", index=False, encoding='utf-8-sig')
            except PermissionError:
                print(f"  [Warning] bar_chart_data.csv is open for {drug_type}, skipping CSV update.")

            # 3. Beeswarm Chart Data
            beeswarm_shap = shap_values_df[sorted_features]
            beeswarm_feat = X_test_df_shap[sorted_features]
            try:
                beeswarm_shap.to_csv(drug_out / "beeswarm_shap_values.csv", index=False, encoding='utf-8-sig')
                beeswarm_feat.to_csv(drug_out / "beeswarm_feature_values.csv", index=False, encoding='utf-8-sig')
            except PermissionError:
                print(f"  [Warning] Beeswarm CSVs are open for {drug_type}, skipping CSV update.")
            
            print("5. Plotting...")
            print("  5.1 Setting up figure")
            # 5. Plotting Custom Figure matching shap_ref.py exactly
            base_length, fixed_increment, colored_ring_width = 4.0, 0.5, 2.0
            num_vars = len(sorted_features)
            one_oclock_offset = np.pi / 21
            if sorted_shap_values.sum() == 0:
                percentages = np.zeros_like(sorted_shap_values)
                widths = np.ones_like(sorted_shap_values) * (2 * np.pi / max(1, num_vars))
            else:
                percentages = (sorted_shap_values / sorted_shap_values.sum()) * 100
                widths = (sorted_shap_values / sorted_shap_values.sum()) * 2 * np.pi
            
            thetas = np.cumsum([0] + widths[:-1].tolist()) - one_oclock_offset
            total_lengths = [base_length + i * fixed_increment for i in range(num_vars)]
            
            # 2. Rose Chart Data
            rose_df = pd.DataFrame({
                'Feature': sorted_features,
                'Mean_Abs_SHAP': sorted_shap_values,
                'Percentage': percentages,
                'Width_rad': widths,
                'Theta_rad': thetas,
                'Radius': total_lengths
            })
            try:
                rose_df.to_csv(drug_out / "rose_chart_data.csv", index=False, encoding='utf-8-sig')
            except PermissionError:
                print(f"  [Warning] rose_chart_data.csv is open for {drug_type}, skipping CSV update.")
            
            inner_heights = [max(0, tl - colored_ring_width) for tl in total_lengths]
            inner_colors = ['#F5F5F5', '#FFFFFF'] * (num_vars // 2 + 1)
            
            cmap = create_optimized_cmap(cmap_base)
            v_min = np.quantile(sorted_shap_values, 0.25)
            v_max = np.quantile(sorted_shap_values, 0.75)
            if v_min == v_max:
                v_max += 1e-5
            color_norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
            colors = cmap(color_norm(sorted_shap_values))
            
            fig1 = plt.figure(figsize=(15, 8), dpi=300, facecolor='white')
            left_margin = 0.08
            right_margin = 0.10
            bottom_margin = 0.15
            top_margin = 0.12
            space_between = 0.12
            plot_bottom = bottom_margin
            plot_height = 1 - bottom_margin - top_margin
            total_plot_width = 1 - left_margin - right_margin - space_between
            left_plot_width = total_plot_width * 0.58
            right_plot_width = total_plot_width * 0.42
            
            cbar_left = left_margin
            colorbar_width = 0.015
            ax_cbar = fig1.add_axes([cbar_left, plot_bottom, colorbar_width, plot_height])
            sm = ScalarMappable(cmap=cmap, norm=color_norm)
            cbar = fig1.colorbar(sm, cax=ax_cbar, orientation='vertical')
            cbar.set_label('', size=16, labelpad=8, fontweight='bold')
            cbar.set_ticks([])
            cbar.ax.yaxis.set_ticks_position('left')
            ax_cbar.text(-3.5, 0.98, 'High', transform=ax_cbar.transAxes, ha='center', va='bottom', fontsize=14, fontweight='bold')
            ax_cbar.text(-3.5, 0.02, 'Low', transform=ax_cbar.transAxes, ha='center', va='top', fontsize=14, fontweight='bold')
            cbar.outline.set_visible(False)
            ax_cbar.text(-3.5, 0.5, 'Mean|SHAP Value|', transform=ax_cbar.transAxes, fontsize=14, rotation=90, va='center', fontweight='bold', ha='center')
            ax_cbar.set_facecolor('white')
            
            print("  5.2 Drawing main ax0")
            main_ax_left = cbar_left + colorbar_width + 0.05
            ax0 = fig1.add_axes([main_ax_left, plot_bottom, left_plot_width, plot_height])
            ax0.xaxis.tick_bottom()
            ax0.xaxis.set_label_position("bottom")
            ax0.invert_xaxis()
            
            ax0.barh(y=range(len(sorted_features)), width=sorted_shap_values, color=colors, height=0.65, edgecolor='white', linewidth=1.2)
            ax0.set_ylim(len(sorted_features) - 0.5, -0.5)  # Enforce exact ylim for ax0
            
            ax0.set_xlabel('Mean|SHAP Value|', size=16, labelpad=12, fontweight='bold')
            ax0.set_yticks([])
            ax0.spines[['left', 'top']].set_visible(False)
            ax0.spines['right'].set_position(('data', 0))
            ax0.spines['right'].set_visible(True)
            ax0.spines['bottom'].set_visible(True)
            ax0.tick_params(axis='x', which='major', direction='in', labelsize=14, length=8, pad=10, width=2)
            ax0.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
            ax0.tick_params(axis='x', which='minor', direction='in', length=5, width=1.5)
            for spine in ax0.spines.values():
                spine.set_linewidth(3)
                spine.set_color('#333333')
                
            max_val = max(sorted_shap_values) if len(sorted_shap_values) > 0 else 1.0
            for i, feature in enumerate(sorted_features):
                # 特征移至右侧横坐标（x=0）的右方
                label_x = -max_val * 0.02
                ax0.text(float(label_x), float(i), str(feature), ha='left', va='center', color='black', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))
                
                # 数字标签在柱子顶部左侧
                value_x = float(sorted_shap_values[i] + max_val * 0.03)
                ax0.text(value_x, float(i), f'{sorted_shap_values[i]:.4f}', ha='right', va='center', color='black', fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9))
                        
            print("  5.3 Drawing polar ax1")
            scale_factor = 2.0
            old_inset_size = min(left_plot_width, plot_height) * 0.75
            inset_size = old_inset_size * scale_factor
            old_inset_left = main_ax_left - 0.08
            old_inset_bottom = plot_bottom - 0.02
            
            # 向右和稍微向上移动，避免底部及数字与横坐标重叠 (适度微调，防止与柱状图数据标签重合)
            cx = old_inset_left + old_inset_size / 2 + 0.04
            cy = old_inset_bottom + old_inset_size / 2 + 0.06
            
            inset_left = cx - inset_size / 2
            inset_bottom = cy - inset_size / 2
            
            inset_ax_rect = [inset_left, inset_bottom, inset_size, inset_size]
            ax1 = fig1.add_axes(inset_ax_rect, projection='polar')
            ax1.patch.set_alpha(0)
            
            ax1.bar(x=thetas, height=inner_heights, width=widths, color=inner_colors[:len(thetas)], align='edge', edgecolor='white', linewidth=2.0)
            ax1.bar(x=thetas, height=[colored_ring_width] * num_vars, width=widths, bottom=inner_heights, color=colors, align='edge', edgecolor='white', linewidth=2.0)
            
            for i in range(num_vars): 
                label_angle_rad = float(thetas[i] + widths[i] / 2)
                label_radius = float(total_lengths[i] + 0.6)
                ax1.text(label_angle_rad, label_radius, f'{percentages[i]:.2f}%', ha='center', va='center', fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='#CCCCCC', alpha=0.9))
                        
            ax1.set_yticklabels([])
            ax1.set_xticklabels([])
            ax1.spines['polar'].set_visible(False)
            ax1.grid(False)
            ax1.set_theta_zero_location('N')
            ax1.set_theta_direction(-1)
            ax1.set_ylim(0, max(total_lengths) + 2.5)
            
            print("  5.4 Drawing shap.summary_plot ax2")
            right_plot_left = main_ax_left + left_plot_width + space_between
            ax2 = fig1.add_axes([right_plot_left, plot_bottom, right_plot_width, plot_height])
            
            shap.summary_plot(beeswarm_shap.values, beeswarm_feat, plot_type="dot", show=False, max_display=MAX_DISPLAY, cmap=selected_color_scheme, plot_size=None)
            
            ax2.set_yticklabels([])
            ax2.set_ylabel('')
            ax2.set_xlabel("SHAP Value", fontsize=16, fontweight='bold', labelpad=12)
            ax2.tick_params(axis='x', labelsize=14, direction='in', width=2, length=8, pad=10)
            ax2.spines['bottom'].set_linewidth(3)
            ax2.spines['bottom'].set_color('#333333')
            for spine_name in ['left', 'top', 'right']:
                if spine_name in ax2.spines:
                    ax2.spines[spine_name].set_visible(False)
                    
            if len(fig1.axes) > 3:
                cbar_ax_right = fig1.axes[-1]
                cbar_ax_right.set_ylabel('Feature Value', size=14, rotation=270, labelpad=15, fontweight='bold')
                cbar_ax_right.tick_params(labelsize=12, width=2)
                tick_labels = cbar_ax_right.get_yticklabels()
                if len(tick_labels) >= 2:
                    tick_labels[0].set_text("Low")
                    tick_labels[-1].set_text("High")
                    cbar_ax_right.set_yticklabels(tick_labels, fontsize=12, fontweight='bold')
                    
            ax2.set_ylim(-0.5, len(sorted_features) - 0.5)  # Enforce exact ylim for ax2
            
            print("  5.5 Saving File")
            output_filename_jpg = drug_out / f'shap_combined_{selected_color_scheme}.jpg'
            output_filename_svg = drug_out / f'shap_combined_{selected_color_scheme}.svg'
            
            plt.savefig(output_filename_jpg, dpi=300, facecolor='white', edgecolor='none')
            plt.savefig(output_filename_svg, facecolor='white', edgecolor='none', format='svg')
            plt.close(fig1)
            
            print(f"Successfully finished {drug_type}. Results saved to {drug_out}")
            
        except Exception as e:
            print(f"Failed processing {drug_type}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()