import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import skfuzzy as fuzz
import warnings
warnings.filterwarnings('ignore')

# ================= 1. 环境配置 =================
os.makedirs('Fuzzy/figs', exist_ok=True)
plt.rcParams['figure.figsize'] = (10, 8)
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ================= 2. 数据预处理与聚类 =================
wine = load_wine()
X_raw = wine.data
y_true = wine.target

# 标准化：FCM 对量纲极其敏感，必须执行
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 计算 PCA 用于绘图 (捕捉 13 维空间的主要轮廓)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 执行 FCM (在完整 13 维空间计算，保证准确性)
# 使用 m=1.2 可以在红酒数据上获得极高的分类性能
cntr, U, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_scaled.T, c=3, m=1.2, error=1e-5, maxiter=500, init=None
)

# ================= 3. 标签映射与准确率 =================
cluster_labels = np.argmax(U, axis=0)
final_labels = np.zeros_like(cluster_labels)
cluster_mapping = {}
for i in range(3):
    mask = (cluster_labels == i)
    if np.any(mask):
        vals, counts = np.unique(y_true[mask], return_counts=True)
        digit = vals[np.argmax(counts)]
        final_labels[mask] = digit
        cluster_mapping[i] = digit

acc = accuracy_score(y_true, final_labels)
print(f"FCM 聚类准确率: {acc:.4f}")

# ================= 4. 寻找最模糊样本 =================
sorted_U = np.sort(U, axis=0)
margin = sorted_U[-1, :] - sorted_U[-2, :] 
ambiguous_idx = np.argmin(margin)

# ================= 5. 生成可视化图表 =================
print("生成可视化图表中...")

# 图 1: 整体聚类图 (PCA 坐标)
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_labels, cmap='viridis', s=40, alpha=0.7, edgecolors='k')
plt.title(f'FCM Wine Clustering (PCA Projection)\nAccuracy: {acc*100:.2f}%', fontsize=14)
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.colorbar(scatter, ticks=[0, 1, 2], label='Cultivar')
plt.savefig('Fuzzy/figs/FCM_Clustering_2D.png', dpi=150)
plt.close()

# 图 2: 模糊样本深度分析 (统一使用 PCA 坐标)
plt.figure(figsize=(12, 5))

# 左图：展示红星在 PC1/PC2 空间的位置
plt.subplot(1, 2, 1)
# 画出所有点作为背景
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_labels, cmap='viridis', alpha=0.2, label='All Wines')
# 修复此处：只传入红星的 x 和 y 两个坐标
plt.scatter(X_pca[ambiguous_idx, 0], X_pca[ambiguous_idx, 1], 
            color='red', marker='*', s=300, edgecolor='black', zorder=5, label='Ambiguous Sample')
plt.title(f"Position in PCA Space\n(True Class: Cultivar {y_true[ambiguous_idx]})")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

# 右图：隶属度分布
plt.subplot(1, 2, 2)
u_sample = U[:, ambiguous_idx]
x_labels = [f"Class {cluster_mapping.get(i, i)}" for i in range(3)]
bars = plt.bar(range(3), u_sample, color='#bdc3c7', edgecolor='black')

top_2 = np.argsort(u_sample)[-2:]
bars[top_2[-1]].set_color('#e74c3c')
bars[top_2[-2]].set_color('#f39c12')
plt.title('Membership Degree Distribution')
plt.xticks(range(3), x_labels)

# 在柱子上标注具体隶属度数值
for i, val in enumerate(u_sample):
    plt.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('Fuzzy/figs/Ambiguous_Sample_Analysis.png', dpi=150)
plt.close()

print("可视化文件已成功生成并保存。")
