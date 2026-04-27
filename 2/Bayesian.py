import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from scipy.stats import multivariate_normal
from tqdm import tqdm

# ================= 1. 环境与可视化配置 =================
os.makedirs('figs', exist_ok=True)
nature_blue = "#00468B"
sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] # 中文支持
plt.rcParams['axes.unicode_minus'] = False

# ================= 2. 数据读取与预处理 =================
print("正在从本地 Scikit-learn 加载威斯康星乳腺癌数据集...")
data = load_breast_cancer()

X = data.data
# 重点：sklearn默认恶性为0，良性为1。这里取反，使 恶性(M)=1，良性(B)=0
y = 1 - data.target 

# 划分数据集 (7:3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ================= 3. 贝叶斯模型参数估计 (训练) =================
print("正在进行贝叶斯参数估计 (多元高斯分布)...")
classes = np.unique(y_train)
priors = {}
means = {}
covs = {}

for c in classes:
    X_c = X_train[y_train == c]
    priors[c] = X_c.shape[0] / X_train.shape[0]
    means[c] = np.mean(X_c, axis=0)
    # 计算完整的30x30协方差矩阵，加上1e-5对角阵保证可逆性
    covs[c] = np.cov(X_c, rowvar=False) + np.eye(X.shape[1]) * 1e-5 

# ================= 4. 模型预测 (测试) =================
y_pred = []
y_prob = []

# # 【核心：对数域阈值偏移】
# # 在30维空间中，这里设定 threshold_shift = 30.0 
# # 它的数学意义是：哪怕模型认为“良性”的对数概率比“恶性”高出 30，
# # 考虑到极度不平衡的医疗代价，我们依然强行把它判定为“恶性(1)”（宁错杀不放过）
# threshold_shift = 46.99

# for x in tqdm(X_test, desc="贝叶斯推断进度", unit="样本"):
#     posteriors = {}
#     for c in classes:
#         likelihood = multivariate_normal.logpdf(x, mean=means[c], cov=covs[c], allow_singular=True)
#         posteriors[c] = likelihood + np.log(priors[c])
    
#     # 这部分仅仅是为了算出一个 0~1 的概率给 ROC 画曲线用，不参与分类决策
#     max_log = max(posteriors.values())
#     exp_posts = {k: np.exp(v - max_log) for k, v in posteriors.items()}
#     prob_1 = exp_posts[1] / sum(exp_posts.values())
#     y_prob.append(prob_1)
    
#     # 【最小风险贝叶斯决策（Log 域绝对稳定版）】
#     log_prob_0 = posteriors[0]
#     log_prob_1 = posteriors[1]
    
#     if (log_prob_1 - log_prob_0) > -threshold_shift:
#         y_pred.append(1)  # 判定为恶性
#     else:
#         y_pred.append(0)  # 判定为良性

for x in tqdm(X_test, desc="贝叶斯推断进度", unit="样本"):
    posteriors = {}
    for c in classes:
        # 核心：计算对数似然 + 对数先验
        likelihood = multivariate_normal.logpdf(x, mean=means[c], cov=covs[c], allow_singular=True)
        posteriors[c] = likelihood + np.log(priors[c])
    
    # 预测为后验概率最大的类别
    pred_c = max(posteriors, key=posteriors.get)
    y_pred.append(pred_c)
    
    # Softmax 计算正类概率用于 ROC
    max_log = max(posteriors.values())
    exp_posts = {k: np.exp(v - max_log) for k, v in posteriors.items()}
    prob_1 = exp_posts[1] / sum(exp_posts.values())
    y_prob.append(prob_1)

y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['良性 (0)', '恶性 (1)']))

# ================= 5. 可视化与保存 =================
print("正在生成高清图表...")

# 5.1 混淆矩阵
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['良性 (0)', '恶性 (1)'], yticklabels=['良性 (0)', '恶性 (1)'])
plt.ylabel('真实标签', fontweight='bold')
plt.xlabel('预测标签', fontweight='bold')
plt.title('贝叶斯决策混淆矩阵', fontweight='bold')
plt.tight_layout()
plt.savefig('figs/Confusion_Matrix_old.png', dpi=300)
plt.close()

# 5.2 ROC 曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color=nature_blue, lw=2, label=f'贝叶斯分类器 (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例率 (FPR)', fontweight='bold')
plt.ylabel('真正例率 (TPR)', fontweight='bold')
plt.title('受试者工作特征曲线 (ROC)', fontweight='bold')
plt.legend(loc="lower right")
sns.despine()
plt.tight_layout()
plt.savefig('figs/ROC_Curve_old.png', dpi=300)
plt.close()

print("运行完毕，图表已保存至 figs/ 目录。")