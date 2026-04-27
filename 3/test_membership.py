import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, '.')
from Fuzzy import FuzzyPatternRecognizer

# 加载数据
data = load_digits()
X = data.data
y = data.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
_, X_test_original_idx, _, _ = train_test_split(np.arange(len(X)), X, test_size=0.3, random_state=42, stratify=y)
X_test_original = X[X_test_original_idx]

# 训练模型
model = FuzzyPatternRecognizer(n_classes=10)
model.fit(X_train, y_train)
y_pred, memberships = model.predict(X_test)

# 找误分类样本
errors_idx = np.where(y_pred != y_test)[0]
sample_idx = errors_idx[0]
sample = X_test[sample_idx]
true_label = y_test[sample_idx]
pred_label = y_pred[sample_idx]

# 计算隶属度
sample_memberships = [model.compute_membership(sample, c) for c in range(10)]
sample_memberships_norm = np.array(sample_memberships) / np.sum(sample_memberships)

print(f'误分类样本: 真实={true_label}, 预测={pred_label}')
print('各类别归一化隶属度:')
for c in range(10):
    marker = ' <- 真实' if c == true_label else (' <- 预测' if c == pred_label else '')
    print(f'  数字{c}: {sample_memberships_norm[c]:.4f}{marker}')