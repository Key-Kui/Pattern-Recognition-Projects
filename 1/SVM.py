# ===================== 1. 导入库 =====================
import pandas as pd  # 用来读取和处理表格数据（类似C++里的数据结构处理）
import numpy as np   # 用于数值计算（矩阵运算）
from sklearn.model_selection import train_test_split  # 划分训练集/测试集
from sklearn.preprocessing import LabelEncoder        # 用于把字符串变成数字
from sklearn.svm import SVC                           # SVM分类器
from sklearn.metrics import accuracy_score            # 计算准确率


# ===================== 2. 读取数据 =====================
# 注意：你的CSV是用“;”分隔的，不是逗号
data = pd.read_csv(r'E:\Study\模式识别与机器学习\作业\1\bank+marketing\bank\bank.csv', sep=';')  # 读取数据


# ===================== 3. 数据预处理 =====================

# 创建一个LabelEncoder对象（用于编码字符串）
le = LabelEncoder()

# 遍历每一列
for col in data.columns:
    # 如果这一列是字符串类型（object）
    if data[col].dtype == 'object':
        # 用LabelEncoder把字符串转成数字
        data[col] = le.fit_transform(data[col])


# ===================== 4. 划分特征和标签 =====================

# X：所有特征（去掉最后一列y）
X = data.drop('y', axis=1)  # axis=1表示按列删除

# y：标签（是否订阅）
y = data['y']


# ===================== 5. 划分训练集和测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# # 【新增内容】引入标准化工具
# from sklearn.preprocessing import StandardScaler

# # 创建标准化器
# scaler = StandardScaler()

# # 对训练集进行拟合和转换
# X_train = scaler.fit_transform(X_train)

# # 对测试集只进行转换（切记不能 fit 测试集，避免数据泄露）
# X_test = scaler.transform(X_test)


# ===================== 6. 创建SVM模型 =====================
# 【修改内容】加入 class_weight='balanced'
# model = SVC(kernel='rbf', class_weight='balanced')
model = SVC(kernel='rbf')


# ===================== 7. 训练模型 =====================

# 用训练数据训练模型
model.fit(X_train, y_train)


# ===================== 8. 预测 =====================

# 用测试数据进行预测
y_pred = model.predict(X_test)


# ===================== 9. 评估模型 =====================

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)

# 输出结果
print("模型准确率:", accuracy)
print("预测前10个:", y_pred[:10])
print("真实前10个:", y_test.values[:10])

# ===================== 混淆矩阵（Seaborn版） =====================
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 创建图像
plt.figure()

# 用seaborn画热力图
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# 标题
plt.title("Confusion Matrix")

# 轴标签
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# # 显示
# plt.show()
plt.savefig('figs/Confusion Matrix old.png')

from sklearn.metrics import roc_curve, auc

# 得到评分
y_score = model.decision_function(X_test)

# 计算曲线
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# 画图
plt.figure()
sns.lineplot(x=fpr, y=tpr)

plt.title(f"ROC Curve (AUC = {roc_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

# plt.show()
plt.savefig('figs/ROC Curve old.png')

# # ===================== PCA可视化分类结果 =====================
# from sklearn.decomposition import PCA

# # 降维到2D
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_test)

# # 创建DataFrame方便画图
# df_plot = pd.DataFrame()
# df_plot['PC1'] = X_pca[:, 0]
# df_plot['PC2'] = X_pca[:, 1]
# df_plot['True'] = y_test.values
# df_plot['Pred'] = y_pred

# # 画图（真实标签）
# plt.figure()
# sns.scatterplot(data=df_plot, x='PC1', y='PC2', hue='True')
# plt.title("True Labels Distribution")
# # plt.show()
# plt.savefig('figs/True PCA.png')

# # 画图（预测标签）
# plt.figure()
# sns.scatterplot(data=df_plot, x='PC1', y='PC2', hue='Pred')
# plt.title("Predicted Labels Distribution")
# # plt.show()
# plt.savefig('figs/Predicted PCA.png')