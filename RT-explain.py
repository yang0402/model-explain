# 导入必要的库
import numpy as np  # 数值计算库
import pandas as pd  # 数据处理库
import shap  # SHAP可解释性库
from sklearn.datasets import load_iris  # 加载鸢尾花数据集
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器
import matplotlib.pyplot as plt  # 绘图库

import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 使用文泉驿微米黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载 Iris 数据集
# iris是一个字典格式的数据集，包含数据、特征名、目标值等信息
iris = load_iris()
# 将数据转换为DataFrame格式，方便处理，并指定列名
X = pd.DataFrame(iris.data, columns=iris.feature_names)
# 目标变量（花的种类）
y = iris.target

# 训练随机森林模型
# n_estimators=100表示使用100棵决策树
# random_state=42固定随机种子，保证结果可复现
model = RandomForestClassifier(n_estimators=100, random_state=42)
# 使用数据训练模型
model.fit(X, y)

# 初始化 SHAP 解释器
# TreeExplainer是专门解释树模型的SHAP解释器
explainer = shap.TreeExplainer(model)

# 计算 SHAP 值
# SHAP值可以量化每个特征对模型输出的贡献
shap_values = explainer.shap_values(X)

# 可视化全局特征重要性
# summary_plot可以展示整体特征重要性
# plot_type="bar"表示使用条形图
# show=False表示不立即显示图形（因为我们要保存）
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
# 添加标题
plt.title("基于SHAP的特征重要性")
# 调整布局防止标签重叠
plt.tight_layout()
# 保存图形为PNG文件
plt.savefig("shap_feature_importance.png")
# 关闭图形，释放内存
plt.close()

# 可视化单个样本的解释（以第一个样本为例）
# force_plot展示单个样本的特征贡献
# explainer.expected_value[0]是基准值（模型输出的平均值）
# shap_values[0][0,:]是第一个样本对于第一类的SHAP值
# X.iloc[0,:]是第一个样本的特征值
# matplotlib=True表示使用matplotlib渲染（默认是JavaScript）
# show=False表示不立即显示图形
shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X.iloc[0,:], matplotlib=True, show=False)
# 添加标题
plt.title("样本0的SHAP力导向图")
# 调整布局
plt.tight_layout()
# 保存图形
plt.savefig("shap_force_plot_sample0.png")
# 关闭图形
plt.close()

# 打印保存信息
print("可视化结果已保存为 'shap_feature_importance.png' 和 'shap_force_plot_sample0.png'")