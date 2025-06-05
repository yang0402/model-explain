import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 加载 Iris 数据集
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)[['sepal length (cm)', 'sepal width (cm)']]
y = iris.data[:, 2]  # petal length (cm)

# 训练线性回归模型
model = LinearRegression()
model.fit(X, y)

# 获取模型系数和截距
coefficients = model.coef_
intercept = model.intercept_
feature_names = X.columns

# 打印模型方程
print("模型方程: petal length (cm) = {:.2f} + {:.2f} * sepal length (cm) + {:.2f} * sepal width (cm)".format(
    intercept, coefficients[0], coefficients[1]))

# 预测第一个样本并解释
sample = X.iloc[0]  # 第一个样本
prediction = model.predict([sample])[0]
print("\n第一个样本 (sepal length = {:.1f}, sepal width = {:.1f}) 的预测值: {:.2f} cm".format(
    sample[0], sample[1], prediction))
print("贡献解释:")
print("  截距: {:.2f} cm".format(intercept))
print("  sepal length 贡献: {:.2f} * {:.1f} = {:.2f} cm".format(coefficients[0], sample[0], coefficients[0] * sample[0]))
print("  sepal width 贡献: {:.2f} * {:.1f} = {:.2f} cm".format(coefficients[1], sample[1], coefficients[1] * sample[1]))

# 可视化预测 vs 实际值
plt.scatter(y, model.predict(X), color='blue')
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=2)
plt.xlabel("Actual Petal Length (cm)")
plt.ylabel("Predicted Petal Length (cm)")
plt.title("Actual vs Predicted Petal Length")
plt.savefig("actual_vs_predicted.png")
plt.close()

print("可视化结果已保存为 'actual_vs_predicted.png'")