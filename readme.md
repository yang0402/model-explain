## SHAP 事后解释性

### 第一张图（Feature Importance based on SHAP（特征重要性图））

1. 总长度：反映特征的整体重要性（对所有类别的平均影响）。
2. 每种颜色的长度：反映该特征对某个特定类别的预测影响大小。
3. 纵轴：列出了 Iris 数据集的四个特征（petal length, petal width, sepal length, sepal width）。横轴：表示 mean(|SHAP value|)，即特征对模型输出的平均影响大小（绝对值的平均）。

### 第二张图（SHAP Force Plot for Sample 0（单个样本在 的力图））

1. 横轴：表示预测值（这里是对 Class 0 的预测概率），从左到右增加。base value（0.3）：模型的平均预测值（即在没有任何特征影响时的预测概率）。f(x) = 1.00：最终预测值，说明模型预测这个样本属于 Class 0 的概率是 1.0（非常高）。红色箭头：表示特征值对预测的正向贡献（推高预测概率）。蓝色箭头：表示特征值对预测的负向贡献（降低预测概率），但这张图中没有蓝色箭头。
2. sepal length (cm) = 5.1：这个特征值对预测概率有正向贡献（红色），稍微推高了预测值。petal width (cm) = 0.2：这个特征值有较大的正向贡献，显著推高了预测概率。petal length (cm) = 1.4：这个特征值也有较大的正向贡献，进一步推高了预测概率。 3.这张图表明，对于第 0 个样本，小的花瓣宽度 (0.2 cm) 和花瓣长度 (1.4 cm) 是模型预测其为 Class 0 的主要原因，最终使预测概率达到 1.0。

## 模型本身的可解释性

### 第三张图(actual_vs_predicted.png)

1. 线性回归的权重（系数）直接反映特征对预测的线性影响，无需额外工具即可理解。这种方法与事后解释（如 SHAP）不同，解释性来自模型本身的设计。
2. 线性回归模型的预测公式是：petal length = intercept + coefficient1 _ sepal length + coefficient2 _ sepal width。每个系数（coefficient）直接表示该特征每单位变化对预测变量（petal length）的线性影响。这是模型结构本身提供的解释性。
3. 我们使用 Iris 数据集，基于 sepal length 和 sepal width 预测 petal length，并展示模型参数如何解释预测。
4. 横轴（Actual Petal Length (cm)）：表示真实的 petal length 值（目标变量）。
   纵轴（Predicted Petal Length (cm)）：表示模型预测的 petal length 值。
   蓝色散点：每个点代表一个样本，横轴是其真实值，纵轴是模型预测值。
   红色虚线：一条从左下到右上的对角线（y=x），表示预测值等于真实值的情况。如果所有点都落在红线上，说明模型预测完全准确。

## 基于 BERT 的文本分类模型的局部后处理可解释性分析

### 第四张图（shap_waterfall.png(即瀑布图)）

1. 图中显示基准值为 0.973，这意味着在没有考虑任何输入词的情况下，模型对正类的初始预测概率为 0.973。图中显示最终预测值为 1.0，表示模型对当前输入句子的预测结果非常接近正类（满分 1.0）。
2. 每个词的 SHAP 值表示它对预测结果的贡献：
   正向贡献 ：增加模型预测为正类的概率。
   负向贡献 ：减少模型预测为正类的概率。
3. X 轴（水平轴） ：
   表示模型的预测值。
   左侧起点是基准值（Base Value），即在没有考虑任何特征时的预测值。
   右侧终点是最终的预测值。
4. Y 轴（垂直轴） ：
   列出了输入句子中的各个词（token），按其对预测结果的贡献大小排序。
   每个词的 SHAP 值表示该词对预测结果的正向或负向影响。
5. 颜色 ：
   红色 ：表示正向贡献，即该词增加了模型预测为正类的概率。
   蓝色 ：表示负向贡献，即该词减少了模型预测为正类的概率。
6. 数值 ：
   每个词旁边的数字表示其 SHAP 值，反映了该词对预测结果的具体贡献大小。
7. 基准值和最终预测值 ：
   基准值 ：图表左侧的起始点，表示在没有考虑任何特征时的预测值。
   最终预测值 ：图表右侧的终点，表示考虑所有特征后的最终预测值。
