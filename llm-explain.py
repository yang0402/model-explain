# 导入必要的库
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # HuggingFace Transformers 提供的模型和分词器
import torch      # PyTorch 深度学习框架
import shap       # SHAP 可解释性库
import matplotlib.pyplot as plt  # 用于绘图

# 加载预训练模型和分词器
# 使用的是在 SST-2 数据集上微调好的 BERT 模型（用于情感分类）
model_name = "textattack/bert-base-uncased-SST-2"
# 加载对应的 tokenizer，用于将文本转换为模型可接受的输入格式
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 加载模型，该模型是一个用于序列分类的 BERT 模型
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 定义一个函数，将模型包装成 SHAP 可以调用的形式
# 输入是文本列表，输出是预测的概率
def f(x):
    # 将输入文本编码为 token id，并进行填充、截断等处理
    inputs = tokenizer(x.tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
    # 不计算梯度，加快推理速度
    with torch.no_grad():
        # 调用模型得到 logits（未归一化的预测结果）
        logits = model(**inputs).logits
    # 对 logits 进行 softmax 操作，得到属于各类别的概率
    return torch.softmax(logits, dim=1).numpy()

# 创建 SHAP 解释器
# f 是我们定义的模型函数，tokenizer 是分词器，用于解析输入
explainer = shap.Explainer(f, tokenizer)

# 定义一个输入句子，用于解释其对预测结果的影响
text = "This movie was absolutely fantastic and I loved every minute of it!"

# 调用 explainer 计算 SHAP 值
shap_values = explainer([text])

# 提取正类（类别 1）的 SHAP 值
# 因为我们关心的是这句话对正面评价的贡献
shap_value = shap_values[0][..., 1]

# 创建一个图像画布，设置图像大小
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制 waterfall 图，展示每个词对预测结果的贡献
# show=False 表示不立即显示图像（适用于脚本中绘图）
shap.plots.waterfall(shap_value, max_display=20, show=False)

# 自动调整子图参数，防止重叠
plt.tight_layout()

# 保存图像为 PNG 文件，分辨率为 300 dpi，确保清晰
# bbox_inches='tight' 避免边缘裁剪
plt.savefig("shap_waterfall.png", dpi=300, bbox_inches='tight')

# 关闭图像，释放内存
plt.close()

# 输出提示信息，说明图像已保存成功
print("图像已保存为 'shap_waterfall.png'")