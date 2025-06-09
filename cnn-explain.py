# 导入必要的库
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import shap
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置随机种子，保证实验可重复
torch.manual_seed(0)
np.random.seed(0)

# 创建保存图像的目录
os.makedirs("shap_explanations", exist_ok=True)

# 数据预处理：将图像归一化到 [0,1] 并转为 Tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 定义一个简单的 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*5*5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 实例化模型并移动到设备上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# 训练模型（演示用，仅训练一个 epoch）
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 设置为训练模式
model.train()
for epoch in range(5):  # 只训练一个 epoch
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 将模型设置为评估模式
model.eval()

# 准备背景数据（用于 SHAP 解释器初始化）
background = next(iter(train_loader))[0].to(device)[:50]

# 初始化 SHAP DeepExplainer
explainer = shap.DeepExplainer(model, background)

# 获取测试样本（前1张）
test_images, test_labels = next(iter(test_loader))
test_images, test_labels = test_images[:1].to(device), test_labels[:1]

# 计算 SHAP 值
with torch.no_grad():
    predictions = model(test_images).cpu().numpy()
predicted_labels = np.argmax(predictions, axis=1)

shap_values = explainer.shap_values(test_images)

# 绘制并保存第一个测试样本的所有类别 SHAP 解释图
image_show = test_images[0].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
image_show = (image_show * 0.5 + 0.5).clip(0, 1)  # 反归一化

# 遍历所有类别，绘制每个类别的 SHAP 解释图
for class_idx in range(len(shap_values)):  # 对于 MNIST，类别数为 10
    plt.figure(figsize=(8, 4))

    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(image_show.squeeze(), cmap='gray')
    plt.title(f'Original\nLabel: {test_labels[0].item()}, Pred: {predicted_labels[0]}')
    plt.axis('off')

    # 显示 SHAP 解释图（当前类别）
    plt.subplot(1, 2, 2)
    shap.image_plot([shap_values[class_idx][0]], -image_show, show=False)
    plt.title(f'SHAP Explanation (Class {class_idx})')
    plt.axis('off')

    # 保存图像
    plt.savefig(f"explanation_class_{class_idx}.png", dpi=150, bbox_inches='tight')
    plt.close()

print("所有 SHAP 解释图已保存完成！")