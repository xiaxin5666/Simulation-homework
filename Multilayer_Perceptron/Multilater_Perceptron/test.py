import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

data_dir = "data"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

try:
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    print(f"图片信息：{images.shape}")
except StopIteration:
    print("加载出错")

import torch.nn as nn
class CNN_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Linear(128, 10)
    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        x = self.softmax(x)
        return self.classifier(x)

model = CNN_MLP().to(device)
model.load_state_dict(torch.load("cifar10_cnn(softmax).pth"))

t = 0
c = 0

y_true = []
y_pred = []
with torch.no_grad():
    i = 0
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # 预测类别
        t += labels.size(0)  # 累加总样本数（每个批次4个，labels.size(0)=4）
        c += (predicted == labels).sum().item()  # 累加正确数
        y_true.append(labels.cpu().numpy())
        y_pred.append(predicted.cpu().numpy())
        i += 1
        if i%5 == 0:
            print(f"第{i}批次的预测结果：")
            print(f'预测结果：{", ".join(classes[label] for label in predicted)}')
            print(f'真实结果：{", ".join(classes[label] for label in labels)}')
            #预测概率
            probabilities = nn.functional.softmax(outputs, dim=1)
            topk, topclass = probabilities.topk(1, dim=1)
            print(f'预测概率（百分比）：{topk.squeeze()*100}')
        if i > 3000:
            break
    #画出混淆矩阵
    y_true_flat = np.concatenate(y_true)
    y_pred_flat = np.concatenate(y_pred)
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    print(cm)
# 计算准确率
accuracy = 100 * c / t
print(f"模型在训练集上的准确率：{accuracy:.2f}%")