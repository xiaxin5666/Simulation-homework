import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn

data_dir = "data"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

try:
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    print(f"图片信息：{images.shape}")
except StopIteration:
    print("加载出错")

class CNN_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        x =  self.classifier(x)
        x = self.softmax(x)
        return x

model = CNN_MLP()

model = CNN_MLP().to(device)
lr = 0.001
epochs = 20
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)
writer = SummaryWriter('logs')

for i in range(epochs):
    avgloss = 0
    accuracy = 0
    for j, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        avgloss += loss.item() / len(trainloader)
        accuracy += (outputs.argmax(1) == labels).sum().item() / len(trainloader.dataset)
    writer.add_scalar('loss', avgloss, i)
    writer.add_scalar('accuracy', accuracy, i)
    print(f"Epoch {i + 1}/{epochs}, Loss: {avgloss:.4f},")
    print(f"Accuracy: {accuracy:.4f}")

torch.save(model.state_dict(), 'cifar10_cnn(softmax).pth')

