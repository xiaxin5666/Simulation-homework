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
        self.classifier = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        x =  self.classifier(x)
        x = self.Softmax(dim=1)
        return x

model = CNN_MLP()
#print(model)