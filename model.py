import torch
import torch.nn as nn
from torch.nn import functional as F


class HybridSN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv3d(in_channels, 8, kernel_size=(7, 3, 3))
        self.conv2 = nn.Conv3d(8, 16, kernel_size=(5, 3, 3))
        self.conv3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3))

        self.conv4 = nn.Conv2d(576, 64, kernel_size=(3, 3))

        # fully connected layers
        self.fc1 = nn.Linear(18496, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.drop = nn.Dropout(p=0.4)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = x.view(-1, x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        x = self.relu(self.conv4(x))

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)

        return x


if __name__ == "__main__":
    x = torch.randn(4, 1, 30, 25, 25)

    model = HybridSN(5, 1)
    model.eval()

    with torch.no_grad():
        out = model(x)
        print(out.shape)
