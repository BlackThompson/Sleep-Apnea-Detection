import torch
import torch.nn as nn


# 定义神经网络模型
class TimeSeriesCNN(nn.Module):
    def __init__(self):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * (180 // 4), 128)
        self.fc2 = nn.Linear(128, 3)
        self.relu = nn.ReLU()

        self.apool1 = nn.AvgPool1d(2)
        self.apool2 = nn.AvgPool1d(2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.apool1(x)  # First average pooling
        x = self.relu(self.conv2(x))
        x = self.apool2(x)  # Second average pooling
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = TimeSeriesCNN()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # print(model)
    x = torch.randn(64, 2, 180)
    # x = torch.randn(64, 180, 2)
    x = x.to(device)
    model = model.to(device)
    print(model(x).shape)
