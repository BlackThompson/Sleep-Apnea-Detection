import torch
import torch.nn as nn


class TimeSeriesCNNBiGRU(nn.Module):
    def __init__(self):
        super(TimeSeriesCNNBiGRU, self).__init__()
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128, 64)  # 修改全连接层输入大小
        self.fc2 = nn.Linear(64, 3)
        self.relu = nn.ReLU()

        self.apool1 = nn.AvgPool1d(2)
        self.apool2 = nn.AvgPool1d(2)
        self.dropout = nn.Dropout(0.25)

        self.bigru = nn.GRU(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.apool1(x)  # First average pooling
        x = self.relu(self.conv2(x))
        x = self.apool2(x)  # Second average pooling
        # 调整维度以适应GRU输入 (batch_size, seq_length, input_size)
        x = x.permute(0, 2, 1)
        x, _ = self.bigru(x)
        x = x[:, -1, :]  # 取最后一个时间步的输出
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = TimeSeriesCNNBiGRU()

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
