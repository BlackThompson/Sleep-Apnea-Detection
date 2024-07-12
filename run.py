import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model.CNN_v1 import TimeSeriesCNN
from model.CRNN_v1 import TimeSeriesCNNBiGRU
from model.NeuroFetalNet import NeuroFetalNet
from dataset import TrainDataset, TestDataset
from train_eval import train_model, test_model
from cal_result import cal_result


if __name__ == "__main__":

    # 固定随机数种子
    torch.manual_seed(2024)

    # 定义数据集和数据加载器
    data_path = r".\data\new_train\new_train_x.npy"
    labels_path = r".\data\new_train\new_train_y.npy"
    dataset = TrainDataset(data_path, labels_path)

    # 按照9:0.5:0.5划分数据集
    train_size = int(0.9 * len(dataset))
    val_size = int(0.05 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # 定义数据加载器
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 实例化模型、损失函数和优化器
    # model = TimeSeriesCNN()
    # model = TimeSeriesCNNBiGRU()
    model = NeuroFetalNet(num_classes=3, in_channels=2)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 训练模型
    model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        learning_rate=0.001,
        num_epochs=200,
        patience=10,
        device=device,
    )

    # 测试模型
    test_model(model, test_loader, device=device)

    # 计算结果
    test_data_path = r".\data\testA\test_x_A.npy"
    test_dataset = TestDataset(test_data_path)
    test_result_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    cal_result(model, test_result_loader, device=device)
