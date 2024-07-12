import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split


# 定义数据集和数据加载器
data_path = r"./data/train/train_x.npy"
labels_path = r"./data/train/train_y.npy"
dataset = TimeSeriesDataset(data_path, labels_path)

test_data = np.load(r"./data/testA/test_x_A.npy")
test_sample =  

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
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

test_loader1 = DataLoader(test_sample, batch_size=1, shuffle=False)


# 实例化模型、损失函数和优化器
model = TimeSeriesCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练和验证模型
def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, patience
):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break


# 训练模型
train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10
)


# 测试模型
def test_model(model, test_loader):
    model.load_state_dict(torch.load("best_model2.pth"))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = model(test_sample)
        _, predicted = torch.max(outputs.data, 1)

    #     for data, labels in test_loader:
    #         outputs = model(data)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    # print(f'Test Accuracy: {100 * correct / total:.2f}%')

    # Convert predictions to numpy array
    test_predictions_np = predicted.numpy()

    # Generate IDs starting from 0
    test_ids = np.arange(test_data.shape[0])

    # Save predictions to a CSV file
    # predictions_df = pd.DataFrame(test_predictions_np, columns=['Prediction'])
    predictions_df = pd.DataFrame({"id": test_ids, "label": test_predictions_np})
    predictions_df.to_csv("./mnt/data/test_predictions2.csv", index=False)

    print("Test predictions saved to test_predictions.csv")


# 测试模型
test_model(model, test_loader1)
