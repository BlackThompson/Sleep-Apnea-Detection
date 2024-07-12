import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time


class EarlyStopping:
    def __init__(
        self,
        patience=15,
        verbose=True,
        delta=0,
        checkpoint_dir=r"E:/Python/Code/Sleep/checkpoint",
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.best_model = None
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_filename = None

    def __call__(self, val_loss, model):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score >= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.best_model = model

        return self.best_model

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )

        if self.checkpoint_filename is None:
            # 获取当前时间戳
            timestamp = time.strftime("%m%d-%H%M")
            # 生成包含时间戳的文件名
            self.checkpoint_filename = (
                f"{self.checkpoint_dir}/best_model_{timestamp}.pth"
            )

        # 保存模型
        torch.save(model.state_dict(), self.checkpoint_filename)
        # 更新最小验证损失
        self.val_loss_min = val_loss


# train and evaluate model
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    learning_rate=0.001,
    num_epochs=100,
    patience=15,
    device="cpu",
):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)
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
                data = data.to(device)
                labels = labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

        best_model = early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            return best_model

    return best_model


# test model
def test_model(model, test_loader, model_path=None, device="cpu"):

    # 如果model_path不为空，则加载模型
    if model_path:
        model.load_state_dict(torch.load(model_path))

    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")
    return 100 * correct / total
