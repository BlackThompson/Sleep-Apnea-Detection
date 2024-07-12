import torch
from torch.utils.data import Dataset
import numpy as np


# 自定义Dataset
class TrainDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.data = np.load(data_path)
        self.labels = np.load(labels_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Mean of feature 1: 93.89775259581221
        Mean of feature 2: 66.02769326296182
        Standard deviation of feature 1: 3.70010817978346
        Standard deviation of feature 2: 10.405189878114523
        """

        heart = torch.tensor(self.data[idx, 0], dtype=torch.float32)
        heart = (heart - 93.89775) / 3.70011

        breath = torch.tensor(self.data[idx, 1], dtype=torch.float32)
        breath = (breath - 66.02769) / 10.40519

        heart = heart.unsqueeze(0)
        breath = breath.unsqueeze(0)
        combined = torch.cat((heart, breath), dim=0)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return combined, label


class TestDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        heart = torch.tensor(self.data[idx, 0], dtype=torch.float32)
        heart = (heart - 93.89775) / 3.70011

        breath = torch.tensor(self.data[idx, 1], dtype=torch.float32)
        breath = (breath - 66.02769) / 10.40519

        heart = heart.unsqueeze(0)
        breath = breath.unsqueeze(0)
        combined = torch.cat((heart, breath), dim=0)

        return combined


if __name__ == "__main__":
    data_path = r".\data\new_train\new_train_x.npy"
    labels_path = r".\data\new_train\new_train_y.npy"
    dataset = TrainDataset(data_path, labels_path)
    print(dataset[8000][0])
    print(dataset[8000][0].shape)
