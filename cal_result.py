import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset import TestDataset
from model.CNN_v1 import TimeSeriesCNN
from model.NeuroFetalNet import NeuroFetalNet


def cal_result(model, test_loader, model_path=None, device="cpu"):

    prediction = []

    if model_path:
        model.load_state_dict(torch.load(model_path))

    model.to(device)

    # Load test data
    for data in test_loader:
        data = data.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        prediction.append(predicted.item())

    test_ids = np.arange(0, len(prediction))
    predictions_df = pd.DataFrame({"id": test_ids, "label": prediction})
    predictions_df.to_csv("submission.csv", index=False, encoding="utf-8")
    print("Submission saved")


if __name__ == "__main__":
    model_path = r"E:\Python\Code\Sleep\checkpoint\best_model_0712-1419.pth"
    test_data_path = r".\data\testA\test_x_A.npy"
    test_dataset = TestDataset(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = NeuroFetalNet(num_classes=3, in_channels=2)

    cal_result(
        model=model, model_path=model_path, test_loader=test_loader, device="cuda"
    )
