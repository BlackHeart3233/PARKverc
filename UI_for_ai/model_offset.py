import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np

class OffsetRegressorDataset(Dataset):
    def __init__(self, label_json, max_detections=10):
        with open(label_json, "r") as f:
            self.labels = json.load(f)
        self.keys = list(self.labels.keys())
        self.max_detections = max_detections

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.labels[key]

        detections = item.get("nearest_lines", [])  # To so tvoje linije
        # Dodaj še avte, če jih imaš v mini datasetu ali dodaj v labels.json

        # Združimo detections (box + class_id) v matriko [max_detections x 5]
        # Format: [x1, y1, x2, y2, class_id]

        boxes = []
        for det in detections:
            bbox = det["bbox"]
            cls_id = 0  # predpostavimo, da linije imajo class_id=0
            if len(bbox) == 5:
                # če imaš OBB, lahko pretvoriš v XYXY, ali pa vzameš x,y,širina,višina
                # Za poenostavitev vzamemo prve 4 koordinate:
                box_xyxy = [bbox[0]-bbox[3]/2, bbox[1]-bbox[4]/2, bbox[0]+bbox[3]/2, bbox[1]+bbox[4]/2]
            else:
                box_xyxy = bbox  # predpostavimo že XYXY
            boxes.append(box_xyxy + [cls_id])

        # Padamo na max_detections
        boxes = boxes[:self.max_detections]
        while len(boxes) < self.max_detections:
            boxes.append([0, 0, 0, 0, -1])  # -1 pomeni "prazno"

        boxes = np.array(boxes, dtype=np.float32)

        offset = float(item["offset"])

        return torch.tensor(boxes), torch.tensor([offset], dtype=torch.float32)

# Model MLP
class OffsetRegressor(nn.Module):
    def __init__(self, max_detections=10, input_dim=5, hidden_dim=128):
        super().__init__()
        self.max_detections = max_detections
        self.input_dim = input_dim
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(max_detections * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # offset output
        )

    def forward(self, x):
        return self.fc(x)

# Primer trening loopa (lahko uporabiš svojo loss in optimizer)
def train():
    dataset = OffsetRegressorDataset("training_data/labels.json", max_detections=10)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = OffsetRegressor(max_detections=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20000):
        model.train()
        running_loss = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        print(f"Epoch {epoch+1} loss: {running_loss / len(dataset):.4f}")

    torch.save(model.state_dict(), "offset_regressor_mlp.pth")

if __name__ == "__main__":
    train()
