import os
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ====================
# 1. Dataset - uporablja labels_mini.json
# ====================

class OffsetFromLabelsMiniDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Naloži samo labels_mini.json
        with open(os.path.join(data_dir, "labels_mini.json")) as f:
            self.labels = json.load(f)

        self.samples = list(self.labels.items())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, sample = self.samples[idx]

        offset = sample["offset"] / 150.0  # normaliziraj offset [-1,1]
        img_w, img_h = sample.get("img_size", [1, 1])
        detections = sample.get("detections", [])

        # Vzemi do 4 bbox center točk (x, y), normalizirane na dimenzije slike
        centers = []
        for det in detections[:4]:
            cx, cy = det["center"]
            centers.extend([cx / img_w, cy / img_h])

        # Če manj kot 4, dopolni z ničlami
        while len(centers) < 8:
            centers.append(0.0)

        input_vec = np.array(centers, dtype=np.float32)

        return torch.tensor(input_vec), torch.tensor([offset], dtype=torch.float32)


# ====================
# 2. Model - preprost MLP
# ====================

class OffsetRegressorFromCoords(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),  # 8 vhodnih vrednosti (do 4 točke (x,y))
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# ====================
# 3. Trening
# ====================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OffsetFromLabelsMiniDataset("training_data")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = OffsetRegressorFromCoords().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1000):
        running_loss = 0.0
        for inputs, offsets in dataloader:
            inputs = inputs.to(device)
            offsets = offsets.to(device)

            preds = model(inputs)
            loss = criterion(preds, offsets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        avg_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")

    torch.save(model.state_dict(), "offset_model_from_coords.pth")
    print("✔ Model shranjen v offset_model_from_coords.pth")


if __name__ == "__main__":
    train()
