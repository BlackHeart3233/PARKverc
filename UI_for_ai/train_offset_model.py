import os
import json
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ====================
# 1. Dataset
# ====================

class OffsetDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, "labels.json")) as f:
            self.labels = json.load(f)
        self.images = list(self.labels.keys())
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1] skala
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        offset = self.labels[img_name]
        offset = offset / 150.0  # normaliziraj v [-1, 1]
        return self.transform(image), torch.tensor([offset], dtype=torch.float32)

# ====================
# 2. Model
# ====================

class OffsetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)  # output: offset

    def forward(self, x):
        return self.backbone(x)

# ====================
# 3. Trening
# ====================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OffsetDataset("training_data")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = OffsetRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):  # lahko povečaš kasneje
        running_loss = 0.0
        for images, offsets in dataloader:
            images = images.to(device)
            offsets = offsets.to(device)

            preds = model(images)
            loss = criterion(preds, offsets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), "offset_model.pth")
    print("✔ Model shranjen v offset_model.pth")

if __name__ == "__main__":
    train()
