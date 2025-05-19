import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time

# Ustvari direktorij za loge
writer = SummaryWriter("runs/dummy_test")

for epoch in range(10):
    accuracy = np.random.rand()  # naključna "accuracy"
    loss = np.random.rand()  # naključni "loss"

    writer.add_scalar("Accuracy", accuracy, epoch)
    writer.add_scalar("Loss", loss, epoch)

    # Dummy slika s kvadratki
    dummy_image = np.zeros((3, 256, 256))
    dummy_image[:, 50:200, 50:200] = np.random.rand(3, 1, 1)  # naključen kvadrat

    writer.add_image("Example Image", dummy_image, epoch)

writer.close()
