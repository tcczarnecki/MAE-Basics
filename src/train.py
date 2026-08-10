"""
train.py
--------
Pretrains TinyMAE on the synthetic slice dataset by reconstruction.
No labels are used anywhere in this file -- that's the point of MAE
pretraining: it only ever sees raw images.

Run:
    python src/train.py
"""

import torch
from torch.utils.data import DataLoader

from dataset import SyntheticMRIDataset
from model import TinyMAE

IMG_SIZE = 64
PATCH_SIZE = 8
MASK_RATIO = 0.75
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = SyntheticMRIDataset(num_samples=2000, img_size=IMG_SIZE, seed=0)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = TinyMAE(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        mask_ratio=MASK_RATIO,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        for imgs in train_loader:
            imgs = imgs.to(device)
            loss, _, _ = model(imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        avg_loss = running_loss / len(train_ds)
        print(f"Epoch {epoch:02d}/{EPOCHS} | reconstruction MSE (masked patches): {avg_loss:.4f}")

    torch.save(model.state_dict(), "mae_pretrained.pth")
    print("Saved weights to mae_pretrained.pth")


if __name__ == "__main__":
    main()
