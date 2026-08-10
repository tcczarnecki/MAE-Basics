"""
finetune.py
-----------
THE key experiment: does MAE pretraining actually help?

We take the encoder half of TinyMAE (patch embed + positional embed +
transformer encoder -- everything except the decoder, which was only
ever needed for the reconstruction pretext task) and attach a small
classification head predicting "does this slice have a lesion?".

We train two versions at several label budgets (5%, 10%, 20%, 50%, 100%
of a labeled training pool):
  - "pretrained": encoder initialized from mae_pretrained.pth
  - "scratch":    encoder initialized randomly (no pretraining at all)

If MAE pretraining is doing its job, "pretrained" should beat "scratch"
by a growing margin as the label budget shrinks -- that's the actual
point of self-supervised pretraining: making scarce labels go further.

Run (after train.py has produced mae_pretrained.pth):
    python src/finetune.py
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from dataset import SyntheticMRIDataset
from model import TinyMAE, patchify

IMG_SIZE = 64
PATCH_SIZE = 8
EMBED_DIM = 96
ENCODER_DEPTH = 4
ENCODER_HEADS = 4

TRAIN_POOL_SIZE = 1000
VAL_SIZE = 300
LABEL_FRACTIONS = [0.05, 0.1, 0.2, 0.5, 1.0]
FT_EPOCHS = 15
LR = 1e-3
CHECKPOINT_PATH = "mae_pretrained.pth"


class MAEEncoderClassifier(nn.Module):
    """Encoder half of TinyMAE (no masking, full sequence) + a linear classification head."""

    def __init__(self, img_size, patch_size, embed_dim, encoder_depth, encoder_heads):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * 1  # 1 channel

        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=encoder_heads, dim_feedforward=embed_dim * 4,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        self.head = nn.Linear(embed_dim, 1)  # binary: lesion present logit

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, imgs):
        patches = patchify(imgs, self.patch_size)
        x = self.patch_embed(patches) + self.pos_embed  # (B, N, D), NO masking -- full image visible
        x = self.encoder(x)
        pooled = x.mean(dim=1)  # global average pool over all patch tokens
        return self.head(pooled).squeeze(-1)  # (B,) raw logits


def build_model(pretrained: bool) -> MAEEncoderClassifier:
    model = MAEEncoderClassifier(IMG_SIZE, PATCH_SIZE, EMBED_DIM, ENCODER_DEPTH, ENCODER_HEADS)
    if pretrained:
        full_mae = TinyMAE(
            img_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM,
            encoder_depth=ENCODER_DEPTH, encoder_heads=ENCODER_HEADS,
        )
        full_mae.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
        # Copy over only the encoder-side weights -- the decoder is discarded,
        # it was only ever needed for the reconstruction pretext task.
        model.patch_embed.load_state_dict(full_mae.patch_embed.state_dict())
        model.pos_embed.data.copy_(full_mae.encoder_pos_embed.data)
        model.encoder.load_state_dict(full_mae.encoder.state_dict())
    return model


def train_one(model, loader, epochs, device):
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = (torch.sigmoid(model(imgs)) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fixed pools: seed=1 for the labeled training pool, seed=2 for validation
    # (different seeds from train.py's pretraining data, so this is a fair test)
    train_pool = SyntheticMRIDataset(num_samples=TRAIN_POOL_SIZE, img_size=IMG_SIZE, seed=1, return_labels=True)
    val_ds = SyntheticMRIDataset(num_samples=VAL_SIZE, img_size=IMG_SIZE, seed=2, return_labels=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # Fixed shuffled index order, reused across fractions, so a 20% budget
    # is a superset of the 10% budget etc. -- avoids budget-to-budget noise
    # from just picking a totally different random subset each time.
    g = torch.Generator().manual_seed(42)
    shuffled_indices = torch.randperm(TRAIN_POOL_SIZE, generator=g).tolist()

    results = {"pretrained": [], "scratch": []}

    for frac in LABEL_FRACTIONS:
        n = max(4, int(TRAIN_POOL_SIZE * frac))  # at least 4 samples so batching works
        subset = Subset(train_pool, shuffled_indices[:n])
        loader = DataLoader(subset, batch_size=min(32, n), shuffle=True)

        for mode in ["pretrained", "scratch"]:
            model = build_model(pretrained=(mode == "pretrained"))
            model = train_one(model, loader, FT_EPOCHS, device)
            acc = evaluate(model, val_loader, device)
            results[mode].append(acc)
            print(f"label_fraction={frac:>4.2f} (n={n:>4d}) | {mode:>10s} | val accuracy = {acc:.3f}")

    # Plot the label-efficiency curve
    plt.figure(figsize=(6, 4))
    n_labels = [max(4, int(TRAIN_POOL_SIZE * f)) for f in LABEL_FRACTIONS]
    plt.plot(n_labels, results["pretrained"], marker="o", label="MAE-pretrained")
    plt.plot(n_labels, results["scratch"], marker="o", label="From scratch")
    plt.xlabel("Number of labeled training examples")
    plt.ylabel("Validation accuracy")
    plt.title("Label efficiency: pretrained vs. from-scratch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("label_efficiency.png", dpi=150)
    print("Saved label_efficiency.png")


if __name__ == "__main__":
    main()
