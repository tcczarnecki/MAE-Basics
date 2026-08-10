"""
visualize.py
------------
Loads a trained TinyMAE checkpoint and shows, for a handful of samples:
  original slice | masked input (grey = removed patches) | reconstruction

This is the plot that actually lets you SEE whether the model learned
anything sensible before you trust any numbers.

Run (after train.py has produced mae_pretrained.pth):
    python src/visualize.py
"""

import torch
import matplotlib.pyplot as plt

from dataset import SyntheticMRIDataset
from model import TinyMAE, patchify, unpatchify

IMG_SIZE = 64
PATCH_SIZE = 8
MASK_RATIO = 0.75
NUM_EXAMPLES = 6


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyMAE(img_size=IMG_SIZE, patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO).to(device)
    model.load_state_dict(torch.load("mae_pretrained.pth", map_location=device))
    model.eval()

    ds = SyntheticMRIDataset(num_samples=NUM_EXAMPLES, img_size=IMG_SIZE, seed=123)  # different seed = unseen-style samples
    imgs = torch.stack([ds[i] for i in range(NUM_EXAMPLES)]).to(device)

    with torch.no_grad():
        loss, pred, mask = model(imgs, mask_ratio=MASK_RATIO)

    # Build a "masked view" of the input: keep visible patches, zero out masked ones
    target_patches = patchify(imgs, PATCH_SIZE)
    masked_patches = target_patches.clone()
    masked_patches[mask.bool()] = 0.0
    masked_imgs = unpatchify(masked_patches, PATCH_SIZE, IMG_SIZE)

    # IMPORTANT: the decoder outputs a prediction for every patch position,
    # including the ones the encoder saw for real -- but the loss only ever
    # trains the MASKED-patch predictions (see model.py). So predictions at
    # visible-patch positions are untrained noise and should be discarded.
    # The standard MAE visualization therefore splices: real pixels for
    # visible patches, model predictions only for masked patches.
    combined_patches = target_patches.clone()
    mask_bool = mask.bool().unsqueeze(-1).expand_as(pred)
    combined_patches[mask_bool] = pred[mask_bool]
    recon_imgs = unpatchify(combined_patches, PATCH_SIZE, IMG_SIZE)

    fig, axes = plt.subplots(NUM_EXAMPLES, 3, figsize=(6, 2 * NUM_EXAMPLES))
    col_titles = ["Original", "Masked input (75% removed)", "Reconstruction"]
    for row in range(NUM_EXAMPLES):
        for col, img in enumerate([imgs[row], masked_imgs[row], recon_imgs[row]]):
            ax = axes[row, col]
            ax.imshow(img.squeeze(0).cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)

    plt.tight_layout()
    plt.savefig("mae_reconstructions.png", dpi=150)
    print(f"Saved figure to mae_reconstructions.png (val loss on these samples: {loss.item():.4f})")


if __name__ == "__main__":
    main()