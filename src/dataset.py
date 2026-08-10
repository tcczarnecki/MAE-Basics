"""
dataset.py
----------
Generates a small SYNTHETIC dataset of brain-slice-like grayscale images:
an elliptical "skull/brain" outline with a random blob "lesion" placed
inside it, plus noise. This lets you run and study the MAE pipeline
end-to-end without needing to register for BraTS first.

Swapping this out for real MRI slices later only requires changing
__getitem__ to load and normalize a real .png/.npy slice instead of
calling _make_synthetic_slice().
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def _make_synthetic_slice(img_size: int, rng: np.random.Generator):
    """Procedurally generate one fake 'brain slice'.

    Returns (img, label): img is a (img_size, img_size) float32 array in
    [0, 1]; label is 1 if a lesion blob was placed, else 0. train.py never
    looks at label -- it's only used by finetune.py's classification task.
    """
    yy, xx = np.mgrid[0:img_size, 0:img_size]
    cy, cx = img_size / 2, img_size / 2

    # Elliptical "brain" region
    a, b = img_size * 0.42, img_size * 0.32
    brain_mask = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1.0

    img = np.zeros((img_size, img_size), dtype=np.float32)
    img[brain_mask] = 0.55  # base tissue intensity

    # Smooth intensity variation so it's not flat (fake "tissue texture")
    texture = np.sin(xx / 6.0) * np.cos(yy / 7.0) * 0.05
    img[brain_mask] += texture[brain_mask]

    # Random "lesion" blob (brighter or darker patch) inside the brain, most of the time
    label = 1 if rng.random() < 0.5 else 0
    if label == 1:
        lesion_r = rng.uniform(img_size * 0.05, img_size * 0.12)
        # keep the lesion center inside the brain ellipse
        while True:
            lx = rng.uniform(cx - a * 0.6, cx + a * 0.6)
            ly = rng.uniform(cy - b * 0.6, cy + b * 0.6)
            if ((lx - cx) / a) ** 2 + ((ly - cy) / b) ** 2 <= 0.8:
                break
        dist = np.sqrt((xx - lx) ** 2 + (yy - ly) ** 2)
        lesion_mask = dist <= lesion_r
        delta = rng.uniform(0.2, 0.35) * rng.choice([-1, 1])
        img[lesion_mask & brain_mask] += delta

    img += rng.normal(0, 0.02, size=img.shape).astype(np.float32)  # sensor-ish noise
    return np.clip(img, 0.0, 1.0), label


class SyntheticMRIDataset(Dataset):
    """A fixed-size dataset of procedurally generated slices, seeded for reproducibility.

    return_labels=False (default): __getitem__ returns just the image tensor
      -- used by train.py, which does unsupervised MAE pretraining and must
      never look at labels.
    return_labels=True: __getitem__ returns (image_tensor, label_tensor)
      -- used by finetune.py's lesion-present/absent classification task.
    """

    def __init__(self, num_samples: int = 2000, img_size: int = 64, seed: int = 0, return_labels: bool = False):
        self.num_samples = num_samples
        self.img_size = img_size
        self.return_labels = return_labels
        self._rng = np.random.default_rng(seed)
        # Pre-generate everything up front — dataset is small, keeps __getitem__ simple/fast
        pairs = [_make_synthetic_slice(img_size, self._rng) for _ in range(num_samples)]
        self._images = [p[0] for p in pairs]
        self._labels = [p[1] for p in pairs]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self._images[idx]
        # (H, W) -> (1, H, W) : one channel, like a single-sequence MRI slice
        tensor = torch.from_numpy(img).unsqueeze(0)
        if self.return_labels:
            label = torch.tensor(self._labels[idx], dtype=torch.float32)
            return tensor, label
        return tensor
