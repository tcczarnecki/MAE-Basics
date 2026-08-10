# MAE MRI Basics

A minimal Masked Autoencoder (MAE) implementation, structured the same
way as [pytorch-cnn-basics](https://github.com/tcczarnecki/pytorch-cnn-basics):
one file per concern, no framework magic, everything readable top to bottom.

## Project structure

```
mae-mri-basics/
├── src/
│   ├── dataset.py     # Data loading (synthetic slices for now)
│   ├── model.py        # TinyMAE: patchify, masking, encoder, decoder
│   ├── train.py         # Pretraining loop (unsupervised, no labels)
│   ├── visualize.py     # Inspect original / masked / reconstructed slices
│   └── finetune.py      # THE key experiment: pretrained vs. from-scratch label efficiency
├── mae_pretrained.pth   # (created after running train.py)
├── mae_reconstructions.png  # (created after running visualize.py)
├── label_efficiency.png # (created after running finetune.py)
└── README.md
```

Compared to your CNN repo, the shapes moving through the files are the
main difference to get comfortable with:

| CNN repo (`pytorch-cnn-basics`) | This repo (MAE) |
|---|---|
| `dataset.py` returns `(image, label)` | `dataset.py` returns just `image` — **no labels at all** |
| `model.py` outputs class logits | `model.py` outputs reconstructed pixel patches |
| `train.py` loss = cross-entropy vs. label | `train.py` loss = MSE vs. the original image, computed only on the patches that were masked out |
| `predict.py` outputs a class + confidence | `visualize.py` outputs a reconstructed image you inspect by eye |

That's the conceptual core of self-supervised pretraining: you never
touch a label, the "task" is manufactured from the image itself
(hide part of it, predict what's missing).

## How to run

```bash
cd mae-mri-basics
pip install torch matplotlib numpy   # or use your existing CNN repo's venv
python src/train.py        # pretrains TinyMAE on 2000 synthetic slices, ~a few minutes on CPU
python src/visualize.py    # produces mae_reconstructions.png
python src/finetune.py     # THE key experiment -- produces label_efficiency.png, takes longer (trains 10 small models)
```

`finetune.py` is the payoff: it takes the encoder half of the pretrained
model (discarding the decoder, which was only needed for reconstruction),
attaches a small classification head predicting "does this slice have a
lesion?", and compares it against an identical model with a randomly
initialized (non-pretrained) encoder -- at five different label budgets
(5% to 100% of a 1000-image labeled pool). The resulting plot should show
the pretrained version winning by the largest margin at the smallest label
budgets, which is the actual reason foundation-model pretraining matters:
it makes scarce labeled data go further.

You should see the reconstruction loss drop over epochs in `train.py`'s
output, and `mae_reconstructions.png` should show blurry-but-recognizable
reconstructions of the masked regions — MAE reconstructions are always
somewhat blurry (mean-squared-error loss favors "safe" averages), that's
expected and matches the original paper's results too.

## Reading order for understanding the code

1. **`model.py` → `patchify()` / `unpatchify()`** — get comfortable with
   the reshape gymnastics first; everything else builds on "image ⇄ grid
   of flattened patches."
2. **`model.py` → `random_masking()`** — this is the one non-obvious bit:
   it shuffles patches, keeps the first `N_keep`, and remembers
   `ids_restore` so the decoder can put things back in the right spots
   later. Trace through it with a tiny example (`N=4`, `mask_ratio=0.5`)
   on paper if the indexing isn't clicking.
3. **`model.py` → `TinyMAE.forward_encoder` / `forward_decoder`** — notice
   the encoder never sees the masked patches at all (cheaper + forces it
   to build genuinely useful representations), while the decoder gets
   mask tokens inserted back before predicting.
4. **`train.py`** — standard PyTorch loop, nothing MAE-specific here
   except that `imgs` is the only thing coming out of the dataloader.
5. **`visualize.py`** — the "did this actually learn anything" sanity check.

## Extending this toward the real project

To move from this toy setup to the actual portfolio project we scoped
(MAE pretraining on BraTS, then comparing label-efficiency against a
supervised baseline):

- Replace `SyntheticMRIDataset` with a `BraTSDataset` that loads real
  `.nii.gz` slices (via `nibabel`), extracts 2D axial slices, and
  normalizes intensities — `__getitem__`'s return shape (`1, H, W`)
  stays the same, so nothing else needs to change.
- `TinyMAE` as written already generalizes to bigger images/patch counts
  by just changing `img_size`/`patch_size`/`embed_dim` — no structural
  changes needed for a first real run.
- The label-efficiency experiment from the project plan is a *separate*
  script (`finetune.py`) that loads `mae_pretrained.pth`, replaces
  `decoder_pred` with a small segmentation head, and fine-tunes on a
  labeled subset — worth writing once this toy version makes sense to you.
