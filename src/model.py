"""
model.py
--------
A deliberately small Masked Autoencoder (MAE), following the He et al. (2022)
recipe, scaled down to run comfortably on a laptop/single GPU:

  image -> split into patches -> randomly mask most patches
        -> encoder sees only the VISIBLE patches
        -> decoder gets [encoder output + mask tokens] and reconstructs
           ALL patches
        -> loss is computed only on the MASKED patches

This is the same core idea used in the brain-MRI foundation models
(BrainFound, BrainIAC) discussed earlier, just at toy scale.
"""

import torch
import torch.nn as nn


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, patch_size*patch_size*C), N = number of patches."""
    B, C, H, W = imgs.shape
    p = patch_size
    h, w = H // p, W // p
    x = imgs.reshape(B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 3, 5, 1)  # (B, h, w, p, p, C)
    x = x.reshape(B, h * w, p * p * C)
    return x


def unpatchify(x: torch.Tensor, patch_size: int, img_size: int, channels: int = 1) -> torch.Tensor:
    """Inverse of patchify: (B, N, patch_size*patch_size*C) -> (B, C, H, W)."""
    p = patch_size
    h = w = img_size // p
    B = x.shape[0]
    x = x.reshape(B, h, w, p, p, channels)
    x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, p, w, p)
    imgs = x.reshape(B, channels, h * p, w * p)
    return imgs


def random_masking(x: torch.Tensor, mask_ratio: float):
    """
    x: (B, N, D) patch embeddings.
    Returns:
      x_visible:  (B, N_keep, D)   -- only the kept patches, for the encoder
      mask:       (B, N)           -- 0 = kept, 1 = masked (removed)
      ids_restore:(B, N)           -- indices to undo the shuffle later
    """
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)  # random score per patch, per sample
    ids_shuffle = torch.argsort(noise, dim=1)          # ascending: low score = keep first
    ids_restore = torch.argsort(ids_shuffle, dim=1)     # inverse permutation

    ids_keep = ids_shuffle[:, :len_keep]
    x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones(B, N, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)  # unshuffle to original patch order

    return x_visible, mask, ids_restore


class TinyMAE(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 1,
        embed_dim: int = 96,
        encoder_depth: int = 4,
        encoder_heads: int = 4,
        decoder_dim: int = 64,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_channels = in_channels
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * in_channels

        # --- Patch embedding (a linear projection of each flattened patch) ---
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # --- Encoder: standard Transformer encoder, sees only visible patches ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=encoder_heads, dim_feedforward=embed_dim * 4,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

        # --- Decoder: smaller Transformer, sees visible tokens + mask tokens ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=decoder_heads, dim_feedforward=decoder_dim * 4,
            batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_pred = nn.Linear(decoder_dim, patch_dim)  # predict raw pixel values per patch

        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward_encoder(self, imgs: torch.Tensor, mask_ratio: float):
        patches = patchify(imgs, self.patch_size)            # (B, N, patch_dim)
        x = self.patch_embed(patches) + self.encoder_pos_embed  # (B, N, D)
        x_visible, mask, ids_restore = random_masking(x, mask_ratio)
        latent = self.encoder(x_visible)                     # (B, N_keep, D)
        return latent, mask, ids_restore

    def forward_decoder(self, latent: torch.Tensor, ids_restore: torch.Tensor):
        x = self.decoder_embed(latent)                       # (B, N_keep, decoder_dim)
        B, N = ids_restore.shape
        num_mask = N - x.shape[1]
        mask_tokens = self.mask_token.repeat(B, num_mask, 1)
        x_full = torch.cat([x, mask_tokens], dim=1)           # (B, N, decoder_dim) but shuffled order
        x_full = torch.gather(
            x_full, 1, ids_restore.unsqueeze(-1).repeat(1, 1, x_full.shape[-1])
        )                                                     # unshuffle back to original patch order
        x_full = x_full + self.decoder_pos_embed
        x_full = self.decoder(x_full)
        pred = self.decoder_pred(x_full)                      # (B, N, patch_dim)
        return pred

    def forward(self, imgs: torch.Tensor, mask_ratio: float = None):
        mask_ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)

        target = patchify(imgs, self.patch_size)
        loss_per_patch = ((pred - target) ** 2).mean(dim=-1)   # (B, N)
        loss = (loss_per_patch * mask).sum() / mask.sum()      # only masked patches count

        return loss, pred, mask
