#!/usr/bin/env python3
"""Gera depth_densification_example.pdf para tg1en/ (EN) ou tg1/ (PT)."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

_parser = argparse.ArgumentParser()
_parser.add_argument("--lang", choices=["pt", "en"], default="pt")
_args, _ = _parser.parse_known_args()
EN = _args.lang == "en"


def t(pt_str, en_str):
    return en_str if EN else pt_str


def dilate_vis(d, r=2):
    """Grey dilation (max over a (2r+1)x(2r+1) window) so the sparse LiDAR
    points become visible when displayed (each valid point -> small blob).
    Visualization only; does not change the reported valid fraction."""
    H, W = d.shape
    out = d.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            sy0, sy1 = max(0, dy), min(H, H + dy)
            sx0, sx1 = max(0, dx), min(W, W + dx)
            ty0, ty1 = max(0, -dy), min(H, H - dy)
            tx0, tx1 = max(0, -dx), min(W, W - dx)
            out[ty0:ty1, tx0:tx1] = np.maximum(out[ty0:ty1, tx0:tx1], d[sy0:sy1, sx0:sx1])
    return out


ROOT = Path(__file__).resolve().parent.parent
SAMPLE  = ROOT / "data" / "cap3"

RGB_PATH    = SAMPLE / "rgb.png"
SPARSE_PATH = SAMPLE / "depth_sparse.png"
DENSE_PATH  = SAMPLE / "depth_dense.npy"
OUT         = ROOT / "figures" / ("en" if EN else "pt") / "depth_densification_example.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ITA: textwidth=6.30in; figure included at width=0.85\textwidth → save at 0.85*6.30 in.
_TW = 6.30
_FS = 10
plt.rcParams.update({
    'font.size':        _FS,
    'axes.titlesize':   _FS,
    'axes.labelsize':   _FS,
    'xtick.labelsize':  _FS - 1,
    'ytick.labelsize':  _FS - 1,
    'legend.fontsize':  _FS - 1,
    'figure.dpi':       150,
})

rgb          = np.asarray(Image.open(RGB_PATH))
depth_sparse = np.asarray(Image.open(SPARSE_PATH), dtype=np.uint16).astype(np.float32) / 256.0
depth_dense  = np.load(DENSE_PATH)

mask = depth_sparse > 0
vmin = float(depth_sparse[mask].min())           if mask.any() else 0.0
vmax = float(np.percentile(depth_sparse[mask], 95)) if mask.any() else 1.0

H, W = rgb.shape[:2]
FIG_W = 0.85 * _TW          # 5.355 in — matches width=0.85\textwidth in LaTeX
row_h = FIG_W * H / W       # keep image aspect ratio per row
fig_h = 3 * row_h + 0.6     # 3 rows + margin for titles
fig, axes = plt.subplots(3, 1, figsize=(FIG_W, fig_h), constrained_layout=True)

axes[0].imshow(rgb)
axes[0].set_title("RGB")
axes[0].axis("off")

# Dilata os pontos só para visualização (o % de válidos no título usa o original).
sparse_disp = dilate_vis(depth_sparse, r=1)
sparse_vis  = np.ma.masked_where(sparse_disp <= 0, sparse_disp)
axes[1].imshow(sparse_vis, cmap="turbo", vmin=vmin, vmax=vmax)
axes[1].set_title(t(f"Profundidade esparsa (válido={mask.mean()*100:.1f}%)",
                    f"Sparse depth (valid={mask.mean()*100:.1f}%)"))
axes[1].axis("off")

im = axes[2].imshow(depth_dense, cmap="turbo", vmin=vmin, vmax=vmax)
axes[2].set_title(t("Profundidade densa (densificada)", "Dense depth (densified)"))
axes[2].axis("off")

fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01,
             label=t("metros", "meters"))
fig.savefig(OUT)
plt.close(fig)
print(t(f"Salvo em {OUT}", f"Saved to {OUT}"))
