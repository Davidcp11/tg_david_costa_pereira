#!/usr/bin/env python3
"""Gera bbox_mask_example.pdf (Fig 3.5) em layout VERTICAL a partir do kitti_minimal-001.

Uso:
  python3 scripts/gen_bbox_mask_en.py --lang en   # rotulos em ingles -> tg1en/
  python3 scripts/gen_bbox_mask_en.py --lang pt   # rotulos em portugues -> tg1/

RGB (em cima) + mapa de profundidade com a regiao de supervisao (embaixo),
empilhados verticalmente para que os titulos fiquem acima das imagens.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

_parser = argparse.ArgumentParser()
_parser.add_argument("--lang", choices=["pt", "en"], default="pt")
_args, _ = _parser.parse_known_args()
EN = _args.lang == "en"


def t(pt_str, en_str):
    return en_str if EN else pt_str


ROOT   = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "cap3"

RGB_PATH   = SAMPLE / "rgb.png"
DENSE_PATH = SAMPLE / "depth_dense.npy"
OUT        = ROOT / "figures" / ("en" if EN else "pt") / "bbox_mask_example.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ITA: textwidth=6.30in; figure included at width=\textwidth → save at 6.30in.
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

# Supervision bbox in normalized coords [y_min, y_max, x_min, x_max]
BBOX_NORM = [0.3784, 0.9761, 0.1582, 0.8613]

rgb         = np.asarray(Image.open(RGB_PATH))
depth_dense = np.load(DENSE_PATH)
H, W        = depth_dense.shape

y_min_n, y_max_n, x_min_n, x_max_n = BBOX_NORM
y_min = int(y_min_n * H);  y_max = int(y_max_n * H)
x_min = int(x_min_n * W);  x_max = int(x_max_n * W)

mask_inside = np.zeros((H, W), dtype=bool)
mask_inside[y_min:y_max, x_min:x_max] = True

valid = depth_dense > 0
vmin = float(depth_dense[valid].min())              if valid.any() else 0.0
vmax = float(np.percentile(depth_dense[valid], 95)) if valid.any() else 1.0

depth_inside  = np.ma.masked_where(~mask_inside, depth_dense)
depth_outside = np.ma.masked_where(mask_inside,  depth_dense)

# Empilhado verticalmente: cada painel ocupa a largura toda; titulos acima das imagens.
img_h = _TW * H / W
fig, axes = plt.subplots(2, 1, figsize=(_TW, 2 * img_h + 0.9), constrained_layout=True)

# Cima: RGB + retangulo da bbox
axes[0].imshow(rgb)
rect = mpatches.Rectangle(
    (x_min, y_min), x_max - x_min, y_max - y_min,
    linewidth=2.5, edgecolor="lime", facecolor="none",
    label=t("Região de supervisão", "Supervision region"))
axes[0].add_patch(rect)
axes[0].legend(loc="upper right", framealpha=0.85)
axes[0].set_title(t("Imagem RGB com a região de supervisão de profundidade",
                    "RGB image with the depth supervision region"))
axes[0].axis("off")

# Baixo: mapa de profundidade — dentro colorido, fora esmaecido
axes[1].imshow(depth_outside, cmap="turbo", vmin=vmin, vmax=vmax, alpha=0.25)
im = axes[1].imshow(depth_inside, cmap="turbo", vmin=vmin, vmax=vmax)
axes[1].set_title(t("Mapa de profundidade: região válida para supervisão",
                    "Depth map: valid region for supervision"))
axes[1].axis("off")

fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01,
             label=t("metros", "meters"))
fig.savefig(OUT)
plt.close(fig)
print(t(f"Salvo em {OUT}", f"Saved to {OUT}"))
