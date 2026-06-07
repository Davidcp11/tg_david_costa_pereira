#!/usr/bin/env python3
"""Gera duas figuras pedagógicas sobre densificação morfológica:

  densification_kernels.png  — os quatro tipos de kernel
  densification_steps.png    — progressão D_sparse → D_dense (matriz 7x7)

Sem dependência de cv2 — usa apenas numpy e matplotlib.

Use --lang en to generate English labels and save to tg1en/.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

_parser = argparse.ArgumentParser()
_parser.add_argument("--lang", choices=["pt", "en"], default="pt")
_args, _ = _parser.parse_known_args()
EN = _args.lang == "en"


def t(pt_str, en_str):
    return en_str if EN else pt_str


_ROOT = Path(__file__).resolve().parent.parent
IMG_DIR     = _ROOT / "figures" / ("en" if EN else "pt")
IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_KERNELS = IMG_DIR / "densification_kernels.pdf"
OUT_STEPS   = IMG_DIR / "densification_steps.pdf"

# ITA template: textwidth=6.30in, body 12pt → figures at 10pt, saved at printed width.
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

# ── Kernels ────────────────────────────────────────────────────────────────────
def cross_kernel(k):
    ker = np.zeros((k, k), dtype=np.uint8)
    ker[k // 2, :] = 1
    ker[:, k // 2] = 1
    return ker

CROSS_3 = cross_kernel(3)
CROSS_5 = cross_kernel(5)
CROSS_7 = cross_kernel(7)
FULL_3  = np.ones((3, 3), dtype=np.uint8)

# ── Morfologia em numpy puro ────────────────────────────────────────────────────
def dilate_np(depth, kernel):
    H, W   = depth.shape
    kH, kW = kernel.shape
    pH, pW = kH // 2, kW // 2
    pad    = np.pad(depth, ((pH, pH), (pW, pW)), constant_values=0)
    out    = np.zeros_like(depth)
    for di in range(kH):
        for dj in range(kW):
            if kernel[di, dj]:
                out = np.maximum(out, pad[di:di+H, dj:dj+W])
    return out

def dilate_mask(mask, kernel):
    H, W   = mask.shape
    kH, kW = kernel.shape
    pH, pW = kH // 2, kW // 2
    pad    = np.pad(mask.astype(np.uint8), ((pH, pH), (pW, pW)), constant_values=0)
    out    = np.zeros_like(mask, dtype=np.uint8)
    for di in range(kH):
        for dj in range(kW):
            if kernel[di, dj]:
                out = np.maximum(out, pad[di:di+H, dj:dj+W])
    return out

def erode_np(depth, kernel):
    H, W   = depth.shape
    kH, kW = kernel.shape
    pH, pW = kH // 2, kW // 2
    pad    = np.pad(depth.astype(float), ((pH, pH), (pW, pW)), constant_values=np.inf)
    out    = np.full_like(depth, np.inf, dtype=float)
    for di in range(kH):
        for dj in range(kW):
            if kernel[di, dj]:
                out = np.minimum(out, pad[di:di+H, dj:dj+W])
    out[out == np.inf] = 0
    return out.astype(depth.dtype)

def close_np(depth, kernel):
    return erode_np(dilate_np(depth, kernel), kernel)

def dilate_fill(depth, mask, kernel):
    d_dil = dilate_np(depth, kernel)
    m_dil = dilate_mask(mask, kernel)
    fill  = (mask == 0) & (m_dil > 0)
    depth = depth.copy(); depth[fill] = d_dil[fill]
    mask  = mask.copy();  mask[fill]  = 1
    return depth, mask, fill

# ── Mapa esparso 7×7 ────────────────────────────────────────────────────────────
D0 = np.zeros((9, 9), dtype=np.float32)
D0[0, 5] = 4.0
D0[2, 7] = 8.0
D0[3, 7] = 4.0
D0[3, 8] = 4.0
D0[7, 6] = 3.0
D0[7, 7] = 7.0
M0 = (D0 > 0).astype(np.uint8)
origin_mask = M0.copy()

steps = [("$D_{\\mathrm{sparse}}$", D0.copy(), M0.copy(), np.zeros_like(M0))]
D, M = D0.copy(), M0.copy()
for label, kern in [(t("após dil. $3{\\times}3$",  "after dil. $3{\\times}3$"),  CROSS_3),
                    (t("após dil. $5{\\times}5$",  "after dil. $5{\\times}5$"),  CROSS_5),
                    (t("após dil. $7{\\times}7$",  "after dil. $7{\\times}7$"),  CROSS_7)]:
    D, M, fill = dilate_fill(D, M, kern)
    steps.append((label, D.copy(), M.copy(), fill.astype(np.uint8)))

D_closed = close_np(D, FULL_3)
fill = M == 0
D = D.copy(); D[fill] = D_closed[fill]
M = M.copy(); M[fill & (D_closed > 0)] = 1
steps.append((t("após closing $3{\\times}3$", "after closing $3{\\times}3$"),
              D.copy(), M.copy(),
              (fill & (D_closed > 0)).astype(np.uint8)))

# ── Cores ───────────────────────────────────────────────────────────────────────
C_ORIGIN  = "#ef5350"
C_NEW     = "#fff9c4"
C_OLD     = "#ffe082"
C_EMPTY   = "#eceff1"
C_KERN_ON = "#42a5f5"
C_KERN_OFF= "#e0e0e0"

# ── Helpers ─────────────────────────────────────────────────────────────────────
def cell(ax, j, i_inv, fc, ec, lw):
    ax.add_patch(patches.FancyBboxPatch(
        (j+0.06, i_inv+0.06), 0.88, 0.88,
        boxstyle="round,pad=0.04",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))

def draw_matrix(ax, data, mask, fill_now, origin, title):
    H, W = data.shape
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, pad=4)
    for i in range(H):
        for j in range(W):
            is_orig = origin[i, j] > 0
            is_new  = (fill_now is not None) and fill_now[i, j] > 0
            is_val  = mask[i, j] > 0
            fc = (C_ORIGIN if is_orig else C_NEW if is_new else
                  C_OLD    if is_val  else C_EMPTY)
            ec = "#555" if is_val else "#bbb"
            cell(ax, j, H-i-1, fc, ec, 1.4 if is_val else 0.6)
            if is_val:
                ax.text(j+0.5, H-i-0.5, f"{data[i,j]:.0f}",
                        ha="center", va="center", fontsize=8,
                        fontweight="bold" if is_orig else "normal",
                        color="#212121", zorder=3)
            else:
                ax.text(j+0.5, H-i-0.5, "·",
                        ha="center", va="center",
                        fontsize=9, color="#9e9e9e", zorder=3)

def draw_kernel(ax, kernel, title):
    H, W = kernel.shape
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, pad=4)
    for i in range(H):
        for j in range(W):
            active = kernel[i, j] > 0
            cell(ax, j, H-i-1,
                 C_KERN_ON  if active else C_KERN_OFF,
                 "#1565c0"  if active else "#bdbdbd",
                 1.2        if active else 0.6)
            ax.text(j+0.5, H-i-0.5, "1" if active else "0",
                    ha="center", va="center", fontsize=7,
                    color="#fff" if active else "#9e9e9e", zorder=3)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 1 — Kernels
# ══════════════════════════════════════════════════════════════════════════════
kernel_specs = [
    (CROSS_3, t("Cruz $3{\\times}3$\n(dilatação local)",   "Cross $3{\\times}3$\n(local dilation)")),
    (CROSS_5, t("Cruz $5{\\times}5$\n(dilatação média)",   "Cross $5{\\times}5$\n(medium dilation)")),
    (CROSS_7, t("Cruz $7{\\times}7$\n(dilatação ampla)",   "Cross $7{\\times}7$\n(wide dilation)")),
    (FULL_3,  t("Cheio $3{\\times}3$\n(closing $31{\\times}31$)",
                "Full $3{\\times}3$\n(closing $31{\\times}31$)")),
]

fig1, axes1 = plt.subplots(1, 4, figsize=(_TW, 2.4),
                            gridspec_kw={"wspace": 0.7})
for ax, (kern, title) in zip(axes1, kernel_specs):
    draw_kernel(ax, kern, title)

fig1.suptitle(t("Elementos estruturantes utilizados na densificação morfológica",
                "Structuring elements used in morphological densification"),
              y=1.02)
fig1.savefig(OUT_KERNELS, bbox_inches="tight")
plt.close(fig1)
print(t(f"Salvo em {OUT_KERNELS}", f"Saved to {OUT_KERNELS}"))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 2 — Progressão passo a passo
# ══════════════════════════════════════════════════════════════════════════════
legend_handles = [
    patches.Patch(facecolor=C_ORIGIN, edgecolor="#555", label=t("Ponto LiDAR original",    "Original LiDAR point")),
    patches.Patch(facecolor=C_NEW,    edgecolor="#555", label=t("Preenchido neste passo",   "Filled in this step")),
    patches.Patch(facecolor=C_OLD,    edgecolor="#555", label=t("Válido (passo anterior)",  "Valid (previous step)")),
    patches.Patch(facecolor=C_EMPTY,  edgecolor="#bbb", label=t("Pixel inválido (·)",       "Invalid pixel (·)")),
]

n = len(steps)
fig2, axes2 = plt.subplots(1, n, figsize=(_TW, 2.8),
                            gridspec_kw={"wspace": 0.35})
fig2.subplots_adjust(top=0.88, bottom=0.18, left=0.02, right=0.98)

for ax, (title, data, mask, fill) in zip(axes2, steps):
    draw_matrix(ax, data, mask,
                fill if title != steps[0][0] else None,
                origin_mask, title)

# Setas entre painéis
fig2.canvas.draw()
for i in range(n - 1):
    p0 = axes2[i].get_position()
    p1 = axes2[i+1].get_position()
    y  = (p0.y0 + p0.y1) / 2
    fig2.add_artist(patches.FancyArrowPatch(
        posA=(p0.x1 + 0.005, y), posB=(p1.x0 - 0.005, y),
        arrowstyle="->", mutation_scale=12,
        color="#546e7a", linewidth=1.3,
        transform=fig2.transFigure, clip_on=False))

fig2.legend(handles=legend_handles, loc="lower center",
            ncol=4, frameon=True,
            framealpha=0.95, edgecolor="#ccc",
            bbox_to_anchor=(0.5, 0.01))
fig2.suptitle(
    t("Progressão da densificação num mapa sintético $9{\\times}9$ (valores em metros)",
      "Densification progression on a synthetic $9{\\times}9$ map (values in meters)"),
    y=0.97)

fig2.savefig(OUT_STEPS, bbox_inches="tight")
plt.close(fig2)
print(t(f"Salvo em {OUT_STEPS}", f"Saved to {OUT_STEPS}"))
