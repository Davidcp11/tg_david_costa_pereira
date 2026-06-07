#!/usr/bin/env python3
"""Gera pred_traj_XX.pdf para todas as sequências de teste do Cap4.

Lê de sample_data/cap4/ — sem dependência de torch, KITTI completo ou checkpoints.

Use --lang en para labels em inglês e salvar em tg1en/.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_parser = argparse.ArgumentParser()
_parser.add_argument("--lang", choices=["pt", "en"], default="pt")
_args, _ = _parser.parse_known_args()
EN = _args.lang == "en"


def t(pt_str, en_str):
    return en_str if EN else pt_str


ROOT     = Path(__file__).resolve().parent.parent
SAMPLE   = ROOT / "data" / "cap4"
OUT_DIR  = ROOT / "figures" / ("en" if EN else "pt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 3   # igual ao config bbox_clip30
SEQUENCES   = ["01", "03", "04", "05", "06", "07", "10"]

# ITA: salvar a 0.48*6.30 = 3.02in para incluir width=0.48\textwidth sem escala.
_TW = 6.30
_FS = 10
plt.rcParams.update({
    'font.size':       _FS,
    'axes.titlesize':  _FS,
    'axes.labelsize':  _FS,
    'xtick.labelsize': _FS - 1,
    'ytick.labelsize': _FS - 1,
    'legend.fontsize': _FS - 1,
    'figure.dpi':      150,
})

# Parâmetros de normalização
with open(SAMPLE / "dataset_stats.json") as f:
    stats = json.load(f)["kitti"]
mean_angles = np.array(stats["mean_angles"])
std_angles  = np.array(stats["std_angles"])
mean_t      = np.array(stats["mean_t"])
std_t       = np.array(stats["std_t"])


def euler_to_rotation(angles):
    rx, ry, rz = angles
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0,           1, 0          ],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0,           0,           1]])
    return Rz @ Ry @ Rx


def post_processing(pred_poses, window_size=3):
    """Média de poses sobrepostas (window_size=3, overlap=2)."""
    if window_size == 2:
        return pred_poses.squeeze(1)
    from collections import deque
    q = deque(maxlen=window_size - 1)
    idx, poses = 0, []
    while len(q) < window_size - 1:
        q.append(pred_poses[idx]); idx += 1
    while idx < pred_poses.shape[0]:
        if idx == window_size - 1:
            poses.append(q[0][0])
            poses.append((q[0][1] + q[1][0]) / 2)
        elif idx < pred_poses.shape[0] - 1:
            poses.append((q[0][1] + q[1][0]) / 2)
        else:
            poses.append(q[1][1])
            idx += 1
        if idx < pred_poses.shape[0] - 1:
            idx += 1
            q.popleft()
            q.append(pred_poses[idx])
    return np.array(poses)


def recover_trajectory(poses, T_init):
    T, traj = T_init, []
    for i in range(len(poses) - 1):
        angles = poses[i, :3] * std_angles + mean_angles
        t_vec  = poses[i, 3:] * std_t      + mean_t
        rot    = euler_to_rotation(angles)
        T_r    = np.vstack([np.hstack([rot, t_vec.reshape(3, 1)]),
                            [0., 0., 0., 1.]])
        T      = T @ T_r
        traj.append(T[:3, 3])
    return np.array(traj)


def umeyama_align(src, dst):
    """Alinhamento de similaridade 7-DoF (Umeyama, com escala): rotaciona,
    escala e translada `src` para minimizar o erro quadratico contra `dst`.
    src, dst: (N, 3). Mesmo alinhamento de 7-DoF usado nas metricas (Tabela 4.1)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    cov = (Dc.T @ Sc) / src.shape[0]
    U, Dsv, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (Sc ** 2).sum() / src.shape[0]
    s = np.trace(np.diag(Dsv) @ S) / var_s
    return (s * (R @ src.T)).T + (mu_d - s * R @ mu_s)


for seq in SEQUENCES:
    pred_path = SAMPLE / f"pred_poses_{seq}.npy"
    gt_path   = SAMPLE / "gt_poses" / f"{seq}.txt"

    if not pred_path.exists():
        print(f"  [skip] {pred_path.name} não encontrado")
        continue
    if not gt_path.exists():
        print(f"  [skip] gt_poses/{seq}.txt não encontrado")
        continue

    gt_raw       = np.loadtxt(gt_path)          # (N, 12)
    gt_trajectory = gt_raw[:, [3, 7, 11]]        # tx, ty, tz
    T_init        = np.vstack([gt_raw[0].reshape(3, 4), [0., 0., 0., 1.]])

    pred_raw  = np.load(pred_path)
    poses     = post_processing(pred_raw, WINDOW_SIZE)
    pred_traj = recover_trajectory(poses, T_init)

    # Alinhamento 7-DoF (Umeyama) da trajetoria estimada ao GT — consistente
    # com as metricas. Usa os pontos correspondentes (mesmo comprimento).
    n = min(len(pred_traj), len(gt_trajectory))
    pred_traj     = umeyama_align(pred_traj[:n], gt_trajectory[:n])
    gt_trajectory = gt_trajectory[:n]

    fig, ax = plt.subplots(figsize=(0.48 * _TW, 3.0))
    ax.plot(pred_traj[:, 0],     pred_traj[:, 2],     "b",
            label=t("estimado", "estimated"))
    ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 2], "r",
            label=t("referência", "reference"))
    ax.grid(True, alpha=0.4)
    ax.set_xlabel(t("Translação em x [m]", "Translation in x [m]"))
    ax.set_ylabel(t("Translação em z [m]", "Translation in z [m]"))
    ax.set_title(t(f"Sequência {seq}", f"Sequence {seq}"))
    ax.legend()
    fig.tight_layout()

    out = OUT_DIR / f"pred_traj_{seq}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(t(f"Salvo: {out}", f"Saved: {out}"))
