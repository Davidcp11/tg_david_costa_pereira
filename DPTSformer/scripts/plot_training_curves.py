"""Gera gráficos de treinamento a partir dos logs TensorBoard."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

_parser = argparse.ArgumentParser()
_parser.add_argument("--lang", choices=["pt", "en"], default="pt")
_args, _ = _parser.parse_known_args()
EN = _args.lang == "en"


def t(pt_str, en_str):
    return en_str if EN else pt_str


LOG_DIR   = Path("checkpoints/kitti/bbox_clip30/bbox_clip30")
_ROOT     = Path(__file__).resolve().parent.parent.parent
SAVE_DIR  = _ROOT / ("tg1en" if EN else "tg1") / "Cap4" / "images"
MAX_EPOCH = 100
SAVE_DIR.mkdir(exist_ok=True, parents=True)

# ITA template: A4, left=3cm, right=2cm → textwidth = 160mm = 6.30in, body font 12pt.
# All figures saved at 0.75*textwidth so LaTeX includes them at width=0.75\textwidth
# with no scaling — fonts render at exactly 10pt.
_TW  = 6.30          # textwidth in inches
_FS  = 10            # base font size
_FIG = (0.75 * _TW, 3.0)   # (4.72in, 3.0in) — used for every figure

plt.rcParams.update({
    'font.size':        _FS,
    'axes.titlesize':   _FS,
    'axes.labelsize':   _FS,
    'xtick.labelsize':  _FS - 1,
    'ytick.labelsize':  _FS - 1,
    'legend.fontsize':  _FS - 1,
    'figure.dpi':       150,
})


def load_scalars(log_dir: Path):
    ea = event_accumulator.EventAccumulator(
        str(log_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = {"step": [e.step for e in events], "value": [e.value for e in events]}
    return data


def filter_epoch(data, tag, max_epoch=MAX_EPOCH):
    steps  = np.array(data[tag]["step"])
    values = np.array(data[tag]["value"])
    return steps[steps <= max_epoch], values[steps <= max_epoch]


def filter_step(data, tag, max_epoch=MAX_EPOCH):
    steps  = np.array(data[tag]["step"])
    values = np.array(data[tag]["value"])
    total_epochs = max(data["loss/train"]["step"])
    max_step = int(max(steps) * max_epoch / total_epochs)
    mask = steps <= max_step
    return steps[mask], values[mask]


def smooth(values, w=30):
    if len(values) < w:
        return values
    return np.convolve(values, np.ones(w) / w, mode="same")


def _step_plot(ax, data, tag, color, ylabel, title=None):
    steps, values = filter_step(data, tag)
    ax.plot(steps, values, alpha=0.2, color=color)
    ax.plot(steps, smooth(values), color=color, lw=1.5,
            label=t("suavizado", "smoothed"))
    ax.set_xlabel(t("Passo", "Step"))
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.4)
    ax.legend()


def _save(fig, fname):
    fig.tight_layout()
    fig.savefig(SAVE_DIR / f"{fname}.pdf")
    plt.close(fig)
    print(t(f"Salvo: {fname}.pdf", f"Saved: {fname}.pdf"))


# ── A1: total loss (train + val per epoch) ────────────────────────────────────

def plot_epoch_losses(data):
    fig, ax = plt.subplots(figsize=_FIG)
    e_tr, v_tr = filter_epoch(data, "loss/train")
    e_vl, v_vl = filter_epoch(data, "loss/val")
    ax.plot(e_tr, v_tr, "b-o", ms=3, label=t("Treino", "Training"))
    ax.plot(e_vl, v_vl, "r-o", ms=3, label=t("Validação", "Validation"))
    ax.set_xlabel(t("Época", "Epoch"))
    ax.set_ylabel(t("Perda", "Loss"))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.legend()
    ax.grid(True, alpha=0.4)
    _save(fig, "loss_train_val")


# ── A2: pose losses ───────────────────────────────────────────────────────────

def plot_pose_losses(data):
    configs = [
        ("train/T_loss", t("Perda de Translação", "Translation Loss"), "steelblue", "loss_translation_train"),
        ("train/R_loss", t("Perda de Rotação",    "Rotation Loss"),    "tomato",    "loss_rotation_train"),
    ]
    for tag, title, color, fname in configs:
        fig, ax = plt.subplots(figsize=_FIG)
        _step_plot(ax, data, tag, color, t("Perda", "Loss"), title)
        _save(fig, fname)


# ── A3: depth losses ──────────────────────────────────────────────────────────

def plot_depth_losses(data):
    configs = [
        ("train/ssi_mse_loss",   "SSI MSE",                           "mediumpurple",   "loss_depth_mse"),
        ("train/ssi_grad_loss",  t("SSI Gradiente", "SSI Gradient"), "mediumseagreen", "loss_depth_grad"),
        ("train/ssi_total_loss", "SSI Total",                          "darkorange",     "loss_depth_total"),
    ]
    for tag, title, color, fname in configs:
        fig, ax = plt.subplots(figsize=_FIG)
        _step_plot(ax, data, tag, color, t("Perda", "Loss"), title)
        _save(fig, fname)


# ── A4: uncertainty scales ────────────────────────────────────────────────────

def plot_uncertainty_scales(data):
    configs = [
        ("train/s_x", "$s_x$", "royalblue", "uncertainty_sx"),
        ("train/s_q", "$s_q$", "crimson",   "uncertainty_sq"),
        ("train/s_d", "$s_d$", "seagreen",  "uncertainty_sd"),
    ]
    for tag, title, color, fname in configs:
        fig, ax = plt.subplots(figsize=_FIG)
        _step_plot(ax, data, tag, color, t("Valor", "Value"), title)
        _save(fig, fname)


# ── A5: translation errors ────────────────────────────────────────────────────

def plot_translation_errors(data):
    configs = [
        ("train/E_tx", "$t_x$", "steelblue",     "error_tx"),
        ("train/E_ty", "$t_y$", "mediumseagreen", "error_ty"),
        ("train/E_tz", "$t_z$", "tomato",         "error_tz"),
    ]
    for tag, title, color, fname in configs:
        fig, ax = plt.subplots(figsize=_FIG)
        _step_plot(ax, data, tag, color, t("Erro", "Error"), title)
        _save(fig, fname)


# ── A6: rotation errors ───────────────────────────────────────────────────────

def plot_rotation_errors(data):
    configs = [
        ("train/E_rx", "$r_x$", "darkorange",   "error_rx"),
        ("train/E_ry", "$r_y$", "mediumpurple", "error_ry"),
        ("train/E_rz", "$r_z$", "deeppink",     "error_rz"),
    ]
    for tag, label, color, fname in configs:
        fig, ax = plt.subplots(figsize=_FIG)
        _step_plot(ax, data, tag, color, t("Erro", "Error"), label)
        _save(fig, fname)


if __name__ == "__main__":
    print(t(f"Carregando logs TensorBoard (até época {MAX_EPOCH})...",
            f"Loading TensorBoard logs (up to epoch {MAX_EPOCH})..."))
    data = load_scalars(LOG_DIR)

    plot_epoch_losses(data)
    plot_pose_losses(data)
    plot_depth_losses(data)
    plot_uncertainty_scales(data)
    plot_translation_errors(data)
    plot_rotation_errors(data)

    print(t(f"\nTodos os gráficos salvos em {SAVE_DIR}/",
            f"\nAll plots saved to {SAVE_DIR}/"))
