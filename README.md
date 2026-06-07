# Depth-Aware Transformer for Monocular Visual Odometry — Reproducibility Package

Code, scripts and minimal data to **reproduce the figures and results** of the undergraduate
final work *"Depth-Aware Transformer for Monocular Visual Odometry"* (DPTSformer applied to
KITTI Odometry), by **David Costa Pereira** (ITA).

The model (DPTSformer / TSformer-VO) is a multi-task space-time Transformer that shares an
encoder between a 6-DoF pose head and a dense depth head. This work adds, on top of it, three
depth-supervision contributions for KITTI: **(i)** masking the depth loss to the reliable
LiDAR coverage region (*bbox masking*), **(ii)** depth clipping at 30 m, and **(iii)** a
learned uncertainty weighting of the depth loss.

---

## ⚠️ Disclaimer — use of AI

Artificial-intelligence tools were used as **assistants** in producing this work:
**writing assistance** (revising and polishing the text) and **help developing the new
auxiliary code** introduced here (e.g., the figure-generation scripts and data-preparation
utilities). The methodology, the experiments, the model training, the results and their
interpretation are the **author's own work**; the AI tools were used only as a support, and
all outputs were reviewed and validated by the author.

---

## Directory structure

```
tg1david/
├── README.md
├── DPTSformer/            # model + training + inference code (no checkpoints)
│   ├── models/  datasets/  utils/  eval/  tools/  scripts/
│   └── configs/kitti/bbox_clip30.json     # configuration of the reported run
├── figure_scripts/       # self-contained scripts to regenerate every thesis figure
│   ├── plot_densification_example.py      # Fig 3.2/3.3 (synthetic)
│   ├── gen_depth_densification_en.py      # Fig 3.4  (uses data/cap3)
│   ├── gen_bbox_mask_en.py                # Fig 3.5  (uses data/cap3)
│   ├── plot_training_curves.py            # Fig 4.2–4.6 (needs the TensorBoard log)
│   └── plot_traj_en.py                    # trajectory figures (uses data/cap4)
├── data/
│   ├── cap3/             # one KITTI frame: rgb.png, depth_sparse.png, depth_dense.npy
│   ├── cap4/             # predicted + GT poses of the 7 test sequences (+ dataset_stats.json)
│   └── tensorboard/      # (place the training-run event file here — see below)
└── figures/
    ├── en/               # reference outputs, English labels
    └── pt/               # reference outputs, Portuguese labels
```

## Requirements

Figures only (lightweight, no GPU/torch):
```bash
python3 -m pip install numpy matplotlib pillow tensorboard
```
Training / inference also need PyTorch and the KITTI dataset (see `DPTSformer/environment.yml`).

---

## 1. Reproduce the figures

All figure scripts take `--lang en|pt` and write to `figures/en/` or `figures/pt/`.

**Self-contained (data already included):**
```bash
cd tg1david
python3 figure_scripts/plot_densification_example.py --lang en   # Fig 3.2 + 3.3 (synthetic 9x9)
python3 figure_scripts/gen_depth_densification_en.py  --lang en   # Fig 3.4  (data/cap3)
python3 figure_scripts/gen_bbox_mask_en.py            --lang en   # Fig 3.5  (data/cap3)
python3 figure_scripts/plot_traj_en.py                --lang en   # estimated vs GT trajectories (data/cap4)
```
- The trajectory plots are aligned to the ground truth with the **Umeyama 7-DoF (similarity)**
  transformation, consistent with the reported metrics.

**Training curves (Fig 4.2–4.6)** need the TensorBoard event file of the 100-epoch run, which
is large and not shipped here. Place the run directory under `data/tensorboard/` (so that the
event file sits at `data/tensorboard/events.out.tfevents.*`) and run:
```bash
python3 figure_scripts/plot_training_curves.py --lang en
```

Reference outputs for every figure are already in `figures/en/` and `figures/pt/`.

---

## 2. Training

Training uses KITTI Odometry (color images + LiDAR depth, densified — see
`DPTSformer/scripts` and `tools/`) and the configuration `DPTSformer/configs/kitti/bbox_clip30.json`
(split: train 00/02/08/09, test 01/03–07/10; `image_size` 192×640; depth clip 30 m; learnable
pose+depth uncertainty weighting).

```bash
cd DPTSformer
python3 scripts/train.py --config configs/kitti/bbox_clip30.json
```
This produces TensorBoard logs (used by `plot_training_curves.py`) and checkpoints.

## 3. Inference / evaluation

```bash
cd DPTSformer
python3 scripts/test.py  --config configs/kitti/bbox_clip30.json   # predicts the test sequences
python3 eval/run_kitti_eval.py                                     # KITTI metrics, 7-DoF alignment
```
`run_kitti_eval.py` reports `t_rel`, `r_rel`, ATE and RPE after a 7-DoF (Umeyama) alignment, for
the test sequences. The predicted poses of the reported run are also provided in `data/cap4/`,
so the trajectory figures can be reproduced without re-running inference.

---

## Data provided

- `data/cap3/` — a single KITTI frame (sequence 00): RGB, sparse projected LiDAR depth and the
  densified depth map, used by the depth-densification and bbox figures.
- `data/cap4/` — predicted poses (`pred_poses_*.npy`) and ground-truth poses (`gt_poses/*.txt`)
  of the seven test sequences, plus `dataset_stats.json` (normalization statistics), used by the
  trajectory figures.

## Credits

- Base architecture (TSformer-VO / DPTSformer): A. O. Françani and M. R. O. A. Maximo.
- Dataset: KITTI Odometry (Geiger *et al.*).
- Depth densification inspired by the IP-Basic pipeline (Ku *et al.*, 2018).
