# sample_data/cap4

Dados mínimos para regenerar as figuras de trajetória do Cap4 sem o KITTI completo nem checkpoints.

| Arquivo | Origem | Usado por |
|---------|--------|-----------|
| `pred_poses_XX.npy` | Inferência do DPTSformer (checkpoint_best, 100 épocas) | plot_traj_en.py |
| `gt_poses/XX.txt` | KITTI Odometry Ground Truth poses | plot_traj_en.py |
| `dataset_stats.json` | `DPTSformer/datasets/dataset_stats.json` | plot_traj_en.py |

Sequências de teste: 01, 03, 04, 05, 06, 07, 10.

## Regenerar figuras

```bash
# PT-BR → tg1/Cap4/images/
python scripts/plot_traj_en.py --lang pt

# EN → tg1en/Cap4/images/
python scripts/plot_traj_en.py --lang en
```

Sem dependência de torch, KITTI completo ou modelo treinado.
