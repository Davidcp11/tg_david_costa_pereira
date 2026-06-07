"""Standalone KITTI odometry evaluation — bypasses torch dependency in eval_odom.py."""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.kitti_odometry import KittiEvalOdom

PRED_DIR  = "checkpoints/kitti/bbox_clip30/bbox_clip30/checkpoint_best/kitti/pred_poses"
GT_DIR    = "D:/TG/tgdavid/data/kitti/poses"
OUT_FILE  = "results/metrics_bbox_clip30.txt"
SEQS      = ["01", "03", "04", "05", "06", "07", "10"]
ALIGN     = "7dof"

eval_tool = KittiEvalOdom()
eval_tool.eval(GT_DIR, PRED_DIR, alignment=ALIGN, seqs=SEQS)

# Copy result.txt generated inside pred_poses/ to results/
src = Path(PRED_DIR) / "result.txt"
dst = Path(OUT_FILE)
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(src, dst)
print(f"\nMétricas salvas em: {dst}")
