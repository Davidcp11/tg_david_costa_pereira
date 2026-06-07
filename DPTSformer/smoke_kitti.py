"""Smoke test do KITTIDataset com a nova estrutura D:/TG/tgdavid/data/kitti.

Carrega seq 09 (a menor, ~1500 frames), pega 1 sample e imprime shapes.
Por enquanto sem depth (estimate_depth=False) — primeiro confirmar pipeline RGB+pose.

Roda a partir da raiz do DPTSformer/:
    python smoke_kitti.py
"""

import sys
from pathlib import Path

import numpy as np
from torchvision import transforms

# Garante que imports relativos do DPTSformer funcionem
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets.kitti import KITTIDataset


DATA_DPATH = Path("D:/TG/tgdavid/data/kitti")


def main() -> None:
    print(f"data_dpath = {DATA_DPATH}  (existe? {DATA_DPATH.is_dir()})")

    preprocess = transforms.Compose(
        [
            transforms.Resize((192, 640)),  # mesmo do __main__ original
            transforms.ToTensor(),
        ]
    )

    ds = KITTIDataset(
        data_dpath=DATA_DPATH,
        sequences=["09"],          # menor seq, mais rapido
        window_size=3,
        overlap=2,
        normalize_gt=True,
        estimate_depth=True,       # le depth_2/<frame>.npy
        transforms=preprocess,
    )
    print(f"len(dataset) = {len(ds)}")

    imgs, gt, depth = ds[0]
    print(f"imgs.shape   = {tuple(imgs.shape)}     (espera (3, 3, 192, 640))")
    print(f"depth.shape  = {tuple(depth.shape)}     (espera (3, 192, 640))")
    print(f"gt.shape     = {tuple(gt.shape)}      (espera (12,) = 2 transicoes x 6 valores)")
    print(f"gt           = {np.asarray(gt)}")

    # Itera 5 samples pra garantir que nao quebra no meio
    for i in (0, 1, 100, len(ds) // 2, len(ds) - 1):
        imgs, gt, depth = ds[i]
        print(f"  sample {i}: imgs {tuple(imgs.shape)}  depth {tuple(depth.shape)}  gt {tuple(gt.shape)}")

    print("OK — smoke test passou sem erros.")


if __name__ == "__main__":
    main()
