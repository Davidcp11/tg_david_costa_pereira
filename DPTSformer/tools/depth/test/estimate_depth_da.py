from pathlib import Path
from typing import Union
import random
import json

import torch
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.depth import load_depth_model
from utils.config_utils import load_config
from datasets.transforms import get_transforms
from utils.data_utils import (
    get_scene_sequence_pairs,
)

random.seed(2)


def preprocess_depth(
    pred: torch.Tensor,
    gt: torch.Tensor,
    min_depth_eval: float = 1e-3,
    max_depth_eval: float = 10.0,
):
    """
    Preprocesses predicted and ground truth depth maps for evaluation using PyTorch tensors.
    Args:
        pred (torch.Tensor): Predicted depth (any shape, typically (1, H, W))
        gt (torch.Tensor): Ground truth depth (same shape as pred)
        min_depth_eval (float): Minimum depth to consider valid
        max_depth_eval (float): Maximum depth to consider valid

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            (processed_pred, processed_gt, valid_mask)
    """
    pred = torch.from_numpy(pred)
    pred = pred.detach().cpu()
    gt = gt.squeeze().detach().cpu()

    # Replace invalid values in prediction
    pred = torch.nan_to_num(
        pred, nan=min_depth_eval, posinf=max_depth_eval, neginf=min_depth_eval
    )
    pred = torch.clamp(pred, min=min_depth_eval, max=max_depth_eval)

    # Compute valid mask based on ground truth
    valid_mask = (gt > min_depth_eval) & (gt < max_depth_eval) & torch.isfinite(gt)
    pred = pred[valid_mask]
    gt = gt[valid_mask]

    # divide by max value
    pred = pred / pred.max()
    gt = gt / gt.max()

    return pred, gt


def read_img_depth_pair(frame_fpath, img_size, transforms_depth):
    # read img
    img = cv2.imread(str(frame_fpath))
    # img = cv2.resize(img, img_size[::-1], interpolation=cv2.INTER_LINEAR)

    # Read depth
    depth_fpath = Path(str(frame_fpath).replace(".color", ".depth"))
    depth = Image.open(depth_fpath)
    depth = transforms_depth(depth)
    # depth[depth == 65535] = -1  # invalid depth value
    depth[depth > 40000] = -1  # invalid depth value
    depth = torch.div(depth, 1000.0)  # depth values in meters

    return img, depth


def save_depth_map(
    output_path: Union[str, Path],
    filename: str,
    depth_tensor: torch.Tensor,
    use_colormap: bool = True,
    colormap: str = "Spectral_r",
) -> str:
    """
    Saves a depth map tensor as an image, with optional colormap.

    Args:
        output_path (Union[str, Path]): Directory where the image will be saved.
        filename (str): Name of the image file (without extension).
        depth_tensor (torch.Tensor): Depth map tensor (H, W) in meters.
        use_colormap (bool): Whether to apply a colormap to the depth map.
        colormap (str): Matplotlib colormap name (e.g., 'Spectral_r', 'magma_r').

    Returns:
        str: Full path to the saved image.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    depth_np = depth_tensor.cpu().numpy()
    invalid_mask = depth_np <= 0

    # Normalize valid values
    valid_mask = ~invalid_mask
    norm_depth = np.zeros_like(depth_np)
    if np.any(valid_mask):
        vmin = depth_np[valid_mask].min()
        vmax = depth_np[valid_mask].max()
        norm_depth[valid_mask] = (depth_np[valid_mask] - vmin) / (vmax - vmin + 1e-6)

    # Set invalid to 0 (black)
    norm_depth[invalid_mask] = 0.0

    plt.figure(figsize=(6, 4))
    plt.axis("off")
    if use_colormap:
        plt.imshow(norm_depth, cmap=colormap)
    else:
        plt.imshow(norm_depth, cmap="gray")
    plt.tight_layout(pad=0)

    save_path = output_path / f"{filename}.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main(config_fpath):

    data_dpath = "C:/workspace/data/7scenes"

    # Load hyperparameters
    config = load_config(config_fpath)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    image_size = config["data"]["image_size"]
    image_size = ""
    # _, transforms_depth = get_transforms(
    #     dataset="7scenes", image_size=image_size, dataset_name_norm="7scenes"
    # )
    transforms_depth = transforms.Compose([transforms.ToTensor()])

    # Build the model
    print("Building model...")
    model_dpath = Path("C:/workspace/projects/ExpTsformerVO/checkpoints/depth-anything")
    depth_encoder = "vitb"
    model = load_depth_model(model_dpath, encoder=depth_encoder)

    pairs = get_scene_sequence_pairs(config, config["data"].get("test_sequences", None))

    index_dict = {}

    # Predict for each test sequence
    for scene, sequence in pairs:

        print(f"Sequence: {scene}-{sequence}" if scene else f"Sequence: {sequence}")

        frames_dpath = Path(data_dpath) / scene / f"seq-{sequence}"
        frames_fnames = sorted([fpath for fpath in frames_dpath.glob("*.color.png")])[
            :-2
        ]
        rdn_idx = random.sample(range(len(frames_fnames)), 10)
        frames_fnames = [frames_fnames[i] for i in rdn_idx]

        key = f"{scene}_seq{sequence}"
        index_dict[key] = rdn_idx
        # frames_fnames = random.sample(frames_fnames, 10)

        for frame_fpath in tqdm(frames_fnames):
            img, gt_depth = read_img_depth_pair(
                frame_fpath, image_size, transforms_depth
            )

            # Estimate depth
            pred_depth_orig = model.infer_image(img)

            # Considering only first frame in window
            _, gt_depth = preprocess_depth(
                pred_depth_orig, gt_depth, min_depth_eval=1e-3, max_depth_eval=15.0
            )

            save_depth_map(
                output_path="tmp/DA/",
                filename=frame_fpath.stem + f"_{scene}_{sequence}_DA",
                depth_tensor=torch.from_numpy(pred_depth_orig),
                use_colormap=True,
                colormap="magma",
            )

    with open("tmp/sampled_indices.json", "w") as f:
        json.dump(index_dict, f, indent=4)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: python -m testing.inference_vbr <config_fpath> <checkpoint_fname>"
        )
        sys.exit(1)

    config_fpath = sys.argv[1]

    # Run main processing
    main(config_fpath)
