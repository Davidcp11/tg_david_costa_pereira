from pathlib import Path
from typing import Union
import random
import json
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from utils.config_utils import load_config
from utils.checkpoint_utils import load_checkpoint
from datasets.dataloader import build_dataloader
from models.build_model import build_model, load_model_state
from utils.data_utils import (
    get_scene_sequence_pairs,
)

random.seed(2)


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


def save_predicted_depths(
    pred_depths: torch.Tensor,
    gt_depths: torch.Tensor,
    frames_list: list,
    save_dpath: Union[str, Path],
    sequence: str,
    scene: str,
    use_colormap: bool = True,
    colormap: str = "Spectral_r",
    num_to_save: int = None,
) -> None:
    """
    Saves the first M predicted depth maps from a tensor.

    Args:
        pred_depths (torch.Tensor): Tensor of shape (N, H, W) with predicted depths.
        save_dpath (Union[str, Path]): Directory to save images.
        sequence (str): Sequence identifier.
        scene (str): Scene identifier (e.g., for 7Scenes).
        use_colormap (bool): Whether to apply a colormap.
        colormap (str): Name of the matplotlib colormap.
        num_to_save (int, optional): Number of depth maps to save. If None, saves all.
    """
    save_dpath = Path("tmp/DPTSformer")
    save_dpath.mkdir(parents=True, exist_ok=True)
    pred_depths = pred_depths[::3]
    gt_depths = gt_depths[::3]
    frames_list = frames_list[::3]

    # random_idx = random.sample([i for i in range(998)], 10)
    with open("tmp/sampled_indices.json") as f:
        index_dict = json.load(f)

    key = f"{scene}_seq{sequence}"
    rdn_idx = index_dict[key]

    for i in rdn_idx:
        fname = frames_list[i] + "_pred"
        save_depth_map(
            output_path=save_dpath / colormap,
            filename=fname,
            depth_tensor=pred_depths[i],
            use_colormap=use_colormap,
            colormap=colormap,
        )

        fname = frames_list[i] + "_gt"
        save_depth_map(
            output_path=save_dpath / colormap,
            filename=fname,
            depth_tensor=gt_depths[i],
            use_colormap=use_colormap,
            colormap=colormap,
        )


class PosePredictor:
    """Class for predicting poses using a Vision Transformer model."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: Union[torch.device, str],
    ):
        """
        Initializes the PosePredictor with the provided model and dataloader.

        Args:
            model (nn.Module): The model to use for pose prediction.
            dataloader (torch.utils.data.DataLoader): The DataLoader for loading data.
            device (Union[torch.device, str]): Either a CPU or a CUDA device.
        """
        self.device = device
        self.model = model.to(self.device)
        self.model = self.model.eval()
        self.dataloader = dataloader
        self.window_size = dataloader.dataset.window_size

    def predict(self) -> np.ndarray:
        """Runs the pose prediction over the dataset.

        Returns:
            np.ndarray: Array of predicted poses.
        """
        pred_depth_list = []
        gt_depths = []
        batch_idx = 0
        frames_list = []
        with tqdm(self.dataloader, unit="batch") as batchs:
            for images, _, depths in batchs:
                images = images.to(self.device)

                with torch.no_grad():
                    _, pred_depth = self.model(images.float())

                # Concatenate depths
                pred_depth_list.append(pred_depth.squeeze(0))
                gt_depths.append(depths.squeeze(0))

                for i in range(pred_depth.shape[1]):
                    frame_dpath = self.dataloader.dataset.windowed_dict[batch_idx][
                        "frames"
                    ][i]
                    frames_list.append(
                        f"{frame_dpath.parents[1].name}-{frame_dpath.parent.name}-{frame_dpath.stem}"
                    )
                batch_idx += 1

        pred_depth_list = torch.cat(pred_depth_list, dim=0)
        gt_depths = torch.cat(gt_depths, dim=0)

        return pred_depth_list, gt_depths, frames_list


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
    pred = pred.squeeze().detach().cpu()
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


def main(config_fpath, checkpoint_name):

    # Load hyperparameters
    config = load_config(config_fpath)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Load checkpoint
    checkpoint_params = config.get("checkpoint", {})
    checkpoint_params["checkpoint_name"] = checkpoint_name
    checkpoint_fpath = (
        Path(checkpoint_params["checkpoint_dpath"])
        / checkpoint_params["checkpoint_name"]
    )
    checkpoint = load_checkpoint(checkpoint_fpath)

    # Build the model
    print("Building model...")
    model = build_model(config.get("model", {}), device)
    model = load_model_state(model, checkpoint)

    pairs = get_scene_sequence_pairs(config, config["data"].get("test_sequences", None))

    # Predict for each test sequence
    for scene, sequence in pairs:

        print(f"Sequence: {scene}-{sequence}" if scene else f"Sequence: {sequence}")

        # Build dataloader
        dataloader = build_dataloader(
            config.get("data", {}),
            split="test",
            sequence=sequence,
            scene=scene,
        )

        # Create PosePredictor instance
        predictor = PosePredictor(model, dataloader, device)

        # Perform predictions
        pred_depths, gt_depths, frames_list = predictor.predict()

        save_predicted_depths(
            pred_depths=pred_depths,
            gt_depths=gt_depths,
            frames_list=frames_list,
            save_dpath="save_dpath",
            sequence=sequence,
            scene=scene,
            colormap="magma_r",
            num_to_save=10,
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(
            "Usage: python -m testing.inference_vbr <config_fpath> <checkpoint_fname>"
        )
        sys.exit(1)

    config_fpath = sys.argv[1]
    checkpoint_name = sys.argv[2]

    # Run main processing
    main(config_fpath, checkpoint_name)
