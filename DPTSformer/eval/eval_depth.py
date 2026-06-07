from pathlib import Path
from typing import Union

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.distributed as dist

from utils.config_utils import load_config
from utils.checkpoint_utils import load_checkpoint
from datasets.dataloader import build_dataloader
from models.build_model import build_model, load_model_state
from utils.data_utils import (
    get_scene_sequence_pairs,
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

    def predict(self, results, nsamples) -> np.ndarray:
        """Runs the pose prediction over the dataset.

        Returns:
            np.ndarray: Array of predicted poses.
        """

        with tqdm(self.dataloader, unit="batch") as batchs:
            for images, _, gt_depth in batchs:
                images, gt_depth = images.to(self.device), gt_depth.to(self.device)

                with torch.no_grad():
                    _, pred_depth = self.model(images.float())

                # Considering only first frame in window
                pred_depth, gt_depth = preprocess_depth(
                    pred_depth, gt_depth, min_depth_eval=1e-3, max_depth_eval=40.0
                )

                cur_results = eval_depth(pred_depth, gt_depth)

                for k in results.keys():
                    results[k] += cur_results[k]
                nsamples += 1

                # # Concatenate depths [B x H x W]
                # pred_depth_list.append(pred_depth.squeeze(0))

        return results, nsamples


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


def eval_depth(pred, target):
    """
    [SOURCE]: https://github.com/DepthAnything/Depth-Anything-V2
    """
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25**2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25**3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / (target))
    sq_rel = torch.mean(torch.pow(diff, 2) / (target))

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log, 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(
        torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2)
    )

    return {
        "d1": d1.item(),
        "d2": d2.item(),
        "d3": d3.item(),
        "abs_rel": abs_rel.item(),
        "sq_rel": sq_rel.item(),
        "rmse": rmse.item(),
        "rmse_log": rmse_log.item(),
        "log10": log10.item(),
        "silog": silog.item(),
    }


def print_metrics(scene, results, nsamples):
    print(f"\n Scene: {scene}")
    print(f"# Samples: {int(nsamples.item())}")
    print(
        "{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}".format(
            *tuple(results.keys())
        )
    )
    print(
        "{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}".format(
            *tuple([(v / nsamples).item() for v in results.values()])
        )
    )
    print()


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

    results = {
        "d1": torch.tensor([0.0]).cuda(),
        "d2": torch.tensor([0.0]).cuda(),
        "d3": torch.tensor([0.0]).cuda(),
        "abs_rel": torch.tensor([0.0]).cuda(),
        "sq_rel": torch.tensor([0.0]).cuda(),
        "rmse": torch.tensor([0.0]).cuda(),
        "rmse_log": torch.tensor([0.0]).cuda(),
        "log10": torch.tensor([0.0]).cuda(),
        "silog": torch.tensor([0.0]).cuda(),
    }
    nsamples = torch.tensor([0.0]).cuda()
    scene_prev = None
    # Predict for each test sequence
    for scene, sequence in pairs:

        if (scene != scene_prev) and (scene_prev is not None):
            print_metrics(scene_prev, results, nsamples)
            results = {
                "d1": torch.tensor([0.0]).cuda(),
                "d2": torch.tensor([0.0]).cuda(),
                "d3": torch.tensor([0.0]).cuda(),
                "abs_rel": torch.tensor([0.0]).cuda(),
                "sq_rel": torch.tensor([0.0]).cuda(),
                "rmse": torch.tensor([0.0]).cuda(),
                "rmse_log": torch.tensor([0.0]).cuda(),
                "log10": torch.tensor([0.0]).cuda(),
                "silog": torch.tensor([0.0]).cuda(),
            }
            nsamples = torch.tensor([0.0]).cuda()

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
        results, nsamples = predictor.predict(results, nsamples)

        scene_prev = scene

    print_metrics(scene, results, nsamples)


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
