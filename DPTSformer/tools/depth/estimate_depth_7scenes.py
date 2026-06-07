from pathlib import Path
from typing import Union

import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
from datasets.dataloader import build_dataloader

import matplotlib.pyplot as plt
from utils.config_utils import load_config
from utils.checkpoint_utils import load_checkpoint
from datasets.transforms import get_transforms
from models.build_model import build_model, load_model_state
from utils.data_utils import (
    get_scene_sequence_pairs,
)
import torch.nn as nn


class DepthPredictor:
    """Class for predicting depths using a Vision Transformer model."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: Union[torch.device, str],
    ):
        """
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

                # # Concatenate depths
                # pred_depth_list.append(
                #     pred_depth[0, 0, :, :].unsqueeze(0)
                # )  # (pred_depth.squeeze(0))
                # gt_depths.append(
                #     (depths[0, 0, :, :].unsqueeze(0))
                # )  # depths.squeeze(0))

                # frame_dpath = self.dataloader.dataset.windowed_dict[batch_idx][
                #     "frames"
                # ][0]
                # frames_list.append(
                #     f"{frame_dpath.parents[1].name}-{frame_dpath.parent.name}-{frame_dpath.stem}"
                # )
                # batch_idx += 1

                # if batch_idx > 60:
                #     break

        pred_depth_list = torch.cat(pred_depth_list, dim=0)
        gt_depths = torch.cat(gt_depths, dim=0)

        return pred_depth_list, gt_depths, frames_list


# def save_depth_map(
#     output_path: Union[str, Path],
#     filename: str,
#     depth_tensor: torch.Tensor,
#     use_colormap: bool = True,
#     colormap: str = "Spectral_r",
# ) -> str:
#     """
#     Saves a depth map tensor as an image.

#     Args:
#         output_path (Union[str, Path]): Directory where the image will be saved.
#         filename (str): Name of the image file (without extension).
#         depth_tensor (torch.Tensor): Depth map tensor (H, W).
#         use_colormap (bool): Whether to apply a colormap for visualization (default: True).
#         colormap (str): Matplotlib colormap name. [Ex: Spectral_r, magma].

#     Returns:
#         str: Full path of the saved image.
#     """
#     output_dir = Path(output_path)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     depth_numpy = depth_tensor.cpu().numpy().squeeze()
#     image_path = output_dir / f"{filename}.png"

#     if use_colormap:
#         plt.imshow(depth_numpy, cmap=colormap)
#         # plt.colorbar()
#         plt.axis("off")
#         plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
#         plt.close()
#     else:
#         depth_normalized = cv2.normalize(
#             depth_numpy, None, 0, 255, cv2.NORM_MINMAX
#         ).astype(np.uint8)
#         cv2.imwrite(str(image_path), depth_normalized)

#     return str(image_path)


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
    save_dpath = Path(save_dpath) / "pred_depths_comp_DA"
    save_dpath.mkdir(parents=True, exist_ok=True)

    N = pred_depths.shape[0]
    M = num_to_save if num_to_save is not None else N
    M = min(M, N)  # prevent out-of-bounds

    random_idx = torch.randperm(N)[:M].tolist()

    for i in random_idx:
        j = 0
        while j <= 3:
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
            i += 1
            j += 1


if __name__ == "__main__":

    config_fpath = "configs/7scenes/exp_01.json"
    checkpoint_name = "checkpoint_e182"
    num_to_save = 15

    # Load hyperparameters
    config = load_config(config_fpath)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Load checkpoint
    data_params = config.get("data", {})
    checkpoint_params = config.get("checkpoint", {})
    checkpoint_params["checkpoint_name"] = checkpoint_name
    checkpoint_fpath = (
        Path(checkpoint_params["checkpoint_dpath"])
        / checkpoint_params["checkpoint_name"]
    )
    checkpoint = load_checkpoint(checkpoint_fpath)
    save_dpath = checkpoint_fpath / config["data"]["dataset"]

    # Build the model
    print("Building model...")
    model = build_model(config.get("model", {}), device)
    model = load_model_state(model, checkpoint)
    model = model.eval()

    pairs = get_scene_sequence_pairs(config, config["data"].get("test_sequences", None))

    # Get transforms based on the dataset
    rgb_transforms, depth_transforms = get_transforms(
        data_params["dataset"],
        data_params["image_size"],
        data_params.get("dataset_name_norm", "7scenes"),
    )

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

        # Create DepthPredictor instance
        predictor = DepthPredictor(model, dataloader, device)

        # Perform predictions
        pred_depths, gt_depths, frames_list = predictor.predict()

        # inverse depth maps
        # pred_depths = 1.0 / (pred_depths + 1e-6)
        # gt_depths = 1.0 / (gt_depths + 1e-6)

        # Save predictions
        save_dpath = checkpoint_fpath / config["data"]["dataset"]

        # Save depths
        # save_predicted_depths(
        #     pred_depths=pred_depths,
        #     gt_depths=gt_depths,
        #     frames_list=frames_list,
        #     save_dpath=save_dpath,
        #     sequence=sequence,
        #     scene=scene,
        #     colormap="Spectral",
        #     num_to_save=num_to_save,
        # )
        save_predicted_depths(
            pred_depths=pred_depths,
            gt_depths=gt_depths,
            frames_list=frames_list,
            save_dpath=save_dpath,
            sequence=sequence,
            scene=scene,
            colormap="magma_r",
            num_to_save=num_to_save,
        )
        # save_predicted_depths(
        #     pred_depths=pred_depths,
        #     gt_depths=gt_depths,
        #     frames_list=frames_list,
        #     save_dpath=save_dpath,
        #     sequence=sequence,
        #     scene=scene,
        #     use_colormap=True,
        #     colormap="gray_r",
        #     num_to_save=num_to_save,
        # )

        # frames_dpath = Path(data_params["data_dpath"]) / scene / f"seq-{sequence}"
        # frame_files = sorted(frames_dpath.glob("*.png"))
        # frame_files = random.sample(frame_files, 10)

        # # Process each frame
        # for frame_fpath in tqdm(frame_files, desc=f"Seq.{sequence}"):
        #     # Load and preprocess the image
        #     img = Image.open(frame_fpath).convert("RGB")
        #     img = rgb_transforms(img)
        #     img = img.unsqueeze(0).unsqueeze(0)

        #     # Read ground truth depth
        #     depth_fpath = Path(str(frame_fpath).replace(".color", ".depth"))
        #     gt_depth = Image.open(depth_fpath)
        #     gt_depth = depth_transforms(gt_depth)
        #     gt_depth[gt_depth > 40000] = -1  # invalid depth value
        #     gt_depth = torch.div(gt_depth, 1000.0)  # depth values in meters
        # gt_depth = 1.0/gt_depth

        # # predict depth
        # with torch.no_grad():
        #     _, pred_depth = model(img.float())

        # # save ground truth depth
        # fname = save_dpath / scene / f"seq-{sequence}" / depth_fpath.stem + "_gt"
        # save_depth_map(
        #     output_path=save_dpath / cmap_name,
        #     filename=fname,
        #     depth_tensor=gt_depth,
        #     use_colormap=True,
        #     colormap=cmap_name,
        # )

        # # save predicted depth
        # fname = save_dpath / scene / f"seq-{sequence}" / depth_fpath.stem + "_pred"
        # save_depth_map(
        #     output_path=save_dpath / cmap_name,
        #     filename=fname,
        #     depth_tensor=pred_depth,
        #     use_colormap=True,
        #     colormap=cmap_name,
        # )
