from pathlib import Path
from typing import Union
import time

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn

from utils.config_utils import load_config
from utils.checkpoint_utils import load_checkpoint
from datasets.dataloader import build_dataloader
from models.build_model import build_model, load_model_state
from utils.data_utils import (
    read_7scenes_split,
    get_scene_sequence_pairs,
    save_depth_map,
    save_predicted_depths,
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
        pred_poses = torch.zeros((1, self.window_size - 1, 6), device=self.device)
        pred_depth_list = []
        inference_times = []

        with tqdm(self.dataloader, unit="batch") as batchs:
            for batch in batchs:
                images = batch[0]
                images = images.to(self.device)

                # Start timing
                start_time = time.time()

                with torch.no_grad():
                    pred_pose, pred_depth = self.model(images.float())

                # End timing
                end_time = time.time()
                inference_times.append(end_time - start_time)

                # Convert to the previous TSformerVO shape
                pred_rot = torch.reshape(pred_pose[0], (self.window_size - 1, 3))
                pred_t = torch.reshape(pred_pose[1], (self.window_size - 1, 3))
                pred_pose = torch.cat((pred_rot, pred_t), dim=1).unsqueeze(dim=0)
                pred_poses = torch.cat((pred_poses, pred_pose), dim=0)

                # Acumula só os primeiros 15 depths na CPU para evitar OOM em seqs longas
                if len(pred_depth_list) < 15:
                    pred_depth_list.append(pred_depth.squeeze(0).cpu())

        pred_depth_list = torch.cat(pred_depth_list, dim=0)

        mean_time = sum(inference_times) / len(inference_times)
        print(f"Mean inference time per batch: {mean_time:.6f} seconds")

        # Return poses as a numpy array
        return pred_poses.cpu().detach().numpy(), pred_depth_list


def save_predictions(
    predictions: np.ndarray,
    save_dpath: Union[str, Path],
    sequence: str,
    scene: str,
) -> None:
    """Saves the predicted poses to a NumPy file.

    Args:
        predictions (np.ndarray): Predicted poses to be saved.
        save_dpath (Union[str, Path]): Directory path to save predictions
        sequence (str): Name of the sequence
        scene (str): Name of the scene (7scenes)
    """

    save_dpath = Path(save_dpath)

    # Create directory if it does not exist
    save_dpath.mkdir(parents=True, exist_ok=True)

    fname = (
        f"pred_poses_{scene}_{sequence}.npy" if scene else f"pred_poses_{sequence}.npy"
    )
    np.save(
        save_dpath / fname,
        predictions,
    )


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

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {num_params}")

    pairs = get_scene_sequence_pairs(config, config["data"].get("test_sequences", None))

    # At test time we don't need depth GT — force estimate_depth=False so the
    # dataloader doesn't filter out sequences without depth_2/ directories.
    data_params_test = {**config.get("data", {}), "estimate_depth": False}

    # Predict for each test sequence
    for scene, sequence in pairs:
        print(f"Sequence: {scene}-{sequence}" if scene else f"Sequence: {sequence}")

        # Build dataloader
        dataloader = build_dataloader(
            data_params_test,
            split="test",
            sequence=sequence,
            scene=scene,
        )

        # Create PosePredictor instance
        predictor = PosePredictor(model, dataloader, device)

        # Perform predictions
        predicted_poses, predicted_depths = predictor.predict()

        # Save predictions — use stem to avoid collision with the .pth file itself
        save_dpath = checkpoint_fpath.with_suffix("") / config["data"]["dataset"]
        save_predictions(predicted_poses, save_dpath, sequence, scene)

        # Save depths
        save_predicted_depths(
            pred_depths=predicted_depths,
            save_dpath=save_dpath,
            sequence=sequence,
            scene=scene,
            colormap="Spectral_r",
            num_to_save=15,
        )
        save_predicted_depths(
            pred_depths=predicted_depths,
            save_dpath=save_dpath,
            sequence=sequence,
            scene=scene,
            colormap="magma",
            num_to_save=15,
        )
        save_predicted_depths(
            pred_depths=predicted_depths,
            save_dpath=save_dpath,
            sequence=sequence,
            scene=scene,
            use_colormap=False,
            colormap="gray",
            num_to_save=15,
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
