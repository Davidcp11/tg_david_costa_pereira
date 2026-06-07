##########################################################################################
## Functions used to compute and visualize the learnt Space-Time attention in TimeSformer
## [source] https://github.com/yiyixuxu/TimeSformer-rolled-attention
## Modified by: André Françani, 2024
##########################################################################################

from pathlib import Path
from typing import Union, Tuple, List

from tqdm import tqdm
import torch
from torch import einsum, nn, Tensor
import cv2
import numpy as np
from torch import einsum
from einops import rearrange, repeat

from utils.checkpoint_utils import load_checkpoint
from utils.config_utils import load_config
from utils.data_utils import load_rgb_normalization
from models.build_model import build_model, load_model_state
from datasets.dataloader import build_dataloader


def combine_divided_attention(
    attn_t: Tensor, attn_s: Tensor, window_size: int
) -> Tensor:
    """
    Combines time and space attention maps into a single attention map.

    Args:
        attn_t (Tensor): Time attention tensor with shape [batch, heads, tokens, tokens].
        attn_s (Tensor): Space attention tensor with shape [batch, heads, tokens, tokens].
        window_size (int): Size of the window for temporal repetition.

    Returns:
        Tensor: Combined attention tensor with shape [frames, tokens, tokens, tokens].
    """
    # Average time attention across heads, add identity matrix, and normalize
    attn_t = attn_t.mean(dim=1)
    I = torch.eye(attn_t.size(-1)).unsqueeze(0)
    # Adding identity matrix to account for skipped connection
    attn_t = torch.cat([I, attn_t], 0) + torch.eye(attn_t.size(-1))[None, ...]
    attn_t = attn_t / attn_t.sum(-1)[..., None]

    # Average space attention across heads, add identity matrix, and normalize
    attn_s = attn_s.mean(dim=1) + torch.eye(attn_s.size(-1))[None, ...]
    attn_s = attn_s / attn_s.sum(-1)[..., None]

    # Combine space and time attention tensors
    attn_ts = einsum("tpk, ktq -> ptkq", attn_s, attn_t)

    # Average cls_token attention across frames and repeat across time dimension
    attn_cls = attn_ts[0, :, :, :].mean(dim=0)
    attn_cls = repeat(attn_cls, "p t -> j p t", j=window_size)

    attn_ts = torch.cat([attn_cls.unsqueeze(0), attn_ts[1:, :, :, :]], 0)
    return attn_ts


class DividedAttentionRollout:
    """
    Class to perform attention rollout for visualizing attention maps in the model.

    Args:
        model (nn.Module): Model to attach attention hooks to.
        window_size (int): Size of the window for repeating temporal attention.
        patch_size (int): Patch size
    """

    def __init__(self, model: nn.Module, window_size: int, patch_size: int) -> None:
        self.model = model
        self.window_size = window_size
        self.patch_size = patch_size
        self.hooks = []
        self.time_attentions = []
        self.space_attentions = []

    def get_attn_t(self, module: nn.Module, input: Tuple, output: Tensor) -> None:
        """Saves time attention output from the model."""
        self.time_attentions.append(output.detach().cpu())

    def get_attn_s(self, module: nn.Module, input: Tuple, output: Tensor) -> None:
        """Saves space attention output from the model."""
        self.space_attentions.append(output.detach().cpu())

    def remove_hooks(self) -> None:
        """Removes all hooks attached to the model."""
        for h in self.hooks:
            h.remove()

    def __call__(self, input_tensor: Tensor) -> np.ndarray:
        """
        Generates the attention rollout mask for the given input.

        Args:
            input_tensor (Tensor): Input tensor to pass through the model.

        Returns:
            np.ndarray: Attention rollout mask.
        """
        # Set model to evaluation mode and clear previous attentions
        self.model.eval()
        self.time_attentions.clear()
        self.space_attentions.clear()

        # Register hooks for capturing attention outputs
        self.hooks = [
            m.register_forward_hook(
                self.get_attn_t
                if "temporal_attn.attn_drop" in name
                else self.get_attn_s
            )
            for name, m in self.model.named_modules()
            if "attn.attn_drop" in name
        ]

        # Run model inference without computing gradients
        with torch.no_grad():
            _ = self.model(input_tensor)

        # Remove hooks after inference
        self.remove_hooks()

        # Combine time and space attentions for each attention layer
        attentions = [
            combine_divided_attention(attn_t, attn_s, self.window_size)
            for attn_t, attn_s in zip(self.time_attentions, self.space_attentions)
        ]

        # Initialize result mask with identity for the rollout calculation
        p, t = attentions[0].shape[0], attentions[0].shape[1]
        result = torch.eye(p * t)

        # Accumulate attention maps across layers
        for attention in attentions:
            attention = rearrange(attention, "p1 t1 p2 t2 -> (p1 t1) (p2 t2)")
            result = torch.matmul(attention, result)

        # Extract and reshape the final mask
        mask = rearrange(result, "(p1 t1) (p2 t2) -> p1 t1 p2 t2", p1=p, p2=p).mean(
            dim=1
        )[0, 1:, :]
        width = int(
            input_tensor.shape[-1] / self.patch_size
        )  # int(mask.size(0) ** 0.5)
        mask = rearrange(mask, "(h w) t -> h w t", w=width).numpy()
        mask = mask / np.max(mask)
        return mask


def show_mask_on_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Overlay the attention mask on an image as a heatmap.

    Args:
        img (np.ndarray): Original image in RGB format.
        mask (np.ndarray): Attention mask.

    Returns:
        np.ndarray: Image with overlaid heatmap.
    """
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam


def create_masks(
    masks_in: List[np.ndarray], np_imgs: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Resizes masks to image dimensions and overlays them.

    Args:
        masks_in (List[np.ndarray]): List of attention masks.
        np_imgs (List[np.ndarray]): List of images corresponding to each mask.

    Returns:
        List[np.ndarray]: List of images with overlaid masks.
    """
    masks = []
    for mask, img in zip(masks_in, np_imgs):
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask = show_mask_on_image(img, mask)
        masks.append(mask)
    return masks


def post_processing_rgb(img: np.ndarray, dataset_name: str) -> np.ndarray:
    """
    Post-processes RGB images for visualization.

    Args:
        img (np.ndarray): Image tensor in RGB format.
        dataset_name (str): Name of the dataset for normalization values.

    Returns:
        np.ndarray: Post-processed image in BGR format for OpenCV display.
    """
    mean, std = load_rgb_normalization("datasets/dataset_stats.json", dataset_name)
    img = ((img * std + mean) * 255).astype(int)[..., ::-1]
    return img


def save_attention_maps(
    att_maps: np.ndarray, save_dpath: Union[Path, str], save_fname: str
) -> None:
    """
    Saves the final attention maps as an image.

    Args:
        att_maps (np.ndarray): Combined attention maps.
        save_dpath (Union[Path, str]): Directory path to save the image.
        save_fname (str): Name of the saved file.
    """
    # Create directory if it does not exist
    save_dpath = Path(save_dpath)
    save_dpath.mkdir(parents=True, exist_ok=True)

    # Write attention maps
    save_fpath = save_dpath / save_fname
    cv2.imwrite(str(save_fpath), att_maps)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(
            "Usage: python -m tools.visualize_attention_map <config_fpath> <checkpoint_fname>"
        )
        sys.exit(1)
    config_fpath = sys.argv[1]
    checkpoint_name = sys.argv[2]

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

    # Predict for each test sequence
    sequences = config["data"]["test_sequences"]
    for sequence in sequences[:1]:
        print(f"Sequence: {sequence}")

        # Build dataloader
        dataloader = build_dataloader(
            config.get("data", {}), split="test", sequence=sequence
        )

        # Generate attention maps
        for batch_idx, (images, _) in enumerate(
            tqdm(dataloader, desc=f"Sequence {sequence}", unit="batch")
        ):
            images = images.to(device)

            # Apply attention rollout and generate masks
            attention_rollout = DividedAttentionRollout(
                model,
                window_size=config["data"]["window_size"],
                patch_size=config["model"]["patch_size"],
            )

            masks = attention_rollout(images.float())

            # Post-process images and create attention overlays
            images = images.squeeze(0).cpu().numpy().transpose(1, 2, 3, 0)
            np_imgs = [
                post_processing_rgb(images[i, :, :, :], config["data"]["dataset"])
                for i in range(images.shape[0])
            ]

            # Create masks
            masks = create_masks(list(rearrange(masks, "h w t -> t h w")), np_imgs)
            attention_maps = np.vstack([np.hstack(np_imgs), np.hstack(masks)])

            # Save the attention maps
            att_save_dir = checkpoint_fpath / "attn_maps" / sequence
            save_attention_maps(
                attention_maps, att_save_dir, f"batch_{batch_idx:04d}.jpg"
            )

            # Stop after 100 batches
            if batch_idx == 100:
                break
