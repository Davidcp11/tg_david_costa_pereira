from typing import List, Union
from pathlib import Path
import torch
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

from models.depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}


def load_depth_model(model_dpath: Path, encoder: str) -> torch.nn.Module:
    """
    Loads the pre-trained DepthAnything v2 model.

    Args:
        model_dpath (Path): Path to the pre-trained model weights.
        encoder (str): Name of the encoder

    Returns:
        torch.nn.Module: The loaded model.
    """
    assert model_dpath.exists(), f"Model weights not found at {model_dpath}"
    model_fpath = f"{model_dpath}/depth_anything_v2_{encoder}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DepthAnythingV2(**model_configs[encoder])
    model_state_dict = torch.load(model_fpath, map_location="cpu", weights_only=False)
    model.load_state_dict(model_state_dict)
    model = model.to(device).eval()

    return model


# Perform inference on a batch of images
def depth_inference(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Performs depth inference on a batch of images.

    Args:
        model (torch.nn.Module): The pre-trained DepthAnything v2 model.
        images (torch.Tensor): The batch of preprocessed images.

    Returns:
        torch.Tensor: Predicted depth maps for the input images.
    """
    with torch.no_grad():  # Disable gradient calculation for inference
        depth_maps = model.infer_image(images)
    return depth_maps


def read_depth_npy(fpath: Union[str, Path]) -> np.ndarray:
    """
    Reads a depth .npy file and returns its contents as a numpy array.

    Args:
        fpath (Union[str, Path]): Path to the .npy file.

    Returns:
        np.ndarray: The depth data stored in the .npy file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid .npy file.
    """
    fpath = Path(fpath)
    if not fpath.is_file():
        raise FileNotFoundError(f"The file {fpath} does not exist.")

    # Read estimated depth
    depth = np.load(fpath)

    return depth


def process_depth(
    depth: np.ndarray, img_size: tuple, clip_max: float = None
) -> torch.Tensor:
    """
    Processes a depth ndarray by resizing, normalizing, and converting it to a PyTorch tensor.

    Args:
        depth (np.ndarray): The input depth data.
        img_size (tuple): The target image size as (h, w).
        clip_max (float, optional): If provided, depth values are clamped to [0, clip_max]
            (in metros) BEFORE the min-max normalization. Pixels inválidos (==0)
            permanecem zero.

    Returns:
        torch.Tensor: The processed depth tensor with shape (1, h, w).
    """
    # Convert depth to a PyTorch tensor
    depth_tensor = torch.from_numpy(depth).float()

    # Optional clip on metric values (mantém zeros como zero)
    if clip_max is not None and clip_max > 0:
        depth_tensor = torch.clamp(depth_tensor, min=0.0, max=float(clip_max))

    # Resize depth to match the image size
    depth_tensor = F.interpolate(
        depth_tensor.unsqueeze(0).unsqueeze(0),
        size=img_size,
        mode="bilinear",
        align_corners=True,
    ).squeeze(0)

    # Min-max normalization
    depth_tensor = (depth_tensor - depth_tensor.min()) / (
        depth_tensor.max() - depth_tensor.min()
    )

    return depth_tensor


if __name__ == "__main__":
    from datasets.kitti import KITTIDataset
    import matplotlib
    from PIL import Image
    import cv2

    # # Load model
    # model_dpath = Path("checkpoints/depth-anything")
    # encoder = "vits"
    # model = load_depth_model(model_dpath, encoder=encoder)

    # Load images
    data_dpath = Path("C:/workspace/data/kitti")
    sequences = ["00"]

    # Create dataloader
    preprocess = transforms.Compose(
        [
            transforms.Resize([192, 640]),
            transforms.ToTensor(),
        ]
    )
    dataset = KITTIDataset(
        data_dpath=data_dpath,
        sequences=sequences,
        window_size=3,
        overlap=2,
        transforms=preprocess,
        estimate_depth=True,
    )
    images, gt = dataset[0]  # Get the first window

    # # Perform inference
    # img1_fpath = "C:/workspace/data/kitti/sequences/00/image_2/001159.png"
    # img2_fpath = "C:/workspace/data/kitti/sequences/02/image_2/001000.png"
    # img1 = Image.open(img1_fpath).convert("RGB")
    # img1 = preprocess(img1)
    # raw_img1 = cv2.imread(img1_fpath)
    # depth = depth_inference(model, img1)

    # Save or visualize the depth maps
    output_folder = Path("./tmp")
    output_folder.mkdir(exist_ok=True)

    idx = 1070
    for i in range(images.shape[1]):
        images, gt = dataset[idx]
        depth = images[3, i, :].numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        cmap = matplotlib.colormaps.get_cmap("Spectral_r")
        # depth = np.repeat(depth[..., np.newaxis], 3, axis=-1) # grayscale
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        cv2.imwrite(output_folder / f"depth_idx_{idx}_{i}.png", depth)
