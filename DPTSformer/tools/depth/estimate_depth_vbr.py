import torch
from pathlib import Path
import cv2
import matplotlib
import numpy as np
from models.depth import load_depth_model
from tqdm import tqdm

# Initialize paths and parameters
data_dpath = Path("C:/workspace/data/vbr_slam")
sequences = ["campus_train0", "campus_train1"]
cmap = matplotlib.colormaps.get_cmap("Spectral_r")

# Load the depth model
model_dpath = Path("checkpoints/depth-anything")
depth_encoder = "vitb"
depth_model = load_depth_model(model_dpath, encoder=depth_encoder)

for sequence in sequences:
    sequence_dpath = data_dpath / sequence / "camera_left" / "data"
    depth_dpath = data_dpath / sequence / "camera_left" / "depth"
    depth_dpath.mkdir(parents=True, exist_ok=True)

    # List all .png files in the directory
    frame_files = sorted(sequence_dpath.glob("*.png"))

    # Process each frame
    for frame_fpath in tqdm(frame_files, desc=f"Seq.{sequence}"):
        # Load and preprocess the image
        img = cv2.imread(str(frame_fpath))

        # Estimate depth
        depth = depth_model.infer_image(img)

        # Save depth as a .npy file
        depth_fname = depth_dpath / (frame_fpath.stem + ".npy")
        np.save(depth_fname, depth)

        # # Save as png
        # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        # depth = depth.astype(np.uint8)
        # depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        # depth_fname = depth_dpath / (frame_fpath.stem + ".png")
        # cv2.imwrite(depth_fname, depth)

        # print(depth_fname)
