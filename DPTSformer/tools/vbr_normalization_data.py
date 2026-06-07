from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils.data_utils import rotation_to_euler, quaternion_to_rotation_matrix
from datasets.vbr_slam import VBRDataset


def compute_pose_statistics(dataset: Dataset, sequences: List[str]) -> None:
    """Computes the mean and std of Euler angles and translations from KITTI poses.

    Args:
        dataset (Dataset): Dataset instance.
        sequences (List[str]): List of sequences to process.
    """
    y = []
    for sequence in sequences:
        gt_poses = dataset.data_dict[sequence]["ground_truth"]
        pose_prev = None

        # Iterate through each pose in the window
        for gt in gt_poses:
            t = gt[:3]  # Translation (tx, ty, tz)
            q = gt[3:]  # Quaternion (qx, qy, qz, qw)

            # Convert quaternion to a rotation matrix
            R = quaternion_to_rotation_matrix(q)

            # Construct a 4x4 transformation matrix
            pose = np.vstack([np.hstack([R, t.reshape(-1, 1)]), [0.0, 0.0, 0.0, 1.0]])

            if pose_prev is not None:
                # Compute the relative transformation between the current and previous poses
                pose_wrt_prev = np.dot(np.linalg.inv(pose_prev), pose)
                R = pose_wrt_prev[:3, :3]
                t = pose_wrt_prev[:3, 3]
                angles = rotation_to_euler(R)

                # row, pitch, yaw, tx, ty, tz (z-forward)
                y.append(list(angles) + list(t))

            pose_prev = pose

    y = np.asarray(y)

    # Compute mean and std for Euler angles and translations
    mean_angles = [np.mean(y[:, 0]), np.mean(y[:, 1]), np.mean(y[:, 2])]
    std_angles = [np.std(y[:, 0]), np.std(y[:, 1]), np.std(y[:, 2])]
    mean_t = [np.mean(y[:, 3]), np.mean(y[:, 4]), np.mean(y[:, 5])]
    std_t = [np.std(y[:, 3]), np.std(y[:, 4]), np.std(y[:, 5])]

    print("--- Euler angles ---")
    print("Mean:", mean_angles)
    print("Std:", std_angles)

    print("\n--- Translation ---")
    print("Mean:", mean_t)
    print("Std:", std_t)


def compute_rgb_statistics(
    dataset: Dataset, sequences: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the mean and standard deviation for each RGB channel across multiple sequences.

    Args:
        dataset (Dataset): Dataset instance.
        sequences (List[str]): List of sequence folder names to include in the calculation.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Mean and std dev for each channel (R, G, B).
    """
    means = []
    stds = []

    # Iterate over each sequence
    for sequence in tqdm(sequences):

        # Process each image in the sequence folder
        frames_fpath = dataset.data_dict[sequence]["frames"]
        for frame_fpath in tqdm(
            frames_fpath,
            desc=f"Sequence {sequence}",
            leave=False,
        ):

            image = Image.open(frame_fpath).convert("RGB")
            image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

            # Compute per-channel mean and std for the current image
            means.append(np.mean(image, axis=(0, 1)))
            stds.append(np.std(image, axis=(0, 1)))

    # Calculate the dataset-wide mean and std dev by averaging per-image values
    mean_rgb = np.mean(means, axis=0)
    std_rgb = np.mean(stds, axis=0)

    return mean_rgb, std_rgb


if __name__ == "__main__":
    data_dpath = "C:/workspace/data/vbr_slam"
    sequences = [
        "campus_train0",
        "campus_train1",
    ]

    # Create dataloader
    dataset = VBRDataset(
        data_dpath=data_dpath,
        sequences=sequences,
        window_size=2,
        overlap=1,
    )

    # Compute pose statistics
    compute_pose_statistics(dataset, sequences)

    # Compute RGB statistics
    mean_rgb, std_rgb = compute_rgb_statistics(dataset, sequences)
    print("\n --- RGB stats ---")
    print("Mean RGB:", mean_rgb)
    print("Std RGB:", std_rgb)
