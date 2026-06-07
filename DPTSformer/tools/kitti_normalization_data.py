from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm
import numpy as np
from PIL import Image
from utils.data_utils import rotation_to_euler


def read_gt(gt_dpath: Path, sequence: str) -> List[np.ndarray]:
    """Reads ground truth poses from KITTI dataset for a specific sequence.

    Args:
        gt_dpath (Path): Path to the directory containing ground truth poses.
        sequence (str): The sequence number.

    Returns:
        List[np.ndarray]: List of poses as 3x4 numpy arrays.
    """
    sequence_fpath = gt_dpath / f"{sequence}.txt"
    poses = []
    with sequence_fpath.open("r") as file:
        for line in file:
            values = list(map(float, line.strip().split()))
            poses.append(np.array(values).reshape(3, 4))
    return poses


def compute_pose_statistics(gt_path: Path, sequences: List[str]) -> None:
    """Computes the mean and std of Euler angles and translations from KITTI poses.

    Args:
        gt_path (Path): Path to the directory containing KITTI ground truth poses.
        sequences (List[str]): List of sequences to process.
    """
    y = []
    for sequence in sequences:
        gt_poses = read_gt(gt_path, sequence)
        pose_prev = None

        for gt_idx, gt in enumerate(gt_poses):
            pose = np.vstack([gt, [0.0, 0.0, 0.0, 1.0]])

            if gt_idx > 0:
                pose_wrt_prev = np.dot(np.linalg.inv(pose_prev), pose)
                rotMat = pose_wrt_prev[:3, :3]
                t = pose_wrt_prev[:3, 3]

                angles = rotation_to_euler(rotMat)
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
    images_dpath: Path, sequences: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the mean and standard deviation for each RGB channel across multiple sequences.

    Args:
        images_dpath (Path): Path to the folder containing sequences (each with RGB images).
        sequences (List[str]): List of sequence folder names to include in the calculation.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Mean and std dev for each channel (R, G, B).
    """
    means = []
    stds = []
    camera_id = 2

    # Iterate over each sequence
    for sequence in tqdm(sequences):
        sequence_dpath = images_dpath / sequence / f"image_{camera_id}"

        # Process each image in the sequence folder
        images_fpath = list(sequence_dpath.glob("*.png"))
        for img_fpath in tqdm(
            images_fpath,
            desc=f"Sequence {sequence}",
            leave=False,
        ):

            image = Image.open(img_fpath).convert("RGB")
            image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

            # Compute per-channel mean and std for the current image
            means.append(np.mean(image, axis=(0, 1)))
            stds.append(np.std(image, axis=(0, 1)))

    # Calculate the dataset-wide mean and std dev by averaging per-image values
    mean_rgb = np.mean(means, axis=0)
    std_rgb = np.mean(stds, axis=0)

    return mean_rgb, std_rgb


if __name__ == "__main__":
    data_dpath = Path("C:/workspace/data/kitti")
    sequences = ["00", "02", "08", "09"]

    # Compute pose statistics
    gt_path = data_dpath / "poses"
    compute_pose_statistics(gt_path, sequences)

    # Compute RGB statistics
    images_dpath = data_dpath / "sequences"
    mean_rgb, std_rgb = compute_rgb_statistics(images_dpath, sequences)
    print("\n --- RGB stats ---")
    print("Mean RGB:", mean_rgb)
    print("Std RGB:", std_rgb)
