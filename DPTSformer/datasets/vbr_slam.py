from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
import os

from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data_utils import (
    quaternion_to_rotation_matrix,
    rotation_to_euler,
    load_normalization,
    vbr_to_kitti_coordinate_system,
)
from models.depth import read_depth_npy, process_depth

# Workaround for OpenMP issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class VBRDataset(Dataset):
    def __init__(
        self,
        data_dpath: Union[str, Path],
        sequences: List[str] = ["campus_train0"],
        window_size: int = 3,
        overlap: int = 2,
        transforms: Optional[callable] = None,
        normalize_gt: bool = True,
        estimate_depth: bool = False,
    ) -> None:
        """
        Initializes the VBRDataset.

        Args:
            data_dpath (Union[str, Path]): Path to the data directory.
            sequences (List[str]): List of sequence names to load.
            window_size (int): Size of the sliding window (number of frames).
            overlap (int): Number of overlapping frames between consecutive windows.
            transforms (Optional[callable]): Optional transformations to apply to images.
            normalize_gt (bool): flag to normalize poses ground truth
            estimate_depth (bool): flag to estimate the depth (monocular)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.transforms = transforms
        self.data_dpath = Path(data_dpath)
        self.sequences = sequences
        self.normalize_gt = normalize_gt
        self.estimate_depth = estimate_depth

        # Normalization parameters
        self.mean_angles, self.std_angles, self.mean_t, self.std_t = load_normalization(
            stats_file="datasets/dataset_stats.json", dataset_name="vbr_slam"
        )

        # Build data dictionary with frames and ground truths for each sequence
        self.data_dict = self._build_data_dict(self.data_dpath, self.sequences)

        # Build windowed dict based on the window size and overlap
        self.windowed_dict = self._create_windowed_dict(
            self.data_dict, self.window_size, self.overlap, self.sequences
        )

    def __len__(self) -> int:
        """
        Returns the total number of windowed samples in the dataset.
        """
        return len(self.windowed_dict)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Retrieves the data for a specific window index.
        """
        # Get frames and ground truth of a given window index
        windowed_data = self.windowed_dict[idx]
        frames, gt_poses = windowed_data["frames"], windowed_data["ground_truth"]

        # Load and transform the images for the current window
        imgs = []
        for frame_fpath in frames:
            img = Image.open(frame_fpath).convert("RGB")
            if self.transforms:
                img = self.transforms(img)

            # Estimate depth
            if self.estimate_depth:
                depth_fpath = (
                    str(frame_fpath)
                    .replace("camera_left\data", "camera_left\depth")
                    .replace(".png", ".npy")
                )
                depth = read_depth_npy(depth_fpath)

                # Process depth
                depth = process_depth(depth, img_size=img.shape[1:])

                # Concatenate depth to channel dim
                img = torch.cat((img, depth), axis=0)

            imgs.append(img.unsqueeze(0))  # (1, C, H, W)

        # Stack images into a single tensor and permute the axes
        imgs = torch.cat(imgs, dim=0)  # (window_size, C, H, W)
        imgs = imgs.permute(1, 0, 2, 3)  # (C, window_size, H, W)

        # Process the ground truth poses for the current window
        gt_poses = self._compute_relative_poses(gt_poses, self.normalize_gt)

        return imgs, gt_poses

    def _build_data_dict(self, data_dpath: Path, sequences: List) -> Dict:
        """
        Build a dictionary containing frame paths and corresponding ground truth data
        for each sequence in the dataset.

        This method iterates over all sequences provided, loads frame file paths and their
        associated ground truth data (translation and rotation). The data is stored in a dictionary
        where each key is a sequence name, and the value is another dictionary containing:
            - 'frames': A list of file paths to the frames.
            - 'ground_truth': A NumPy array of ground truth values for each frame.

        Args:
            data_dpath (Path): The base directory path where the sequence data is located.

            sequences (List[str]): A list of sequence names (strings) to iterate over and load data for.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary where each key is a sequence name, and the value
                is a dictionary containing 'frames' (list of frame paths) and 'ground_truth' (NumPy array).
        """
        data_dict = {}

        # Iterate over all sequences
        for sequence in sequences:
            sequence_dpath = data_dpath / sequence

            if sequence_dpath.is_dir():
                # Get all frame file paths
                frames_fpath = self._list_frames(sequence_dpath / "camera_left/data/")

                # Load ground truth data (translation and rotation)
                gt_sequence = self._load_ground_truth(
                    data_dpath / sequence / f"{sequence}_gt.txt"
                )

                # Ensure the number of frames and ground truth entries match
                clip_list_idx = min(len(gt_sequence), len(frames_fpath))
                frames_fpath = frames_fpath[:clip_list_idx]
                gt_sequence = gt_sequence.iloc[:clip_list_idx, 1:].reset_index(
                    drop=True
                )

                # Convert x-forward to the KITTI z-forward format
                gt_sequence = vbr_to_kitti_coordinate_system(gt_sequence.values)

                # Store the frames and ground truth in the data_dict
                data_dict[sequence] = {
                    "frames": frames_fpath,
                    "ground_truth": gt_sequence,
                }

        return data_dict

    def _create_windowed_dict(
        self, data_dict: Dict, window_size: int, overlap: int, sequences: List
    ) -> Dict:
        """
        Create a dictionary of windowed data for all sequences.

        This method generates windows of data from the provided data_dict. Each window contains
        a fixed number of frames (window_size) and is created with the specified overlap
        between windows.

        Args:
            data_dict Dict: A dictionary where each key is a sequence name
                and the value is another dictionary containing 'frames' and 'ground_truth'.
            window_size (int): The size of each window (number of frames).
            overlap (int): The number of frames that overlap between consecutive windows.
            sequences List: A list of sequence names.

        Returns:
            Dict: A dictionary where each key is a window index (w_idx)
                and the value is a dictionary containing 'frames' and 'ground_truth' for that window.
        """
        windowed_dict = {}
        w_idx = 0

        # Iterate through each unique sequence in the data_dict
        for sequence in sequences:
            if sequence in data_dict:
                sequence_frames = data_dict[sequence]["frames"]
                sequence_gt = data_dict[sequence]["ground_truth"]

                # Create windows for the current sequence
                row_idx = 0
                while row_idx + window_size <= len(sequence_frames):
                    # Extract windowed data
                    windowed_data = {
                        "frames": sequence_frames[row_idx : row_idx + window_size],
                        "ground_truth": sequence_gt[row_idx : row_idx + window_size],
                    }

                    # Store the windowed data in the windowed_dict
                    windowed_dict[w_idx] = windowed_data

                    # Update indices for the next window
                    row_idx += window_size - overlap
                    w_idx += 1

        return windowed_dict

    def _load_ground_truth(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Loads the ground truth pose data from a file.

        Args:
            file_path (Union[str, Path]): Path to the ground truth file.

        Returns:
            pd.DataFrame: A DataFrame containing ground truth data (timestamps, translations, and rotations).
        """
        # Read the ground truth file and assign column names
        df = pd.read_csv(file_path, sep=r"\s+", comment="#", header=None)
        df.columns = ["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
        return df

    def _list_frames(self, frames_dpath: Path) -> List[str]:
        """
        Lists all the frame file paths in the given directory.

        Args:
            frames_dpath (Path): Path to the directory containing the frame files.

        Returns:
            List[str]: A list of sorted file paths for all frames.
        """
        # Return a sorted list of all PNG files in the directory
        return sorted([fpath for fpath in frames_dpath.glob("*.png")])

    def _compute_relative_poses(self, gt_poses: list, normalize_gt: bool) -> np.ndarray:
        """
        Compute relative poses between consecutive frames in a window.

        Args:
            gt_poses (np.ndarray): Ground truth poses in the format (tx, ty, tz, qx, qy, qz, qw).
            normalize_gt (bool): Flag to normalize ground truth poses

        Returns:
            np.ndarray: Flattened array of relative poses
        """
        y = []
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

                # Normalize 6-DoF
                if normalize_gt:
                    angles, t = self._normalize_pose(angles, t)

                # row, pitch, yaw, tx, ty, tz (z-forward)
                y.append(list(angles) + list(t))

            pose_prev = pose

        return np.asarray(y).flatten()

    def _normalize_pose(
        self, angles: List[float], t: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalizes the angles and translations using mean and standard
        deviation values.

        Args:
            angles (List[float]): List of angle values to be normalized.
            t (List[float]): List of translation values to be normalized.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing normalized angles and translations.
        """
        normalized_angles = (np.array(angles) - self.mean_angles) / self.std_angles
        normalized_t = (np.array(t) - self.mean_t) / self.std_t
        return normalized_angles, normalized_t


# Example usage
if __name__ == "__main__":
    from torchvision import transforms
    from utils.data_utils import (
        visualize_trajectory_with_frames,
        visualize_samples,
        visualize_sequence_trajectory,
    )

    data_dpath = Path("C:/workspace/data/vbr_slam")
    sequences = ["campus_train0", "campus_train1"]

    transforms = transforms.Compose(
        [
            transforms.Resize((192, 640)),  # (192, 640) kitti
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.34721234, 0.36705238, 0.36066107],
            #     std=[0.30737526, 0.31515116, 0.32020183],
            # ),
        ]
    )

    dataset = VBRDataset(
        data_dpath=data_dpath,
        sequences=sequences,
        window_size=3,
        overlap=2,
        transforms=transforms,
    )
    imgs, gt = dataset[0]  # Get the first window
    print("Images shape:", imgs.shape)
    print("Ground truth shape:", gt.shape)

    # visualize_samples(dataset)
    for sequence in sequences:
        gt_poses = dataset.data_dict[sequence]["ground_truth"]
        frames_fpath = dataset.data_dict[sequence]["frames"]
        visualize_trajectory_with_frames(
            frames_fpath,
            gt_poses,
            sequence=sequence,
            num_frames=500,
            delay=0.04,
            scale=1.0,
        )
        # visualize_sequence_trajectory(gt_poses, sequence)
