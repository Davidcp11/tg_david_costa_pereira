from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from typing import List, Tuple, Union
from torchvision import transforms

from utils.data_utils import (
    rotation_to_euler,
    load_normalization,
    quaternion_to_rotation_matrix,
    euroc_to_kitti_coordinate_system,
)


class EuRoCDataset(torch.utils.data.Dataset):
    """
    Dataloader for the EuRoC MAV Dataset
        https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

    Args:
        data_dpath (str): path to data sequences
        gt_dpath (str): path to poses
        camera_id (Union[str, int]): camera identifier (default="2")
        sequences (List[str]): list of sequence IDs to be loaded
        window_size (int): sliding window size for frames
        overlap (int): overlap size between consecutive windows
        read_poses (bool): flag to load ground truth poses (default=True)
        normalize_gt (bool): flag to normalize poses ground truth
        dataset_name_norm (str): Name of the dataset to load the normalization parameters
        transform (callable): transformation to be applied on frames
    """

    def __init__(
        self,
        data_dpath: Union[Path, str] = "data/euroc",
        camera_id: Union[str, int] = "0",
        sequences: List[str] = ["MH_01_easy", "MH_02_easy"],
        window_size: int = 3,
        overlap: int = 1,
        read_poses: bool = True,
        normalize_gt: bool = True,
        dataset_name_norm: str = "euroc",
        transforms: transforms.Compose = None,
    ):
        self.data_dpath = Path(data_dpath)
        self.camera_id = camera_id
        self.window_size = window_size
        self.overlap = overlap
        self.read_poses = read_poses
        self.normalize_gt = normalize_gt
        self.transforms = transforms
        self.sequences = sequences
        self.dataset_name_norm = dataset_name_norm

        # Normalization parameters
        self.mean_angles, self.std_angles, self.mean_t, self.std_t = load_normalization(
            stats_file="datasets/dataset_stats.json",
            dataset_name=self.dataset_name_norm,
        )

        # Build data dictionary with frames and ground truths for each sequence
        self.data_dict = self._build_data_dict(
            self.data_dpath, self.sequences, self.camera_id
        )

        # Build windowed dict based on the window size and overlap
        self.windowed_dict = self._create_windowed_dict(
            self.data_dict, self.window_size, self.overlap, self.sequences
        )

    def __len__(self) -> int:
        """Return the number of unique window indices."""
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
            imgs.append(img.unsqueeze(0))  # (1, C, H, W)

        # Stack images into a single tensor and permute the axes
        imgs = torch.cat(imgs, dim=0)  # (window_size, C, H, W)
        imgs = imgs.permute(1, 0, 2, 3)  # (C, window_size, H, W)

        # Process the ground truth poses for the current window
        gt_poses = self._compute_relative_poses(gt_poses, self.normalize_gt)

        return imgs, gt_poses

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

    def _load_ground_truth(self, gt_fpath) -> list:
        """
        Read ground truth poses in a given data path.

        Returns:
            list: List of ground truth pose vectors
        """
        # Check if the file exists
        if not gt_fpath.exists():
            raise FileNotFoundError(f"Ground truth file not found at: {gt_fpath}")

        # Read the data.csv with the correct columns
        columns = [
            "#timestamp",
            " p_RS_R_x [m]",
            " p_RS_R_y [m]",
            " p_RS_R_z [m]",
            " q_RS_w []",
            " q_RS_x []",
            " q_RS_y []",
            " q_RS_z []",
        ]
        gt_df = pd.read_csv(gt_fpath, usecols=columns)

        # Rename columns for convenience
        gt_df.columns = ["timestamp", "tx", "ty", "tz", "qw", "qx", "qy", "qz"]

        # Reorder columns to "qx, qy, qz, qw"
        gt_df = gt_df[["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]]

        return gt_df

    def _build_data_dict(
        self, data_dpath: Path, sequences: list, camera_id: str
    ) -> dict:
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
            sequences (list): A list of sequence names (strings) to iterate over and load data for.
            camera_id (Union[str, int]): Camera identifier

        Returns:
            dict: A dictionary where each key is a sequence name, and the value
                is a dictionary containing 'frames' and 'ground_truth'.
        """
        data_dict = {}

        # Iterate over all sequences
        for sequence in sequences:
            sequence_dpath = data_dpath / sequence / "mav0" / f"cam{camera_id}" / "data"
            gt_dpath = data_dpath / sequence / "mav0" / "state_groundtruth_estimate0"
            if sequence_dpath.is_dir():
                # Get all frame file paths
                frames_fpath = self._list_frames(sequence_dpath)

                # Load ground truth data (translation and rotation)
                gt_fpath = gt_dpath / "data.csv"
                gt_sequence = self._load_ground_truth(gt_fpath)

                # Synchronize timestamps
                gt_sequence = self._get_matched_ground_truths(frames_fpath, gt_sequence)

                # Convert x-forward to the KITTI z-forward format
                # gt_sequence = euroc_to_kitti_coordinate_system(gt_sequence)

                # Store the frames and ground truth in the data_dict
                data_dict[sequence] = {
                    "frames": frames_fpath,
                    "ground_truth": gt_sequence,
                }

        return data_dict

    def _get_matched_ground_truths(
        self, frames_fpath: List[Path], gt_sequence: pd.DataFrame
    ) -> np.ndarray:
        """
        Finds the best-fit ground truth entries for each frame based on the timestamp in the filename.

        Args:
            frames_fpath (List[Path]): List of paths to image frames, where each filename contains a timestamp.
            gt_sequence (pd.DataFrame): DataFrame containing ground truth data with a "timestamp" column
                                        and columns for 6-DoF pose information.

        Returns:
            np.ndarray: A list of ground truth rows (as Series) that best fit each frame's timestamp.
        """
        best_gts = []

        # Iterate over each frame path
        for frame_path in frames_fpath:
            # Extract timestamp from the filename (assuming filename format is <timestamp>.png)
            frame_timestamp = int(frame_path.stem)

            # Find the index of the smallest difference
            time_diffs = (gt_sequence["timestamp"] - frame_timestamp).abs()
            closest_idx = time_diffs.idxmin()

            # Get corresponding [tx, ty, tz, qx, qy, qz, qw]
            best_gts.append(gt_sequence.iloc[closest_idx, 1:].values)

        return np.asarray(best_gts)

    def _create_windowed_dict(
        self, data_dict: dict, window_size: int, overlap: int, sequences: list
    ) -> dict:
        """
        Create a dictionary of windowed data for all sequences.

        This method generates windows of data from the provided data_dict. Each window contains
        a fixed number of frames (window_size) and is created with the specified overlap
        between windows.

        Args:
            data_dict (dict): A dictionary where each key is a sequence name
                and the value is another dictionary containing 'frames' and 'ground_truth'.
            window_size (int): The size of each window (number of frames).
            overlap (int): The number of frames that overlap between consecutive windows.
            sequences (list): A list of sequence names.

        Returns:
            dict: A dictionary where each key is a window index (w_idx)
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

                # row, pitch, yaw, tx, ty, tz
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


if __name__ == "__main__":

    from utils.data_utils import (
        visualize_trajectory_with_frames,
        visualize_samples,
        visualize_sequence_trajectory,
    )

    data_dpath = "G:/data/euroc"
    sequences = [
        # "MH_01_easy",
        # "MH_02_easy",
        "MH_04_difficult",
        # "V1_01_easy",
        # "V1_02_medium",
        # "V2_01_easy",
    ]  # , "MH_01_easy", "MH_02_easy"]

    # Create dataloader
    preprocess = transforms.Compose(
        [
            transforms.Resize((192, 640)),
            transforms.ToTensor(),
        ]
    )
    dataset = EuRoCDataset(
        data_dpath=data_dpath,
        sequences=sequences,
        window_size=3,
        overlap=2,
        transforms=preprocess,
        normalize_gt=False,
    )

    imgs, gt = dataset[0]  # Get the first window
    print("Images shape:", imgs.shape)
    print("Ground truth shape:", gt.shape)

    # visualize_samples(dataset)
    for sequence in sequences:
        gt_poses = dataset.data_dict[sequence]["ground_truth"]
        frames_fpath = dataset.data_dict[sequence]["frames"]
        visualize_trajectory_with_frames(
            frames_fpath, gt_poses, sequence=sequence, num_frames=500
        )

        # visualize_sequence_trajectory(np.asarray(gt_poses), sequence)
