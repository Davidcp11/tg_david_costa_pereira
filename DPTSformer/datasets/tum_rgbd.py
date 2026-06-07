from typing import Union, Tuple, List, Dict
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from utils.data_utils import (
    rotation_to_euler,
    load_normalization,
    quaternion_to_rotation_matrix,
    tum_to_kitti_coordinate_system,
)


class TUMRGBDDataset(torch.utils.data.Dataset):
    """
    Pytorch Dataset for TUM RGB-D SLAM
    https://cvg.cit.tum.de/data/datasets/rgbd-dataset
    """

    def __init__(
        self,
        data_dpath: Union[str, Path] = "data/tum_rgbd",
        split: str = "train",
        scenes: List[str] = [],
        window_size: int = 3,
        overlap: int = 2,
        read_depth: bool = True,
        transforms: Union[None, torch.nn.Module] = None,
        transforms_depth: Union[None, torch.nn.Module] = None,
        normalize_gt: bool = False,
        sequences: Union[None, List[str]] = None,
    ):
        """
        Initializes the TUM RGB-D dataset.

        Args:
            data_dpath (Union[str, Path]): Path to data directory.
            split (str): Dataset split, either 'train' or 'test'.
            scenes (List[str]): List of scenes to include.
            window_size (int): Number of frames per window.
            overlap (int): Overlap between windows.
            read_depth (bool): Whether to read depth data.
            transform (torch.nn.Module, optional): Transform for RGB images.
            transform_depth (torch.nn.Module, optional): Transform for depth images.
            normalize_gt (bool): Flag to normalize ground truth pose.
            sequences (List[str], optional): Specific sequences to include.
        """

        self.data_dpath = Path(data_dpath)
        self.scenes = scenes
        self.window_size = window_size
        self.overlap = overlap
        self.split = split
        self.transforms = transforms
        self.transforms_depth = transforms_depth
        self.read_depth = read_depth
        self.normalize_gt = normalize_gt
        self.sequences = sequences

        # 7Scenes normalization parameters
        self.mean_angles, self.std_angles, self.mean_t, self.std_t = load_normalization(
            stats_file="datasets/dataset_stats.json", dataset_name="7scenes"
        )

        # Build data dictionary with frames and ground truths for each sequence
        self.data_dict = self._build_data_dict(self.data_dpath, self.sequences)

        # Build windowed dict based on the window size and overlap
        self.windowed_dict = self._create_windowed_dict(
            self.data_dict, self.window_size, self.overlap
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

            # Read depth and concatenate in the channel dimension
            if self.read_depth:
                depth_fpath = Path(str(frame_fpath).replace(".color", ".depth"))
                depth = Image.open(depth_fpath)
                if self.transforms_depth:
                    depth = self.transforms_depth(depth)
                # depth[depth == 65535] = 0  # invalid depth value
                depth = torch.div(depth, 65535)  # [0,1] range
                # depth = torch.div(depth, 1000)  # depth values in meters
                img = torch.cat([img, depth], axis=0)
            imgs.append(img.unsqueeze(0))  # (1, C, H, W)

        # Stack images into a single tensor and permute the axes
        imgs = torch.cat(imgs, dim=0)  # (window_size, C, H, W)
        imgs = imgs.permute(1, 0, 2, 3)  # (C, window_size, H, W)

        # Process the ground truth poses for the current window
        gt_poses = self._compute_relative_poses(gt_poses, self.normalize_gt)

        return imgs, gt_poses

    def _build_data_dict(
        self,
        data_dpath: Path,
        sequences: Union[None, List[str]],
    ) -> Dict:
        """
        Builds a dictionary of frame paths and ground truth poses for each sequence.

        Args:
            data_dpath (Path): Path to data directory.
            sequences (List[str]): List of sequences to include.

        Returns:
            Dict: Dictionary of frames and ground truth.
        """
        data_dict = {}

        # Iterate over all sequences
        for sequence in sequences:

            # Define paths for associations file
            associations_fpath = data_dpath / sequence / "associations.txt"

            # Read associations file
            # rgb_timestamp rgb_path depth_timestamp depth_path gt_timestamp gt
            associations = self._read_associations(associations_fpath)

            # List all frames and depths file paths
            frames_fpath = [
                data_dpath / sequence / fname for fname in associations[:, 1]
            ]
            depths_fpath = [
                data_dpath / sequence / fname for fname in associations[:, 3]
            ]

            # Load ground truth data (tx, ty, tz, qx, qy, qz, qw)
            gt_sequence = associations[:, 5:].astype(float)

            # Convert to the KITTI z-forward format
            gt_sequence = tum_to_kitti_coordinate_system(np.asarray(gt_sequence))

            # Store the frames and ground truth in the data_dict
            data_dict[sequence] = {
                "frames": frames_fpath,
                "depths": depths_fpath,
                "ground_truth": gt_sequence,
            }

        return data_dict

    def _read_associations(self, file_fpath: Path) -> List:
        """
        Reads an associations.txt file at the specified file path and returns its data as a list.

        Each line in the associations.txt file contains timestamped information, including:
        - RGB image timestamp and path
        - Depth image timestamp and path
        - Ground truth timestamp
        - Pose values (7 floats representing tx, ty, tz, qx, qy, qz, qw)

        Parameters:
        ----------
        file_fpath : Path
            Path to the associations.txt file.

        Returns:
        -------
        List: A aray where each element contains:
            - RGB timestamp (float), RGB image path (str)
            - Depth timestamp (float), Depth image path (str)
            - Ground truth timestamp (float)
            - Pose values as floats (tx, ty, tz, qx, qy, qz, qw)
        """
        associations = []

        # Read and parse the associations.txt file
        with file_fpath.open("r") as file:
            for line in file:
                # Split the line into elements by whitespace
                elements = line.strip().split()

                # Parse and unpack each part to its appropriate data type
                rgb_timestamp = float(elements[0])
                rgb_path = elements[1]
                depth_timestamp = float(elements[2])
                depth_path = elements[3]
                gt_timestamp = float(elements[4])
                tx = float(elements[5])
                ty = float(elements[6])
                tz = float(elements[7])
                qx = float(elements[8])
                qy = float(elements[9])
                qz = float(elements[10])
                qw = float(elements[11])

                # Append as a tuple to the list
                associations.append(
                    [
                        rgb_timestamp,
                        rgb_path,
                        depth_timestamp,
                        depth_path,
                        gt_timestamp,
                        tx,
                        ty,
                        tz,
                        qx,
                        qy,
                        qz,
                        qw,
                    ]
                )

        # Convert list of lists to a structured NumPy array (mixed types)
        associations = np.array(associations, dtype=object)

        return associations

    def _create_windowed_dict(
        self, data_dict: Dict, window_size: int, overlap: int
    ) -> Dict:
        """
        Creates windows from frames and poses with overlap.

        Args:
            data_dict (dict): Dictionary of frames and ground truth.
            window_size (int): Size of each window.
            overlap (int): Overlap between windows.

        Returns:
            Dict: Windowed data dictionary.
        """
        windowed_dict = {}
        w_idx = 0

        # Iterate through each unique sequence in the data_dict
        sequences = list(data_dict.keys())
        for sequence in sequences:
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


if __name__ == "__main__":
    from utils.data_utils import (
        visualize_trajectory_with_frames,
        visualize_sequence_trajectory,
        visualize_samples,
    )
    from torchvision import transforms

    # Define the data path and scenes
    data_dpath = "G:/data/tum_rgbd"
    sequences = [
        # "rgbd_dataset_freiburg1_desk",
        # "rgbd_dataset_freiburg1_floor",
        # "rgbd_dataset_freiburg1_room",
        # "rgbd_dataset_freiburg1_xyz",
        # "rgbd_dataset_freiburg2_xyz",
        # "rgbd_dataset_freiburg3_xyz",
        "rgbd_dataset_freiburg3_sitting_xyz",
        # "rgbd_dataset_freiburg3_walking_halfsphere",
        # "rgbd_dataset_freiburg3_sitting_static",
        # "rgbd_dataset_freiburg3_sitting_rpy",
    ]
    image_size = (480, 640)

    # Define image and depth transforms
    transforms_rgb = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )
    transforms_depth = transforms.Compose(
        [transforms.Resize(image_size), transforms.ToTensor()]
    )

    for sequence in sequences:
        # Create the SevenScenes dataset instance
        dataset = TUMRGBDDataset(
            data_dpath=data_dpath,
            split="test",
            sequences=[sequence],
            window_size=3,
            overlap=2,
            read_depth=True,
            transforms=transforms_rgb,
            transforms_depth=transforms_depth,
            normalize_gt=False,
        )

        # Get a sample window of images and ground truth poses
        imgs, gt = dataset[0]
        print("Images shape:", imgs.shape)  # Expected shape: (C, T, H, W)
        print("Ground truth shape:", gt.shape)

        # Visualize samples from the dataset
        # visualize_samples(dataset)

        # Visualize trajectory
        gt_poses = dataset.data_dict[sequence]["ground_truth"]
        frames_fpath = dataset.data_dict[sequence]["frames"]
        visualize_trajectory_with_frames(
            frames_fpath,
            gt_poses,
            sequence=sequence,
            num_frames=400,
            delay=0.01,
        )
        # visualize_sequence_trajectory(gt_poses=gt_poses, sequence=sequence, depth=True)
