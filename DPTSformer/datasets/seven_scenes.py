from typing import Union, Tuple, List, Dict
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from utils.data_utils import (
    rotation_to_euler,
    load_normalization,
    read_7scenes_split,
    seven_scenes_to_kitti_coordinate_system,
    save_depth_map,
)


class SevenScenesDataset(torch.utils.data.Dataset):
    """
    Pytorch Dataset for SevenScenes
    https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
    """

    def __init__(
        self,
        data_dpath: Union[str, Path] = "data/7scenes",
        split: str = "train",
        scenes: List[str] = [
            "chess",
            "fire",
            "heads",
            "office",
            "pumpkin",
            "redkitchen",
            "stairs",
        ],
        window_size: int = 3,
        overlap: int = 2,
        read_depth: bool = True,
        transforms: Union[None, torch.nn.Module] = None,
        transforms_depth: Union[None, torch.nn.Module] = None,
        normalize_gt: bool = False,
        sequences: Union[None, List[str]] = None,
    ):
        """
        Initializes the SevenScenes dataset.

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
        self.data_dict = self._build_data_dict(
            self.data_dpath, self.sequences, self.split, self.scenes
        )

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
        depths = []
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
                # depth[depth == 65535] = -1  # invalid depth value
                depth[depth > 40000] = -1  # invalid depth value

                # depth = torch.div(depth, 65535)  # [0,1] range
                depth = torch.div(depth, 1000.0)  # depth values in meters
                # save_depth_map(
                #     "tmp/7scenes",
                #     depth_fpath.stem,
                #     depth,
                #     use_colormap=True,
                # )
                depths.append(depth)
            imgs.append(img.unsqueeze(0))  # (1, C, H, W)

        # Stack images into a single tensor and permute the axes
        imgs = torch.cat(imgs, dim=0)  # (window_size, C, H, W)
        imgs = imgs.permute(1, 0, 2, 3)  # (C, window_size, H, W)
        depths = torch.cat(depths, dim=0)

        # Process the ground truth poses for the current window
        gt_poses = self._compute_relative_poses(gt_poses, self.normalize_gt)

        return imgs, gt_poses, depths

    def _build_data_dict(
        self,
        data_dpath: Path,
        sequences: Union[None, List[str]],
        split: str,
        scenes: List[str],
    ) -> Dict:
        """
        Builds a dictionary of frame paths and ground truth poses for each sequence.

        Args:
            data_dpath (Path): Path to data directory.
            sequences (List[str]): List of sequences to include.
            split (str): Dataset split, either 'train' or 'test'.
            scenes (List[str]): List of scenes to include.

        Returns:
            Dict: Dictionary of frames and ground truth.
        """
        data_dict = {}

        for scene in scenes:
            sequences = self._define_sequences(data_dpath, split, scene)

            # Iterate over all sequences
            for sequence_number in sequences:

                # Define paths for frames and ground truths
                frames_dpath = data_dpath / scene / f"seq-{sequence_number}"

                # List all frames file paths
                frames_fpath = self._list_frames(frames_dpath)

                # Load ground truth data (translation and rotation)
                gt_sequence = self._load_ground_truth(frames_fpath)

                # Convert x-forward to the KITTI z-forward format
                # gt_sequence = seven_scenes_to_kitti_coordinate_system(
                #     np.asarray(gt_sequence)
                # )

                # Store the frames and ground truth in the data_dict
                sequence_name = f"{scene}-{sequence_number}"
                data_dict[sequence_name] = {
                    "frames": frames_fpath,
                    "ground_truth": gt_sequence,
                }

        return data_dict

    def _list_frames(self, frames_dpath: Path) -> List[str]:
        """
        Lists all the frame file paths in the given directory.

        Args:
            frames_dpath (Path): Path to the directory containing the frame files.

        Returns:
            List[str]: A list of sorted file paths for all frames.
        """
        # Return a sorted list of all PNG files in the directory
        return sorted([fpath for fpath in frames_dpath.glob("*.color.png")])

    def _load_ground_truth(self, frames_fpath: List[str]) -> np.ndarray:
        """
        Loads ground truth poses from files.

        Args:
            frames_fpath (List[str]): List of frame file paths.

        Returns:
            np.ndarray: Array of ground truth poses.
        """

        gt = []
        # Read ground truth poses
        for gt_fpath in frames_fpath:
            gt_fpath = Path(str(gt_fpath).replace(".color.png", ".pose.txt"))
            # Homogeneous pose matrix [4 x 4]
            pose = np.loadtxt(gt_fpath)
            gt.append(pose.ravel().tolist())
        return np.array(gt)

    def _define_sequences(self, data_dpath: Path, split: str, scene: str) -> List[str]:
        """
        Defines sequences based on split.

        Args:
            data_dpath (Path): Path to data directory.
            split (str): Dataset split, either 'train' or 'test'.
            scene (str): Scene name.

        Returns:
            List[str]: List of sequence numbers as strings.
        """
        if split == "train":
            split_fpath = data_dpath / scene / "TrainSplit.txt"
            sequences = read_7scenes_split(split_fpath)
            sequences = ["{:02d}".format(seq) for seq in sequences]
        else:
            # run for specific sequence during test
            sequences = self.sequences
            # split_fpath = data_dpath / scene / "TestSplit.txt"
        return sequences

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

    def _compute_relative_poses(self, poses: list, normalize_gt: bool) -> np.ndarray:
        """
        Compute relative poses between consecutive frames in a window.

        Args:
            poses (list): List of poses in the window
            normalize_gt (bool): Flag to normalize ground truth poses

        Returns:
            np.ndarray: Flattened array of relative poses
        """
        y = []
        for i in range(1, len(poses)):
            pose_prev = np.reshape(poses[i - 1], (4, 4))
            pose_curr = np.reshape(poses[i], (4, 4))
            pose_wrt_prev = np.dot(np.linalg.inv(pose_prev), pose_curr)

            # Rotation and translation
            R = pose_wrt_prev[:3, :3]
            t = pose_wrt_prev[:3, 3]

            # Rotation to Euler angles
            angles = rotation_to_euler(R, seq="xyz")

            # Pose normalization
            if normalize_gt:
                angles, t = self._normalize_pose(angles, t)

            # Concatenate rotation and translation
            y.append(np.concatenate([angles, t]))

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
    from torchvision import transforms

    # Define the data path and scenes
    data_dpath = "C:/workspace/data/7scenes"
    scenes = ["chess", "fire"]  # ["heads", "office", "pumpkin", "redkitchen", "stairs"]
    image_size = (240, 320)

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

    # Visualize trajectories for each scene
    for scene in scenes:

        # Read sequences
        split_fpath = Path(data_dpath) / scene / "TestSplit.txt"
        sequences = read_7scenes_split(split_fpath)
        sequences = ["{:02d}".format(seq) for seq in sequences]

        for sequence in sequences:
            # Create the SevenScenes dataset instance
            dataset = SevenScenesDataset(
                data_dpath=data_dpath,
                split="test",
                scenes=[scene],
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
            sequence_name = f"{scene}-{sequence}"
            gt_poses = dataset.data_dict[sequence_name]["ground_truth"]
            gt_poses = gt_poses[:, [3, 7, 11]]
            frames_fpath = dataset.data_dict[sequence_name]["frames"]
            visualize_trajectory_with_frames(
                frames_fpath,
                gt_poses,
                sequence=sequence_name,
                num_frames=200,
                delay=0.1,
            )
            # visualize_sequence_trajectory(gt_poses, sequence_name, depth=True)
