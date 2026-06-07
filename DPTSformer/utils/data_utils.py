from typing import Tuple, Union
from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import cv2
import torch

from scipy.spatial.transform import Rotation as R


def rotation_to_euler(
    rotation: np.ndarray, seq: str = "xyz"
) -> Tuple[float, float, float]:
    """Convert rotation matrix to Euler angles.

    Args:
        rotation (np.ndarray): 3x3 rotation matrix.
        seq (str): Order of rotations for the Euler angles (default 'xyz').

    Returns:
        Tuple[float, float, float]: Euler angles in radians.
    """
    rotation = R.from_matrix(rotation)
    return rotation.as_euler(seq, degrees=False)


def euler_to_rotation(
    euler_angles: Tuple[float, float, float], seq: str = "xyz"
) -> np.ndarray:
    """Convert Euler angles to a rotation matrix.

    Args:
        euler_angles (Tuple[float, float, float]): Euler angles in radians.
        seq (str): Order of rotations for the Euler angles (default 'xyz').

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    rotation = R.from_euler(seq, euler_angles, degrees=False)
    return rotation.as_matrix()


def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Args:
        quaternion (List[float]): Quaternion in the format [qx, qy, qz, qw].

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    rotation = R.from_quat(q)
    return rotation.as_matrix()


def rotation_to_quaternion(rotation: np.ndarray) -> list:
    """
    Convert a 3x3 rotation matrix to a quaternion.

    Args:
        rotation (np.ndarray): 3x3 rotation matrix.

    Returns:
        list: Quaternion in the format [qx, qy, qz, qw].
    """
    rotation_obj = R.from_matrix(rotation)
    return rotation_obj.as_quat().tolist()


def vbr_to_kitti_coordinate_system(poses: np.ndarray) -> np.ndarray:
    """
    Transform a set of poses from the VBR coordinate system (x-forward, y-left, z-up)
    to the KITTI coordinate system (x-right, y-downward, z-forward).

    What it was observed:
        VBR-SLAM: (x-forward, y-left, z-up)
        KITTI: (x-right, y-downward, z-forward)

    Args:
        poses (np.ndarray): Array of shape (N, 7), where each row is a pose in the format
                            [tx, ty, tz, qx, qy, qz, qw].

    Returns:
        np.ndarray: Transformed poses in the same shape (N, 7).
    """
    # Transform translations: (x, y, z) -> (-y, -z, x)
    translations = poses[:, :3]
    translations_new = np.zeros_like(translations)
    translations_new[:, 0] = -translations[:, 1]
    translations_new[:, 1] = -translations[:, 2]
    translations_new[:, 2] = translations[:, 0]

    # Transform rotations
    quaternions = poses[:, 3:]
    rotation_original = R.from_quat(quaternions)

    # Define the transformation rotation:
    # 90º on the Z-axis, then 90º on the X-axis
    rotation_transform = R.from_euler("zx", [90, 90], degrees=True)
    rotation_new = rotation_transform * rotation_original
    quaternions_new = rotation_new.as_quat()

    # Combine translations and rotations
    transformed_poses = np.hstack((translations_new, quaternions_new))
    return transformed_poses


def euroc_to_kitti_coordinate_system(poses: np.ndarray) -> np.ndarray:
    """
    Transform a set of poses from one coordinate system to another.

    What it was observed:
        EuRoC: (x-right, y-forward, z-up)
        KITTI: (x-right, y-downward, z-forward)

    Args:
        poses (np.ndarray): Array of shape (N, 7), where each row is a pose in the format
                            [tx, ty, tz, qx, qy, qz, qw].

    Returns:
        np.ndarray: Transformed poses in the same shape (N, 7).
    """
    # Transform translations (x, y, z) -> (x, -z, y)
    translations = poses[:, :3]
    translations_new = np.zeros_like(translations)
    translations_new[:, 0] = translations[:, 0]
    translations_new[:, 1] = -translations[:, 2]
    translations_new[:, 2] = translations[:, 1]

    # Transform rotations
    quaternions = poses[:, 3:]
    rotation_original = R.from_quat(quaternions)

    # Define the transformation rotation: 90 degrees on the X-axis
    rotation_transform = R.from_euler("x", 90, degrees=True)
    rotation_new = rotation_transform * rotation_original
    quaternions_new = rotation_new.as_quat()

    # Combine translations and rotations
    transformed_poses = np.hstack((translations_new, quaternions_new))
    return transformed_poses


def seven_scenes_to_kitti_coordinate_system(gt_poses: np.ndarray) -> np.ndarray:
    """
    What it was observed:
        7Scenes: (x-right, y-down, z-forward)
        KITTI: (x-right, y-down, z-forward)

    Args:
        gt_poses (np.ndarray): A 2D array of shape (N, 16) representing the poses in (x, y, z).

    Returns:
        np.ndarray: A array in the KITTI z-forward format.
    """
    # Initialize an empty array for transformed data
    kitti_poses = np.copy(gt_poses)

    # Apply the transformation
    kitti_poses[:, 3] = gt_poses[:, 3]
    kitti_poses[:, 7] = gt_poses[:, 7]
    kitti_poses[:, 11] = gt_poses[:, 11]

    return kitti_poses


def tum_to_kitti_coordinate_system(gt_poses: np.ndarray) -> np.ndarray:
    """
    What it was observed:
        TUM-RGBD: (x-, y-, z-)
        KITTI: (x-right, y-down, z-forward)

    Args:
        gt_poses (np.ndarray): A 2D array of shape (N, 16) representing the poses in (x, y, z).

    Returns:
        np.ndarray: A array in the KITTI z-forward format.
    """
    # Initialize an empty array for transformed data
    kitti_poses = np.copy(gt_poses)

    # Apply the transformation
    # kitti_poses[:, 0] = gt_poses[:, 1]
    # kitti_poses[:, 1] = -gt_poses[:, 2]
    # kitti_poses[:, 2] = -gt_poses[:, 0]

    kitti_poses[:, 0] = gt_poses[:, 0]
    kitti_poses[:, 1] = -gt_poses[:, 2]
    kitti_poses[:, 2] = gt_poses[:, 1]

    return kitti_poses


def load_normalization(stats_file: str, dataset_name: str):
    """
    Loads normalization parameters from a JSON file for the specified dataset.

    Args:
        stats_file (str): Path to the JSON file.
        dataset_name (str): Name of the dataset to retrieve parameters from (e.g., "kitti", "7scenes").

    Returns:
        mean_angles (np.array): Mean angles for the dataset.
        std_angles (np.array): Standard deviation of angles for the dataset.
        mean_t (np.array): Mean translation for the dataset.
        std_t (np.array): Standard deviation of translation for the dataset.
    """
    # Read the JSON file
    with open(stats_file, "r") as f:
        data = json.load(f)

    # Ensure the dataset_name exists in the JSON data
    if dataset_name not in data:
        raise ValueError(f"Dataset '{dataset_name}' not found in the JSON file.")

    # Extract normalization parameters
    dataset_params = data[dataset_name]
    mean_angles = np.array(dataset_params["mean_angles"])
    std_angles = np.array(dataset_params["std_angles"])
    mean_t = np.array(dataset_params["mean_t"])
    std_t = np.array(dataset_params["std_t"])

    return mean_angles, std_angles, mean_t, std_t


def load_rgb_normalization(
    stats_file: str, dataset_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads normalization parameters from a JSON file for the specified dataset.

    Args:
        stats_file (str): Path to the JSON file.
        dataset_name (str): Name of the dataset to retrieve parameters from (e.g., "kitti", "7scenes").

    Returns:
        mean_rgb (np.ndarray): Mean RGB values for image normalization.
        std_rgb (np.ndarray): Standard deviation of RGB values for image normalization.
    """
    # Read the JSON file
    with open(stats_file, "r") as f:
        data = json.load(f)

    # Ensure the dataset_name exists in the JSON data
    if dataset_name not in data:
        raise ValueError(f"Dataset '{dataset_name}' not found in the JSON file.")

    # Extract normalization parameters
    dataset_params = data[dataset_name]

    # Extract mean and std RGB values
    mean_rgb = np.array(dataset_params.get("mean_rgb", [0.0, 0.0, 0.0]))
    std_rgb = np.array(dataset_params.get("std_rgb", [1.0, 1.0, 1.0]))

    return mean_rgb, std_rgb


def read_7scenes_split(split_fpath: Union[Path, str]) -> list:
    """
    Reads sequence split file for the 7 Scenes dataset.

    Args:
        split_fpath (Path): Path to split file.

    Returns:
        List[int]: List of sequence numbers.
    """
    with open(split_fpath, "r") as f:
        sequences = [int(l.split("sequence")[-1]) for l in f if not l.startswith("#")]
    return sequences


def visualize_trajectory_with_frames(
    frames_fpath, gt_poses, sequence="Sequence", num_frames=50, delay=0.1, scale=0.1
):
    """
    Visualizes the trajectory alongside frames in real-time.
    Parameters:
        frames_fpath: List with paths to all frames
        gt_poses: Ground truth poses in (N, 3) format
        sequence: Sequence name for display
        num_frames: Number of frames to visualize
        delay: Time delay between frame updates in seconds
        scale: scale to resize the orientation axis
    """
    # Initialize the plot with 1 row, 2 columns
    fig, (ax_img, ax_traj) = plt.subplots(1, 2, figsize=(15, 5))
    img_handle = ax_img.imshow(np.zeros((400, 600, 3)))  # Blank image to start
    ax_img.axis("off")

    # Get translation and rotation from ground truth and center it at the origin
    t = gt_poses[:num_frames, :3]
    t = t - t[0, :]
    r = gt_poses[:num_frames, 3:]  # Rotation (Euler angles)

    # Set up the 3D trajectory plot
    ax_traj = fig.add_subplot(122, projection="3d")
    ax_traj.set_title(f"{sequence} Trajectory")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("z [m]")
    ax_traj.set_zlabel("y [m]")
    ax_traj.grid(True)
    ax_traj.set_xlim([min(t[:, 0]), max(t[:, 0])])
    ax_traj.set_ylim([min(t[:, 2]), max(t[:, 2])])
    ax_traj.set_zlim([min(-t[:, 1]), max(-t[:, 1])])

    # Keep track of the orientation arrows to remove them
    orientation_handles = []

    # Loop through frames and update both the image, trajectory, and orientation axes
    for i in range(num_frames):
        # Update image
        img = Image.open(frames_fpath[i]).convert("RGB")
        img_handle.set_data(img)

        # Update trajectory
        ax_traj.plot3D(t[: i + 1, 0], t[: i + 1, 2], -t[: i + 1, 1], "k")

        # Remove previous orientation axes
        while orientation_handles:
            handle = orientation_handles.pop()
            handle.remove()

        # Compute and plot orientation axes for the current pose
        if i < len(r):
            if len(r[i]) > 3:
                rot_matrix = quaternion_to_rotation_matrix(r[i])
            else:
                rot_matrix = euler_to_rotation(r[i])

            # Define local axes
            origin = t[i]
            x_axis = origin + rot_matrix[:, 0] * scale
            y_axis = origin + rot_matrix[:, 1] * scale
            z_axis = origin + rot_matrix[:, 2] * scale

            # Plot axes
            (x_handle,) = ax_traj.plot(
                [origin[0], x_axis[0]],
                [origin[2], x_axis[2]],
                [-origin[1], -x_axis[1]],
                "r",
            )
            (y_handle,) = ax_traj.plot(
                [origin[0], y_axis[0]],
                [origin[2], y_axis[2]],
                [-origin[1], -y_axis[1]],
                "g",
            )
            (z_handle,) = ax_traj.plot(
                [origin[0], z_axis[0]],
                [origin[2], z_axis[2]],
                [-origin[1], -z_axis[1]],
                "b",
            )

            # Store handles to remove later
            orientation_handles.extend([x_handle, y_handle, z_handle])

        plt.pause(delay)  # Pause to create the real-time effect
        plt.draw()  # Draw the updated frame and trajectory

    plt.show()


def visualize_sequence_trajectory(
    gt_poses: np.ndarray, sequence: str, depth: bool = False
) -> None:
    """
    Visualizes athe ground truth trajectory of a sequence.
    """
    # Get translation from ground truth
    if len(gt_poses[0]) > 7:
        t = gt_poses[:, [3, 7, 11]]
    else:
        t = gt_poses[:, :3]
    # t = t - t[0, :]

    # Plot 2D trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(t[:, 0], t[:, 2])
    plt.title("2D Trajectory")
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.grid()
    plt.title(sequence)
    plt.show()

    # Plot 3D trajectory
    if depth:
        plt.figure(figsize=(10, 6))
        ax = plt.axes(projection="3d")
        ax.plot3D(t[:, 0], t[:, 2], -t[:, 1], "k")
        ax.set_title(f"{sequence}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_zlabel("y [m]")
        ax.grid(True)
        plt.show()


def visualize_samples(dataset: Dataset, num_samples: int = 5):
    """
    Visualizes a few image samples and plots their 2D and 3D trajectory.
    """
    # Get the first window of data
    imgs, _ = dataset[0]  # Get the first window

    # Ensure we don't try to visualize more samples than available
    window_size = imgs.shape[1]  # Number of images in the window
    num_samples = min(num_samples, window_size)  # Update num_samples to available count

    # Visualize image samples
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for i in range(num_samples):
        img = (
            imgs[:3, i, :, :].numpy().transpose(1, 2, 0)
        )  # Change shape from (C, H, W) to (H, W, C)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"Sample {i+1}")

    plt.show()
    plt.close()


def get_scene_sequence_pairs(config, test_sequences):
    pairs = []

    if test_sequences:
        pairs = [(None, seq) for seq in test_sequences]  # Create (None, sequence) pairs
    else:
        scenes = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
        for scene in scenes:
            split_fpath = Path(config["data"]["data_dpath"]) / scene / "TestSplit.txt"
            sequences = read_7scenes_split(split_fpath)
            sequences = ["{:02d}".format(seq) for seq in sequences]
            for seq in sequences:
                pairs.append((scene, seq))  # Create (scene, sequence) pairs
    return pairs


def save_depth_map(
    output_path: Union[str, Path],
    filename: str,
    depth_tensor: torch.Tensor,
    use_colormap: bool = True,
    colormap: str = "Spectral_r",
) -> str:
    """
    Saves a depth map tensor as an image.

    Args:
        output_path (Union[str, Path]): Directory where the image will be saved.
        filename (str): Name of the image file (without extension).
        depth_tensor (torch.Tensor): Depth map tensor (H, W).
        use_colormap (bool): Whether to apply a colormap for visualization (default: True).
        colormap (str): Matplotlib colormap name. [Ex: Spectral_r, magma].

    Returns:
        str: Full path of the saved image.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_numpy = depth_tensor.cpu().numpy().squeeze()
    image_path = output_dir / f"{filename}.png"

    if use_colormap:
        plt.imshow(depth_numpy, cmap=colormap)
        plt.colorbar()
        plt.axis("off")
        plt.savefig(image_path, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        depth_normalized = cv2.normalize(
            depth_numpy, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        cv2.imwrite(str(image_path), depth_normalized)

    return str(image_path)


def save_predicted_depths(
    pred_depths: torch.Tensor,
    save_dpath: Union[str, Path],
    sequence: str,
    scene: str,
    use_colormap: bool = True,
    colormap: str = "Spectral_r",
    num_to_save: int = None,
) -> None:
    """
    Saves the first M predicted depth maps from a tensor.

    Args:
        pred_depths (torch.Tensor): Tensor of shape (N, H, W) with predicted depths.
        save_dpath (Union[str, Path]): Directory to save images.
        sequence (str): Sequence identifier.
        scene (str): Scene identifier (e.g., for 7Scenes).
        use_colormap (bool): Whether to apply a colormap.
        colormap (str): Name of the matplotlib colormap.
        num_to_save (int, optional): Number of depth maps to save. If None, saves all.
    """
    save_dpath = Path(save_dpath) / "predicted_depths"
    save_dpath.mkdir(parents=True, exist_ok=True)

    N = pred_depths.shape[0]
    M = num_to_save if num_to_save is not None else N
    M = min(M, N)  # prevent out-of-bounds

    for i in range(M):
        fname = f"pred_depth_{scene}_{sequence}_{i:05d}"
        save_depth_map(
            output_path=save_dpath / colormap,
            filename=fname,
            depth_tensor=pred_depths[i],
            use_colormap=use_colormap,
            colormap=colormap,
        )
