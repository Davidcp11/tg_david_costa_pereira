import queue
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from datasets.dataloader import build_dataloader
from utils.data_utils import (
    load_normalization,
    euler_to_rotation,
    quaternion_to_rotation_matrix,
    get_scene_sequence_pairs,
)
from utils.config_utils import load_config


def save_trajectory(
    poses: np.ndarray, scene: str, sequence: str, save_dir: Union[str, Path]
):
    """
    Save predicted poses in .txt file
    Args:
        poses (ndarray): list with all 4x4 pose matrix
        scene (str): scene (7scenes)
        sequence (str): sequence number
        save_dir (Union[str, Path]): path to save pose
    """
    save_dir = Path(save_dir)

    # create directory
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{scene}_{sequence}.txt" if scene else f"{sequence}.txt"
    output_filename = save_dir / output_filename
    with open(output_filename, "w") as f:
        for pose in poses:
            pose = pose.flatten()[:12]
            line = " ".join([str(x) for x in pose]) + "\n"
            f.write(line)


def post_processing(pred_poses: np.ndarray, window_size: int) -> np.ndarray:
    """
    Post-processes predicted poses by averaging overlapping poses based on the specified window size.

    Args:
        pred_poses (np.ndarray): Array of shape (N, W, 6) with predicted 6-DoF poses, where N is the number of batches,
                                 W is the number of frames per window, and 6 represents the pose parameters.
        window_size (int): Size of the prediction window; used to determine overlap for averaging.

    Returns:
        np.ndarray: Array of post-processed poses with overlapping frames averaged based on window size.
    """

    # Handle case where there is no overlap
    if window_size == 2:
        pred_poses = pred_poses.squeeze(1)
        return np.asarray(pred_poses)

    num_batchs = pred_poses.shape[0]
    q = queue.Queue(window_size - 1)  # The max size is 5.
    idx = 0
    poses = []

    # Fill the queue with initial frames up to max size (window_size - 1)
    while not q.full():
        q.put(pred_poses[idx, :, :])
        idx = idx + 1

    # Process poses with overlap and averaging
    while idx < num_batchs:
        # Process when queue is full for the first time
        if idx == (window_size - 1):
            poses.append(q.queue[0][0, :])

            # Specific averaging logic for window sizes 3 overlap 2
            avg_pose = (q.queue[0][1, :] + q.queue[1][0, :]) / 2
            poses.append(avg_pose)

            if window_size == 4:
                # implemented for specific case window_size = 4 and overlap = 3
                avg_pose = (q.queue[0][2, :] + q.queue[1][1, :] + q.queue[2][0, :]) / 3
                poses.append(avg_pose)

        elif idx < (num_batchs - 1):
            if window_size == 3:
                # Implemented for specific case window_size = 3 and overlap = 2
                avg_pose = (q.queue[0][1, :] + q.queue[1][0, :]) / 2
                poses.append(avg_pose)

            elif window_size == 4:
                # implemented for specific case window_size = 4 and overlap = 3
                avg_pose = (q.queue[0][2, :] + q.queue[1][1, :] + q.queue[2][0, :]) / 3
                poses.append(avg_pose)

        # Process last full queue (idx == num_batchs-1)
        else:
            if window_size == 3:
                # Implemented for specific case window_size = 3 and overlap = 2
                poses.append(q.queue[1][1, :])

            elif window_size == 4:
                # Implemented for specific case window_size = 4 and overlap = 2
                avg_pose = (q.queue[1][2, :] + q.queue[2][1, :]) / 2
                poses.append(avg_pose)
                poses.append(q.queue[2][2, :])

            idx = idx + 1

        # Update queue with new frames
        if idx < (num_batchs - 1):
            idx = idx + 1
            first = q.get()  # Remove the first element in the queue
            q.put(pred_poses[idx, :, :])

    return np.asarray(poses)


def recover_trajectory_and_poses(
    poses: np.ndarray, normalize_gt: bool, dataset_name: str, T_init: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recovers the predicted trajectory and poses in 6-DoF format, with an option to undo normalization.

    Args:
        poses (np.ndarray): Array of shape (N, 6) containing N poses, where each pose consists of 3 rotation angles
                            and 3 translation values.
        normalize_gt (bool): If True, reverses the normalization for ground truth poses using stored mean and standard
                             deviation values.
        dataset_name (str): Name of the dataset to load the normalization parameters

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Array of 4x4 transformation matrices for each pose.
            - Array of 3D positions (trajectory) extracted from the translation component of each transformation.
    """

    predicted_poses = []
    predicted_trajectory = []

    # recover predicted trajectory
    for i in range(len(poses) - 1):
        if i == 0:
            T = T_init  # np.eye(4)

        angles = poses[i, :3]
        t = poses[i, 3:]

        # Undo normalization
        if normalize_gt:
            mean_angles, std_angles, mean_t, std_t = load_normalization(
                stats_file="datasets/dataset_stats.json", dataset_name=dataset_name
            )
            angles = np.multiply(angles, std_angles) + mean_angles
            t = np.multiply(t, std_t) + mean_t

        # Euler angles to rotation matrix
        rot = euler_to_rotation(angles)

        # Relative transformation matrix
        T_r = np.concatenate(
            (
                np.concatenate([rot, np.reshape(t, (3, 1))], axis=1),
                [[0.0, 0.0, 0.0, 1.0]],
            ),
            axis=0,
        )
        T_abs = np.dot(T, T_r)
        T = T_abs

        predicted_poses.append(T)
        predicted_trajectory.append(T_abs[:3, 3])

    return np.asarray(predicted_poses), np.asarray(predicted_trajectory)


if __name__ == "__main__":

    import argparse

    _parser = argparse.ArgumentParser()
    _parser.add_argument("config_fpath")
    _parser.add_argument("checkpoint_fname")
    _parser.add_argument("--lang", choices=["pt", "en"], default="pt")
    _args = _parser.parse_args()

    EN = _args.lang == "en"

    def t(pt_str, en_str):
        return en_str if EN else pt_str

    config_fpath    = _args.config_fpath
    checkpoint_name = _args.checkpoint_fname

    # Load hyperparameters
    config = load_config(config_fpath)
    checkpoint_fpath = (Path(config["checkpoint"]["checkpoint_dpath"]) / checkpoint_name).with_suffix("")

    # Extract configuration parameters — disable depth for test sequences
    data_params = {**config.get("data", {}), "estimate_depth": False}

    pairs = get_scene_sequence_pairs(config, config["data"].get("test_sequences", None))

    # Predict for each test sequence
    for scene, sequence in pairs:
        # read ground test data and predicted poses
        pred_fname = (
            f"pred_poses_{scene}_{sequence}.npy"
            if scene
            else f"pred_poses_{sequence}.npy"
        )
        pred_path = checkpoint_fpath / data_params["dataset"] / pred_fname
        try:
            pred_poses = np.load(pred_path)
        except:
            continue

        # get ground truth trajectories
        dataloader = build_dataloader(
            data_params, split="test", sequence=sequence, scene=scene
        )
        sequence_name = f"{scene}-{sequence}" if scene else sequence
        gt_poses = dataloader.dataset.data_dict[sequence_name]["ground_truth"]
        if len(gt_poses[0]) == 16:
            gt_trajectory = gt_poses[:, [3, 7, 11]]
            T_init = np.reshape(gt_poses[0, :], (4, 4))

        elif len(gt_poses[0]) == 12:
            gt_trajectory = gt_poses[:, [3, 7, 11]]
            T_init = np.vstack(
                [np.reshape(gt_poses[0, :], (3, 4)), [[0.0, 0.0, 0.0, 1.0]]]
            )
        else:
            gt_trajectory = gt_poses[:, :3]
            t = gt_poses[0, :3]  # Translation (tx, ty, tz)
            angles = gt_poses[0, 3:]

            # Convert angles to rotation matrix
            if angles.size == 3:
                Rot = euler_to_rotation(angles)
            else:
                Rot = quaternion_to_rotation_matrix(angles)

            # Construct a 4x4 transformation matrix
            T_init = np.vstack(
                [np.hstack([Rot, t.reshape(-1, 1)]), [0.0, 0.0, 0.0, 1.0]]
            )

        # gt_trajectory = np.asarray(gt_poses)[:, [3, 7, 11]]
        # start in [0,0]
        # gt_trajectory = gt_trajectory - gt_trajectory[0, :]

        # post processing and recover trajectory
        poses = post_processing(pred_poses, data_params["window_size"])
        pred_poses, pred_trajectory = recover_trajectory_and_poses(
            poses,
            normalize_gt=data_params.get("normalize_gt", True),
            dataset_name=data_params.get("dataset_name_norm", ""),
            T_init=T_init,
        )
        save_trajectory(
            pred_poses,
            scene,
            sequence,
            save_dir=checkpoint_fpath / data_params["dataset"] / "pred_poses",
        )

        # ITA template: textwidth=6.30in. Trajectories: 2 per row at 0.48\textwidth.
        # Save at exactly 0.48*6.30=3.02in so LaTeX includes with no scaling → fonts at 10pt.
        _FS = 10
        plt.rcParams.update({
            'font.size':       _FS,
            'axes.titlesize':  _FS,
            'axes.labelsize':  _FS,
            'xtick.labelsize': _FS - 1,
            'ytick.labelsize': _FS - 1,
            'legend.fontsize': _FS - 1,
        })
        fig, ax = plt.subplots(figsize=(0.48 * 6.30, 3.0))
        ax.plot([p[0] for p in pred_trajectory], [p[2] for p in pred_trajectory], "b",
                label=t("estimado", "estimated"))
        ax.plot([p[0] for p in gt_trajectory], [p[2] for p in gt_trajectory], "r",
                label=t("referência", "reference"))
        ax.grid(True, alpha=0.4)
        ax.set_xlabel(t("Translação na direção x [m]", "Translation in x direction [m]"))
        ax.set_ylabel(t("Translação na direção z [m]", "Translation in z direction [m]"))
        ax.legend()
        fig.tight_layout()

        # create checkpoints folder
        save_dir = checkpoint_fpath / data_params["dataset"] / "plots"
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        save_fname = (
            f"pred_traj_{scene}_{sequence}.pdf"
            if scene
            else f"pred_traj_{sequence}.pdf"
        )
        fig.savefig(save_dir / save_fname)
        if EN:
            tg1en_dir = Path(__file__).resolve().parent.parent.parent / "tg1en" / "Cap4" / "images"
            tg1en_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(tg1en_dir / save_fname)
        else:
            tg1_dir = Path(__file__).resolve().parent.parent.parent / "tg1" / "Cap4" / "images"
            tg1_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(tg1_dir / save_fname)
        plt.close(fig)
