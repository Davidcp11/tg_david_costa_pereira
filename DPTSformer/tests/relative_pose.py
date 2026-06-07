import matplotlib.pyplot as plt
import numpy as np
from datasets.euroc import EuRoCDataset
from utils.data_utils import (
    quaternion_to_rotation_matrix,
    rotation_to_euler,
    euler_to_rotation,
)


def compute_relative_poses(gt_poses: list) -> np.ndarray:
    """
    Compute relative poses between consecutive frames in a window.

    Args:
        gt_poses (np.ndarray): Ground truth poses in the format (tx, ty, tz, qx, qy, qz, qw).

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
        rot = quaternion_to_rotation_matrix(q)

        # Construct a 4x4 transformation matrix
        pose = np.vstack([np.hstack([rot, t.reshape(-1, 1)]), [0.0, 0.0, 0.0, 1.0]])

        if pose_prev is not None:
            # Compute the relative transformation between the current and previous poses
            pose_wrt_prev = np.dot(np.linalg.inv(pose_prev), pose)
            rot = pose_wrt_prev[:3, :3]
            t = pose_wrt_prev[:3, 3]
            angles = rotation_to_euler(rot)

            # row, pitch, yaw, tx, ty, tz (z-forward)
            y.append(list(angles) + list(t))

        pose_prev = pose

    return np.asarray(y)


def recover_poses(poses: np.ndarray, init: np.ndarray = None) -> np.ndarray:
    """
    Recovers the poses in 6-DoF format

    Args:
        poses (np.ndarray): Array of shape (N, 6) containing N poses, where each pose consists of 3 rotation angles
                            and 3 translation values.

    Returns:
        np.ndarray: Array with 6-DoF poses.
    """
    if init is None:
        init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    t = init[:3]  # Translation (tx, ty, tz)
    q = init[3:]  # Quaternion (qx, qy, qz, qw)

    # Convert quaternion to a rotation matrix
    rot = quaternion_to_rotation_matrix(q)

    # Construct a 4x4 transformation matrix
    T = np.vstack([np.hstack([rot, t.reshape(-1, 1)]), [0.0, 0.0, 0.0, 1.0]])

    y = [list(t) + list(rotation_to_euler(rot))]

    # recover predicted trajectory
    for i in range(len(poses)):
        angles = poses[i, :3]
        t = poses[i, 3:]

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

        # Extract t and R
        t = T[:3, 3]
        rot = T[:3, :3]
        angles = rotation_to_euler(rot)

        # row, pitch, yaw, tx, ty, tz (z-forward)
        y.append(list(angles) + list(t))

    return np.asarray(y)


if __name__ == "__main__":
    ## EuRoC
    sequence = "MH_01_easy"
    dataset = EuRoCDataset(
        data_dpath="G:/data/euroc",
        sequences=[sequence],
        window_size=3,
        overlap=2,
    )

    # Get ground truth
    gt_poses = dataset.data_dict[sequence]["ground_truth"]

    # Compute relative poses
    y = compute_relative_poses(gt_poses)

    # Recover ground truth poses
    y_pred = recover_poses(y, gt_poses[0, :])

    # Convert gt_poses to euler angles
    gt = []
    for p in gt_poses:
        t = p[:3]
        q = p[3:]
        rot = quaternion_to_rotation_matrix(q)
        angles = rotation_to_euler(rot)
        gt.append(list(t) + list(angles))
    gt = np.asarray(gt)

    plt.figure()
    plt.subplot(1, 4, 1)
    plt.plot(gt[:, 0], gt[:, 2], "k")
    plt.title("ground truth")
    plt.subplot(1, 4, 2)
    plt.plot(y_pred[:, 3], y_pred[:, 5], "r")
    plt.title("recovered")
    plt.subplot(1, 4, 3)
    plt.plot(gt[:, 0], gt[:, 2], "k")
    plt.plot(y_pred[:, 3], y_pred[:, 5], "r")
    plt.subplot(1, 4, 4)
    plt.plot(gt[:, 0] - y_pred[:, 3], gt[:, 2] - y_pred[:, 5], "b")
    plt.show()
