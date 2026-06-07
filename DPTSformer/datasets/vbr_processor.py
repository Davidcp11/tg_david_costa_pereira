from pathlib import Path
import pickle
import os
from datetime import datetime

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class IMUDataProcessor:
    def __init__(self, data_dpath):
        """
        Initializes the IMUDataProcessor with the path to the data directory.

        Parameters:
        data_dpath (str or Path): Directory path where data files are located.
        """
        self.data_dpath = Path(data_dpath)

    def read_timestamps(self, file_fpath):
        """
        Reads timestamps from a file and converts them to datetime objects.

        Parameters:
        file_fpath (str or Path): Path to the timestamps file.

        Returns:
        List[datetime]: List of datetime objects representing the timestamps.
        """
        with open(file_fpath, "r") as f:
            timestamps = f.read().strip().splitlines()
        return [
            datetime.fromisoformat(ts[:26]) for ts in timestamps
        ]  # Truncate to microseconds

    def extract_position_and_rotation(self, file_fpath, dt=0.01):
        """
        Extracts position and Euler angles from IMU data.

        Parameters:
        file_fpath (str or Path): Path to the 'imu.txt' file.
        dt (float): Time step for integration.

        Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Positions and Euler angles.
        """
        data = pd.read_csv(file_fpath)

        position = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.0, 0.0, 0.0])
        positions = []
        euler_angles = []

        for index, row in data.iterrows():
            acc = np.array([row["acc_x"], row["acc_y"], row["acc_z"]])
            velocity += acc * dt
            position += velocity * dt
            positions.append(position.copy())
            quat = np.array(
                [row["quat_w"], row["quat_x"], row["quat_y"], row["quat_z"]]
            )
            euler = self.quaternion_to_euler_angles(quat)
            euler_angles.append(euler)

        return np.array(positions), np.array(euler_angles)

    @staticmethod
    def quaternion_to_euler_angles(quat):
        """
        Converts a quaternion to Euler angles.

        Parameters:
        quat (numpy.ndarray): Array containing the quaternion [w, x, y, z].

        Returns:
        numpy.ndarray: Array of Euler angles [roll, pitch, yaw].
        """
        w, x, y, z = quat
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = np.arcsin(2 * (w * y - z * x))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return np.array([roll, pitch, yaw])

    def synchronize_data(self, output_fpath):
        """
        Synchronizes camera and IMU data and saves it to a CSV file.

        Parameters:
        output_fpath (str or Path): Path to the output CSV file.
        """
        camera_timestamps = self.read_timestamps(
            self.data_dpath / "camera_left/timestamps.txt"
        )
        imu_timestamps = self.read_timestamps(self.data_dpath / "imu/timestamps.txt")

        imu_data_path = self.data_dpath / "imu/data/imu.txt"
        positions, euler_angles = self.extract_position_and_rotation(imu_data_path)

        output_data = []

        for cam_ts in tqdm(camera_timestamps, desc="Synchronizing"):
            closest_idx = min(
                range(len(imu_timestamps)),
                key=lambda i: abs((imu_timestamps[i] - cam_ts).total_seconds()),
            )
            pos = positions[closest_idx]
            euler = euler_angles[closest_idx]
            image_name = (
                f"{len(output_data):010d}.png"  # Assuming filenames are sequential
            )
            row = [cam_ts.isoformat(), image_name] + pos.tolist() + euler.tolist()
            output_data.append(row)

        columns = ["timestamp", "image_name", "x", "y", "z", "roll", "pitch", "yaw"]
        df = pd.DataFrame(output_data, columns=columns)
        df.to_csv(output_fpath, index=False)


def plot_2d_trajectory(positions):
    """
    Plots the 2D trajectory from the given positions.

    Parameters:
    positions (numpy.ndarray): An array of shape (n, 3) where n is the number of time steps,
                                and the first two columns correspond to x and y coordinates.
    """
    x = positions[:, 0]
    y = positions[:, 1]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Trajectory", color="b")
    plt.title("2D Trajectory from IMU Data")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)
    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y))
    plt.show()


# Example usage
if __name__ == "__main__":
    data_dpath = Path(
        "D:/Doutorado/Codes/VisualOdometry/dataset/vbr_slam/campus_test1_03_kitti"
    )
    output_dpath = data_dpath / "synchronnized_data_v2.csv"

    processor = IMUDataProcessor(data_dpath)
    processor.synchronize_data(output_dpath)

    # Optionally, plot the trajectory
    t, _ = processor.extract_position_and_rotation(
        data_dpath / "imu/data/imu.txt"
    )
    plot_2d_trajectory(t)
