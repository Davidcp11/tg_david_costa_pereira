from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch.utils.data import random_split, DataLoader
from datasets.transforms import get_transforms

from datasets.kitti import KITTIDataset
from datasets.seven_scenes import SevenScenesDataset
from datasets.vbr_slam import VBRDataset
from datasets.euroc import EuRoCDataset


def define_sequences(
    sequence: Optional[str], split: str, training_sequences: list, test_sequences: list
) -> list:
    """
    Define the list of sequences to be used for creating a dataset.

    Args:
        sequence (Optional[str]): A specific sequence to use. If provided, it overrides
            the `split` parameter and directly defines the sequence.
        split (str): Dataset split type, either "train" or "test".
        training_sequences (list): List of sequences for training.
        test_sequences (list): List of sequences for testing.
    Returns:
        list: A list of sequences to be used for the dataset.

    Raises:
        ValueError: If `split` is not "train" or "test".
    """
    if sequence:
        sequences = [sequence]
    else:
        if split == "train":
            sequences = training_sequences
        else:
            sequences = test_sequences
    return sequences


def build_dataloader(
    data_params: dict,
    split: str,
    sequence: Optional[str] = None,
    scene: Optional[str] = None,
) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    """
    Builds and returns the DataLoader(s) for the specified dataset and split.

    Args:
        data_params (dict): Dictionary containing the dataset configuration and parameters.
        split (str): The dataset split, either 'train', 'val', or 'test'. Defaults to 'train'.
        sequence (Optional[str]): Sequence ID to load specific sequences. Defaults to None.
        scene (Optional[str]): Scene name for datasets like 7Scenes. Defaults to None.

    Returns:
        Union[Tuple[DataLoader, DataLoader], DataLoader]:
        A tuple of (train_loader, val_loader) for 'train', or a test_loader for 'test'.
    """
    # Extract parameters
    dataset_name = data_params["dataset"]
    data_dpath = data_params["data_dpath"]
    image_size = data_params["image_size"]
    window_size = data_params["window_size"]
    overlap = data_params["overlap"]
    batch_size = data_params["bsize"]
    num_workers = data_params["num_workers"]
    normalize_gt = data_params["normalize_gt"]
    dataset_name_norm = data_params.get("dataset_name_norm", "kitti")
    training_sequences = data_params.get("training_sequences", [])
    test_sequences = data_params.get("test_sequences", [])
    estimate_depth = data_params.get("estimate_depth", False)
    depth_clip_max = data_params.get("depth_clip_max", None)

    # Get transforms based on the dataset
    transforms = get_transforms(dataset_name, image_size, dataset_name_norm)

    # Define sequences to create dataset
    sequences = define_sequences(sequence, split, training_sequences, test_sequences)

    if dataset_name == "kitti":
        dataset = KITTIDataset(
            data_dpath=data_dpath,
            sequences=sequences,
            window_size=window_size,
            overlap=overlap,
            transforms=transforms,
            normalize_gt=normalize_gt,
            estimate_depth=estimate_depth,
            depth_clip_max=depth_clip_max,
        )

    elif dataset_name == "euroc":
        dataset = EuRoCDataset(
            data_dpath=data_dpath,
            sequences=sequences,
            window_size=window_size,
            overlap=overlap,
            transforms=transforms,
            normalize_gt=normalize_gt,
            dataset_name_norm=dataset_name_norm,
        )

    elif dataset_name == "7scenes":
        all_scenes = [
            "chess",
            "fire",
            "heads",
            "office",
            "pumpkin",
            "redkitchen",
            "stairs",
        ]

        transforms, transforms_depth = transforms
        dataset = SevenScenesDataset(
            data_dpath=Path(data_dpath),
            split=split,
            scenes=[scene] if scene else all_scenes,
            sequences=sequences,
            window_size=window_size,
            overlap=overlap,
            read_depth=data_params.get("read_depth", False),
            transforms=transforms,
            transforms_depth=transforms_depth,
            normalize_gt=normalize_gt,
        )

    elif dataset_name == "vbr_slam":
        dataset = VBRDataset(
            data_dpath=Path(data_dpath),
            sequences=sequences,
            window_size=window_size,
            overlap=overlap,
            transforms=transforms,
            normalize_gt=normalize_gt,
            estimate_depth=estimate_depth,
        )

    else:
        raise ValueError(f"--- Undefined dataset: {dataset_name} ---")

    # Create validation data for 'train' split
    if split == "train":
        val_split = data_params.get("val_split", 0.1)
        nb_samples_val = round(val_split * len(dataset))

        # Set the random seed for reproducibility
        generator = torch.Generator().manual_seed(2)
        train_data, val_data = random_split(
            dataset,
            [len(dataset) - nb_samples_val, nb_samples_val],
            generator=generator,
        )

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader

    elif split == "test":
        test_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return test_loader
