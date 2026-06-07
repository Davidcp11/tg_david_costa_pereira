from typing import Tuple, Union
from torchvision import transforms
from utils.data_utils import load_rgb_normalization


def get_transforms(
    dataset: str,
    image_size: Tuple[int, int],
    dataset_name_norm: str,
) -> Union[transforms.Compose, Tuple[transforms.Compose, transforms.Compose]]:
    """
    Retrieves the appropriate transform pipeline for a specified dataset.

    Args:
        dataset (str): The name of the dataset for which to retrieve transforms.
        image_size (Tuple[int, int]): Target image size as (height, width) for resizing.
        dataset_name_norm (str): Name of the dataset to read the normalization parameters

    Returns:
        Union[transforms.Compose, Tuple[transforms.Compose, transforms.Compose]]:
            - A single `transforms.Compose` for datasets with only one transformation pipeline.
            - A tuple of `transforms.Compose` for datasets with multiple pipelines (e.g., RGB and depth).
    """
    # Get rgb statistics
    mean_rgb, std_rgb = load_rgb_normalization(
        stats_file="datasets/dataset_stats.json", dataset_name=dataset_name_norm
    )

    # Define transforms for RGB frames
    rgb_transforms = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=mean_rgb,
            #     std=std_rgb,
            # ),
        ]
    )

    if dataset == "7scenes":
        # Define transforms for depth
        depth_transforms = transforms.Compose(
            [transforms.Resize(image_size), transforms.ToTensor()]
        )

        return rgb_transforms, depth_transforms

    else:
        return rgb_transforms
