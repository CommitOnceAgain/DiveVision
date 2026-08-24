from pathlib import Path
from typing import Self

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset

from divevision.src.datasets.abstract_dataset import AbstractDataset


class LSUIDataset(Dataset, AbstractDataset):
    """LSUI Benchmark Dataset."""

    name = "LSUI"

    def __init__(
        self,
        root_dir: str = "divevision/data/LSUI/",
        transform=None,
    ):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.data = self.load_data()

    def load_data(self) -> tuple[list[str], list[str]]:
        """Initialize the LSUI Benchmark Dataset and return the file paths to the Degraded Images and the corresponding Reference Images as a list of Path objects."""
        dataset_path = Path(self.root_dir)
        reference_dir = dataset_path.joinpath("GT")
        degraded_dir = dataset_path.joinpath("input")
        # Check that there are two subdirectories in root_dir called "GT" and "input"
        assert (
            reference_dir.is_dir() and degraded_dir.is_dir()
        ), "The root directory must contain two subdirectories called 'GT' and 'input'"

        # Check that there are the same number of images in both directories
        assert len(list(degraded_dir.iterdir())) == len(
            list(reference_dir.iterdir())
        ), "The two subdirectories must contain the same number of images"

        degraded_filepaths = sorted([str(x) for x in degraded_dir.glob("*.jpg")])
        reference_filepaths = sorted([str(x) for x in reference_dir.glob("*.jpg")])

        return degraded_filepaths, reference_filepaths

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data[0])

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        degraded_filepath, reference_filepath = self.data[0][idx], self.data[1][idx]

        # Load the images as PIL images
        degraded_image, reference_image = (
            Image.open(degraded_filepath),
            Image.open(reference_filepath),
        )

        if self.transform is not None:
            degraded_image = self.transform(degraded_image)
            reference_image = self.transform(reference_image)
        else:  # If no transform is provided, convert the images to tensors
            degraded_image = torchvision.transforms.ToTensor()(degraded_image)
            reference_image = torchvision.transforms.ToTensor()(reference_image)

        return degraded_image, reference_image
