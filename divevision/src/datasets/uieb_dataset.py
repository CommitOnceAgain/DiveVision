from pathlib import Path

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset

from divevision.src.datasets.abstract_dataset import AbstractDataset


class UIEBDataset(Dataset, AbstractDataset):
    """UIEB Benchmark Dataset PyTorch Dataset implementation. More information are found on the project page (https://li-chongyi.github.io/proj_benchmark.html).

    This dataset should only be used for academic purposes."""

    name = "UIEB"

    def __init__(
        self,
        root_dir: str = "divevision/data/UIEB/",
        transform=None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.data = self.load_data()

    def load_data(self) -> tuple[list[str], list[str]]:
        """Initialize the UIEB Benchmark Dataset and return a tuple containing two lists, respectively the paths to the Degraded Images, and the paths to the Reference Images."""
        dataset_path = Path(self.root_dir)
        degraded_dir = dataset_path.joinpath("raw-890")
        reference_dir = dataset_path.joinpath("reference-890")
        # Check that subdirectories exists
        assert (
            degraded_dir.is_dir() and reference_dir.is_dir()
        ), "Subdirectories 'raw-890' and 'reference-890' must exist in the dataset directory."

        # Check that subdirectories contain the same number of images
        assert len(list(degraded_dir.iterdir())) == len(
            list(reference_dir.iterdir())
        ), "The two subdirectories must contain the same number of images"

        degraded_filepaths = sorted([str(x) for x in degraded_dir.glob("*.png")])
        reference_filepaths = sorted([str(x) for x in reference_dir.glob("*.png")])

        return degraded_filepaths, reference_filepaths

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        degraded_filepath, reference_filepath = self.data[0][idx], self.data[1][idx]
        # Load the images as PIL images
        degraded_image, reference_image = (
            Image.open(degraded_filepath),
            Image.open(reference_filepath),
        )
        # Apply the transforms to both images
        if self.transform is not None:
            degraded_image = self.transform(degraded_image)
            reference_image = self.transform(reference_image)
        else:  # If no transform is provided, convert the images to tensors
            degraded_image = torchvision.transforms.ToTensor()(degraded_image)
            reference_image = torchvision.transforms.ToTensor()(reference_image)

        return degraded_image, reference_image
