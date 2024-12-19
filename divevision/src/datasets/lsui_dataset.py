from pathlib import Path
from typing import Self
import torch
from torch.utils.data import Dataset
from PIL import Image


class LSUIDataset(Dataset):
    """LSUI dataset."""

    def __init__(
        self,
        root_dir: str = "divevision/data/LSUI/",
        transform=None,
    ) -> Self:
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = self.init_lsui_dataset()

    def init_lsui_dataset(self) -> tuple[list[Path], list[Path]]:
        """Initialize the LSUI dataset and return the file paths to the images and the corresponding ground truth images as a list of Path objects."""
        dataset_path = Path(self.root_dir)
        labels_dir = dataset_path.joinpath("GT")
        inputs_dir = dataset_path.joinpath("input")
        # Check that there are two subdirectories in root_dir called "GT" and "input"
        assert (
            labels_dir.is_dir() and inputs_dir.is_dir()
        ), "The root directory must contain two subdirectories called 'GT' and 'input'"

        # Check that there are the same number of images in both directories
        labels_filepaths = list(labels_dir.glob("*.jpg"))
        inputs_filepaths = list(inputs_dir.glob("*.jpg"))
        assert len(labels_filepaths) == len(
            inputs_filepaths
        ), "The two subdirectories must contain the same number of images"

        return inputs_filepaths, labels_filepaths

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data[0])

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        input_filepath, label_filepath = self.data[0][idx], self.data[1][idx]

        # Load the images as PIL images
        input_img, label_img = Image.open(input_filepath), Image.open(label_filepath)

        if self.transform is not None:
            input = self.transform(input_img)
            label = self.transform(label_img)

        return input, label
