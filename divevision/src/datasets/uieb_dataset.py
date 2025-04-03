from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision


class UIEBDataset(Dataset):
    """IUEB dataset PyTorch Dataset implementation. More information are found on the project page (https://li-chongyi.github.io/proj_benchmark.html).

    This dataset should only be used for academic purposes."""

    dataset_name = "UIEB"

    def __init__(
        self,
        root_dir: str = "divevision/data/UIEB/",
        transform=None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.data = self.load_data()

    def load_data(self) -> tuple[list[Path], list[Path]]:
        """Initialize the dataset and return a tuple containing two lists, respectively the paths to the raw images, and the paths to the target images."""
        dataset_path = Path(self.root_dir)
        inputs_path = dataset_path.joinpath("raw-890")
        labels_path = dataset_path.joinpath("reference-890")
        # Check that subdirectories exists
        assert (
            inputs_path.is_dir() and labels_path.is_dir()
        ), "Subdirectories 'raw-890' and 'reference-890' must exist in the dataset directory."

        # Check that subdirectories contain the same number of images
        assert len(list(inputs_path.iterdir())) == len(
            list(labels_path.iterdir())
        ), "The two subdirectories must contain the same number of images"

        inputs_filepaths = sorted([str(x) for x in inputs_path.glob("*.png")])
        labels_filepaths = sorted([str(x) for x in labels_path.glob("*.png")])

        return inputs_filepaths, labels_filepaths

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        input_filepath, label_filepath = self.data[0][idx], self.data[1][idx]
        # Load the images as PIL images
        input_img, label_img = Image.open(input_filepath), Image.open(label_filepath)
        # Apply the transforms to both images
        if self.transform is not None:
            input = self.transform(input_img)
            label = self.transform(label_img)
        else:  # If no transform is provided, convert the images to tensors
            input = torchvision.transforms.ToTensor()(input_img)
            label = torchvision.transforms.ToTensor()(label_img)

        return input, label
