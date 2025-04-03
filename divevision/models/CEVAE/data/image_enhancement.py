from typing import List, Union

from src.data.base import ImagePairDatasetFromPaths
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.target = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example


class DatasetTrainFromImageFileList(BaseDataset):
    def __init__(
        self,
        size,
        training_images_list_file: list | str,
        target_images_list_file: list | None,
        random_crop: bool = True,
        random_flip: bool = True,
        max_size: int | None = None,
        color_jitter: dict | None = None,
        perspective: dict | None = None,
        scale: dict | None = None,
    ):
        super().__init__()
        if isinstance(training_images_list_file, str):
            training_images_list_file = [training_images_list_file]
        paths = []
        for training_list_file in training_images_list_file:
            with open(training_list_file, "r") as f:
                paths += f.read().splitlines()

        if isinstance(target_images_list_file, str):
            target_images_list_file = [target_images_list_file]
        paths_t = []
        for target_list_file in target_images_list_file:
            with open(target_list_file, "r") as f:
                paths_t += f.read().splitlines()
        self.data = ImagePairDatasetFromPaths(
            paths=paths,
            path_target=paths_t,
            size=size,
            random_crop=random_crop,
            random_flip=random_flip,
            max_size=max_size,
            color_jitter=color_jitter,
            perspective=perspective,
            scale=scale,
        )


class DatasetTestFromImageFileList(BaseDataset):
    def __init__(self, size, test_images_list_file, test_target_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        with open(test_target_images_list_file, "r") as f:
            paths_t = f.read().splitlines()
        self.data = ImagePairDatasetFromPaths(
            paths=paths, path_target=paths_t, size=size, random_crop=False, is_test=True
        )
