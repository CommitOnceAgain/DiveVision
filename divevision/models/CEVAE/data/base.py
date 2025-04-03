from typing import Any

import albumentations
import cv2
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _load_img(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    array = np.array(image).astype(np.uint8)
    return array


class ImagePairDatasetFromPaths(Dataset):
    def __init__(
        self,
        paths: list[str],
        path_target: str,
        size: int | None = None,
        random_crop: bool = False,
        labels: dict | None = None,
        random_flip: bool = False,
        max_size: int | None = None,
        color_jitter: dict | None = None,
        perspective: dict | None = None,
        scale: dict | None = None,
        is_test: bool = False,
    ):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self.labels["file_path_target"] = path_target
        self._length = len(paths)
        self.is_test = is_test

        if self.size is not None and self.size > 0:
            if not is_test:
                self.rescaler = albumentations.SmallestMaxSize(
                    max_size=self.size if max_size is None else max_size
                )
                if scale is not None:
                    self.rescaler = albumentations.Compose(
                        [self.rescaler, albumentations.RandomScale(p=1.0, **scale)]
                    )

                if not self.random_crop:
                    self.cropper = albumentations.CenterCrop(
                        height=self.size, width=self.size
                    )
                else:
                    self.cropper = albumentations.RandomCrop(
                        height=self.size, width=self.size
                    )

                augmentations = [self.rescaler, self.cropper]
                if random_flip:
                    augmentations.append(albumentations.HorizontalFlip(p=0.5))
                if color_jitter is not None:
                    augmentations.append(
                        albumentations.ColorJitter(p=0.5, **color_jitter)
                    )
                if perspective is not None:
                    augmentations.append(
                        albumentations.Perspective(p=0.5, **perspective)
                    )
            else:
                augmentations = [
                    albumentations.Resize(
                        height=self.size,
                        width=self.size,
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                ]

            self.preprocessor = albumentations.Compose(
                augmentations,
                additional_targets={"image_target": "image"},
                is_check_shapes=False,
            )
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self) -> int:
        return self._length

    def preprocess_image(self, image_path: str) -> np.ndarray:
        image = _load_img(image_path)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def preprocess_images(
        self, image_path: str, target_path: str
    ) -> tuple[np.ndarray, np.ndarray]:
        image = _load_img(image_path)
        target = _load_img(target_path)

        trasnformed_images = self.preprocessor(image=image, image_target=target)

        image = (trasnformed_images["image"] / 127.5 - 1.0).astype(np.float32)
        target = (trasnformed_images["image_target"] / 127.5 - 1.0).astype(np.float32)
        return image, target

    def __getitem__(self, i: int) -> dict[str, Any]:
        example = dict()
        example["image"], example["target"] = self.preprocess_images(
            self.labels["file_path_"][i], self.labels["file_path_target"][i]
        )
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example
