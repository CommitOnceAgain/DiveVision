from typing import Type

import torch
from divevision.src.datasets.lsui_dataset import LSUIDataset
from divevision.src.metrics.abstract_metric import AbstractMetric
from divevision.src.metrics.psnr import PSNRMetric
from divevision.src.metrics.ssim import SSIMMetric
from divevision.src.models.abstract_model import AbstractModel
from divevision.src.models.u_shape_model import UShapeModelWrapper
from torch.utils.data import Dataset


def simple_test_routine(
    model: AbstractModel,
    dataset: Dataset,
    metric: Type[AbstractMetric],
) -> None:

    input, label = dataset[0]
    output = model.forward(input)

    input_img = model.postprocessing(input)
    output_img = model.postprocessing(output)
    label_img = model.postprocessing(label)

    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")
    input_img.save("input.jpg")
    output_img.save("output.jpg")
    label_img.save("label.jpg")

    val_metric = metric.compute(input, label)
    print(f"{metric.name} value = {val_metric.item()}")


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = UShapeModelWrapper(training=False, device=device)
    dataset = LSUIDataset(transform=model.preprocessing)
    metric = SSIMMetric()
    simple_test_routine(model, dataset, metric)
