import itertools
from typing import Type

import torch
from tqdm import tqdm
from divevision.src.datasets.lsui_dataset import LSUIDataset
from divevision.src.metrics.abstract_metric import AbstractMetric
from divevision.src.metrics.psnr import PSNRMetric
from divevision.src.metrics.ssim import SSIMMetric
from divevision.src.models.abstract_model import AbstractModel
from divevision.src.models.u_shape_model import UShapeModelWrapper
from torch.utils.data import Dataset, DataLoader


def simple_test_routine(
    model: AbstractModel,
    dataset: Dataset,
    metric: Type[AbstractMetric],
) -> None:

    # Retrieve the model device
    device = next(model.parameters()).device

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    metric_val_list: list[float] = []

    # Iterate over the dataset
    for input, label in tqdm(
        dataloader,
        desc="Iterating over the test dataset...",
        unit="batch",
    ):
        # Infer an output from the model
        with torch.no_grad():
            # Forward pass
            output: torch.Tensor = model.forward(input.to(device))

            # Compute metric between model output and label
            val_metric: torch.Tensor = metric.compute(output, label)

        # Store metric values
        metric_val_list.append(val_metric.tolist())

    # If batch size > 1, 'metrics' is composed of list of lists, and need to be flattened
    flatten_metrics = list(itertools.chain.from_iterable(metric_val_list))
    print(f"{metric.name} value = {sum(flatten_metrics) / len(flatten_metrics)}")


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = UShapeModelWrapper(device=device)
    dataset = LSUIDataset(transform=model.preprocessing)
    metric = SSIMMetric()
    simple_test_routine(model, dataset, metric)
