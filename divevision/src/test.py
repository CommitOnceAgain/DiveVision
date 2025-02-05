from collections import defaultdict
import itertools
from pathlib import Path
from typing import Callable, Type
import time

import numpy as np
import torch
from tqdm import tqdm
from divevision.src.datasets.lsui_dataset import LSUIDataset
from divevision.src.metrics.abstract_metric import AbstractMetric
from divevision.src.metrics.psnr import PSNRMetric
from divevision.src.metrics.ssim import SSIMMetric
from divevision.src.models.abstract_model import AbstractModel
from divevision.src.models.u_shape_model import UShapeModelWrapper
from torch.utils.data import Dataset, DataLoader
import mlflow
from dotenv import load_dotenv
import os


def extract_last_metrics(
    metrics: defaultdict[list],
    prefix: str = "",
    agg_fun: Callable = np.mean,
) -> dict:
    output_dict = {}
    # Iterate over metrics
    for metric_name, listed_values in metrics.items():
        # Extract the last value of the list
        val = listed_values[-1]
        if (
            type(val) is list
        ):  # If the value is a list of values, aggregate them into a single value
            output_dict[prefix + metric_name] = agg_fun(val)
        else:  # Or store the value directly
            output_dict[prefix + metric_name] = val

    return output_dict


def simple_test_routine(
    model: AbstractModel,
    dataset: Dataset,
    metrics: list[Type[AbstractMetric]],
) -> None:

    # Retrieve the model device
    device = next(model.parameters()).device

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,  # For reproducibility
        num_workers=4,
        pin_memory=True,
    )
    metrics_val_list: dict[str, list[float]] = defaultdict(list)
    # Save MLFlow experiment
    mlflow.set_experiment("Model testing")
    # Start a run
    run = mlflow.start_run()

    start_testing_loop = time.process_time()
    # Iterate over the dataset
    for batch_step, (input, label) in tqdm(
        enumerate(dataloader),
        desc="Iterating over the test dataset...",
        unit="batch",
    ):
        # Infer an output from the model
        with torch.no_grad():
            # Return fraction time (in seconds)
            start = time.process_time()
            # Forward pass
            output: torch.Tensor = model.forward(input.to(device))
            elapsed = time.process_time() - start

            metrics_val_list["elapsed_s"].append(
                [elapsed]
            )  # Store as a list for compatiblity (see later use of itertools.chain.from_iterable)
            # Compute metrics between model output and label
            for metric in metrics:
                val_metric: torch.Tensor = metric.compute(output, label)
                # Store metric values
                metrics_val_list[metric.name].append(val_metric.tolist())

            mlflow.log_metrics(
                metrics=extract_last_metrics(metrics_val_list, prefix="batch_"),
                step=batch_step,
                run_id=run.info.run_id,
            )

    testing_loop_elapsed = time.process_time() - start_testing_loop

    # Report all metrics
    metrics_final = {}
    for metric_name, values in metrics_val_list.items():
        # If batch size > 1, 'metrics' is composed of list of lists, and need to be flattened
        flatten_values = list(itertools.chain.from_iterable(values))
        metrics_final["global_" + metric_name] = np.mean(flatten_values)

    # Update 'global_elapsed_s' with the correct time
    metrics_final["global_elapsed_s"] = testing_loop_elapsed
    mlflow.log_metrics(
        metrics=metrics_final,
        run_id=run.info.run_id,
    )

    # End run properly
    mlflow.end_run()


if __name__ == "__main__":
    # Load environment variables from .env file, returns True if at least one environment variable is set
    assert load_dotenv(Path(".env").resolve())

    mlflow.set_tracking_uri(
        uri=f"http://{os.environ["MLFLOW_HOST"]}:{os.environ["MLFLOW_PORT"]}"
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = UShapeModelWrapper(device=device)
    dataset = LSUIDataset(transform=model.preprocessing)
    metrics = [SSIMMetric(), PSNRMetric()]
    simple_test_routine(model, dataset, metrics)
