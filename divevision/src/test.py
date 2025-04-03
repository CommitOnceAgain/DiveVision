import itertools
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Type

import mlflow
import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from divevision.src.datasets.lsui_dataset import LSUIDataset
from divevision.src.datasets.uieb_dataset import UIEBDataset
from divevision.src.metrics.abstract_metric import AbstractMetric
from divevision.src.metrics.psnr import PSNRMetric
from divevision.src.metrics.ssim import SSIMMetric
from divevision.src.models.abstract_model import AbstractModel
from divevision.src.models.cvae_model import CVAEModelWrapper
from divevision.src.models.u_shape_model import UShapeModelWrapper


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


def full_test_routine(
    models: list[AbstractModel],
    dataset_classes: list[type[Dataset]],
    metrics: list[Type[AbstractMetric]],
) -> None:

    # Save MLFlow experiment
    mlflow.set_experiment("Model testing")

    for model in models:
        # Retrieve the model device
        device = next(model.parameters()).device

        for dataset in dataset_classes:
            # Start a run per model and per dataset
            run = mlflow.start_run()

            # Instanciate dataset object
            data = dataset(transform=model.preprocessing)

            dataloader = DataLoader(
                data,
                batch_size=8,
                shuffle=False,  # For reproducibility
                num_workers=4,
                pin_memory=True,
            )
            metrics_val_list: dict[str, list[float]] = defaultdict(list)

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

            # Log information about the model and dataset used for testing as parameters
            mlflow.log_params(
                {
                    "model_name": model.model_name,
                    "dataset_name": data.dataset_name,
                }
            )

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

    models = [
        UShapeModelWrapper(device=device),
        CVAEModelWrapper(device=device),
    ]

    datasets_classes = [
        UIEBDataset,
        LSUIDataset,
    ]

    metrics = [
        SSIMMetric(),
        PSNRMetric(),
    ]

    full_test_routine(models, datasets_classes, metrics)
