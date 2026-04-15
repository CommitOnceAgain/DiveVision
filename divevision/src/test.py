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

from divevision.src.datasets.abstract_dataset import AbstractDataset
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


def run_benchmark(
    enhancement_models: list[AbstractModel],
    dataset_classes: list[type[AbstractDataset]],
    evaluation_metrics: list[AbstractMetric],
) -> None:

    # Set the MLflow Experiment that will group every Benchmark Run below
    mlflow.set_experiment("Model testing")

    for enhancement_model in enhancement_models:
        # Retrieve the Enhancement Model device
        device = next(enhancement_model.parameters()).device

        for dataset_class in dataset_classes:
            # Start a Benchmark Run per Enhancement Model and per Benchmark Dataset
            benchmark_run = mlflow.start_run()

            # Instanciate the Benchmark Dataset
            benchmark_dataset = dataset_class(transform=enhancement_model.preprocessing)

            dataloader = DataLoader(
                benchmark_dataset,
                batch_size=8,
                shuffle=False,  # For reproducibility
                num_workers=4,
                pin_memory=True,
            )
            evaluation_metric_values: dict[str, list[float]] = defaultdict(list)

            benchmark_run_start = time.process_time()
            # Iterate over the Benchmark Dataset
            for batch_step, (degraded_image, reference_image) in tqdm(
                enumerate(dataloader),
                desc="Iterating over the Benchmark Dataset...",
                unit="batch",
            ):
                # Infer an output from the Enhancement Model
                with torch.no_grad():
                    # Return fraction time (in seconds)
                    start = time.process_time()
                    # Forward pass
                    model_output: torch.Tensor = enhancement_model.forward(
                        degraded_image.to(device)
                    )
                    elapsed = time.process_time() - start

                    evaluation_metric_values["elapsed_s"].append(
                        [elapsed]
                    )  # Store as a list for compatiblity (see later use of itertools.chain.from_iterable)
                    # Compute Evaluation Metrics between the model output and Reference Image
                    for evaluation_metric in evaluation_metrics:
                        metric_value: torch.Tensor = evaluation_metric.compute(
                            model_output, reference_image
                        )
                        # Store metric values
                        evaluation_metric_values[evaluation_metric.name].append(
                            metric_value.tolist()
                        )

                    mlflow.log_metrics(
                        metrics=extract_last_metrics(
                            evaluation_metric_values, prefix="batch_"
                        ),
                        step=batch_step,
                        run_id=benchmark_run.info.run_id,
                    )

            benchmark_run_elapsed = time.process_time() - benchmark_run_start

            # Report all metrics
            aggregated_metric_values = {}
            for metric_name, values in evaluation_metric_values.items():
                # If batch size > 1, 'metrics' is composed of list of lists, and need to be flattened
                flattened_values = list(itertools.chain.from_iterable(values))
                aggregated_metric_values["global_" + metric_name] = np.mean(
                    flattened_values
                )

            # Update 'global_elapsed_s' with the correct time
            aggregated_metric_values["global_elapsed_s"] = benchmark_run_elapsed

            # Log information about the Enhancement Model and Benchmark Dataset used as parameters
            mlflow.log_params(
                {
                    "model_name": enhancement_model.name,
                    "dataset_name": benchmark_dataset.name,
                }
            )

            mlflow.log_metrics(
                metrics=aggregated_metric_values,
                run_id=benchmark_run.info.run_id,
            )

            # End the Benchmark Run properly
            mlflow.end_run()


if __name__ == "__main__":

    mlflow.set_tracking_uri(
        uri=f"http://{os.environ["MLFLOW_HOST"]}:{os.environ["MLFLOW_PORT"]}"
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    enhancement_models = [
        UShapeModelWrapper(device=device),
        CVAEModelWrapper(device=device),
    ]

    dataset_classes = [
        UIEBDataset,
        LSUIDataset,
    ]

    evaluation_metrics = [
        SSIMMetric(),
        PSNRMetric(),
    ]

    run_benchmark(enhancement_models, dataset_classes, evaluation_metrics)
