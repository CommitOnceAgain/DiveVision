import torch
from skimage.metrics import structural_similarity as ssim

from divevision.src.metrics.abstract_metric import AbstractMetric


class SSIMMetric(AbstractMetric):
    """Structural Similarity Index Metric (SSIM)"""

    name = "SSIM"

    def compute(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the Structural Similarity Index Metric (SSIM) between two images. Handles batched data."""
        # Check that both tensors have the same shape
        assert input.ndim == target.ndim

        # Add batch dimension if they are not batched, for computation compatibility
        if input.ndim == 3:
            input = torch.unsqueeze(input, dim=0)
            target = torch.unsqueeze(target, dim=0)

        # Compute the SSIM metric for pair of item of the batch
        return torch.tensor(
            [
                ssim(
                    input[idx].detach().cpu().numpy(),
                    target[idx].detach().cpu().numpy(),
                    channel_axis=0,
                    data_range=1,
                )
                for idx in range(input.shape[0])
            ]
        )
