import torch
from divevision.src.metrics.abstract_metric import AbstractMetric
from skimage.metrics import structural_similarity as ssim


class SSIMMetric(AbstractMetric):
    """Structural Similarity Index Metric (SSIM) metric"""

    name = "SSIM"

    def compute(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the SSIM metric between two images"""
        return torch.tensor(
            ssim(
                input.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                channel_axis=0,
                data_range=1,
            )
        )
