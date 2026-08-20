import torch
from skimage.metrics import structural_similarity as ssim

from divevision.src.metrics.abstract_metric import AbstractMetric


class SSIMMetric(AbstractMetric):
    """Structural Similarity Index Metric (SSIM)"""

    name = "SSIM"

    def compute(
        self, enhanced_image: torch.Tensor, reference_image: torch.Tensor
    ) -> torch.Tensor:
        """Compute the Structural Similarity Index Metric (SSIM) between an Enhanced Image and its Reference Image. Handles batched data."""
        # Check that both tensors have the same shape
        assert enhanced_image.ndim == reference_image.ndim

        # Add batch dimension if they are not batched, for computation compatibility
        if enhanced_image.ndim == 3:
            enhanced_image = torch.unsqueeze(enhanced_image, dim=0)
            reference_image = torch.unsqueeze(reference_image, dim=0)

        # Compute the SSIM metric for pair of item of the batch
        return torch.tensor(
            [
                ssim(
                    enhanced_image[idx].detach().cpu().numpy(),
                    reference_image[idx].detach().cpu().numpy(),
                    channel_axis=0,
                    data_range=1,
                )
                for idx in range(enhanced_image.shape[0])
            ]
        )
