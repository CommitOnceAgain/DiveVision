import torch
from skimage.metrics import peak_signal_noise_ratio as psnr

from divevision.src.metrics.abstract_metric import AbstractMetric


class PSNRMetric(AbstractMetric):
    """Peak signal to noise ratio metric"""

    name = "PSNR"

    def compute(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the Peak Signal-to-Noise Ratio between two images. Handles batched data."""
        # Check that both tensors have the same shape
        assert input.ndim == target.ndim

        # Add batch dimension if they are not batched, for computation compatibility
        if input.ndim == 3:
            input = torch.unsqueeze(input, dim=0)
            target = torch.unsqueeze(target, dim=0)

        # Compute the PSNR metric for pair of item of the batch
        return torch.tensor(
            [
                psnr(
                    input[idx].detach().cpu().numpy(),
                    target[idx].detach().cpu().numpy(),
                    data_range=1,
                )
                for idx in range(input.shape[0])
            ]
        )
