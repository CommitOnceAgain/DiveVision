import torch
from divevision.src.metrics.abstract_metric import AbstractMetric
from skimage.metrics import peak_signal_noise_ratio as psnr


class PSNRMetric(AbstractMetric):
    """Peak signal to noise ratio metric"""

    name = "PSNR"

    def compute(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return psnr(
            input.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
        )
