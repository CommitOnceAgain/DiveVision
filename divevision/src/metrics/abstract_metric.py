from abc import ABC, abstractmethod
from typing import Final

import torch


class AbstractMetric(ABC):
    """Abstract class for metrics"""

    name: Final[str]

    @abstractmethod
    def compute(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the metric between input and target"""
        raise NotImplementedError
