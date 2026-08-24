from abc import ABC, abstractmethod
from typing import Final

import torch

from . import _metric_registry


class AbstractMetric(ABC):
    """Abstract class for Evaluation Metrics"""

    name: str

    def __init_subclass__(cls):
        # Register subclasses in the global registry when they are defined
        if cls not in _metric_registry:
            _metric_registry[cls.name] = cls()

    @abstractmethod
    def compute(
        self, enhanced_image: torch.Tensor, reference_image: torch.Tensor
    ) -> torch.Tensor:
        """Compute the Evaluation Metric between an Enhanced Image and its Reference Image"""
        raise NotImplementedError
